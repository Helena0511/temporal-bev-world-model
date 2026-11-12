"""Analytical geometry and data-contract tests for the native training path."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import torch
from torch import nn

from resworld.driving_data import pose_matrix, DrivingDataset, iter_json_rows
from resworld.driving_model import CameraLiftSplat, DrivingResWorld, ModelConfig, planning_loss
from scripts.train_driving import trajectory_metrics


class OnePixel(nn.Module):
    def forward(self, images):
        return images.new_ones(images.shape[0], 4, 1, 1)


class DrivingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_quaternion_rotation_and_inverse(self):
        root_half = 2 ** -.5
        pose = pose_matrix(dict(rotation=[root_half, 0, 0, root_half], translation=[2, 3, 0]))
        point = torch.tensor([1., 0., 0., 1.], dtype=torch.float64)
        torch.testing.assert_close(pose @ point, torch.tensor([2., 4., 0., 1.], dtype=torch.float64))
        torch.testing.assert_close(torch.linalg.solve(pose, pose @ point), point)

    def test_streaming_metadata_across_chunk_boundary(self):
        rows = [{'token': str(i), 'payload': 'x'*1000} for i in range(1100)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'table.json'
            path.write_text(json.dumps(rows))
            self.assertEqual(list(iter_json_rows(path)), rows)

    def test_depth_supervision_uses_nearest_visible_surface(self):
        dataset = object.__new__(DrivingDataset)
        dataset.image_size = (16, 16)
        dataset.depth_bins, dataset.depth_min, dataset.depth_max = 5, 1., 5.
        points = torch.tensor([[0., 0., 2., 1.], [0., 0., 4., 1.],
                               [0., 0., -2., 1.], [100., 0., 2., 1.]], dtype=torch.float64)
        target = dataset.depth_target(points, torch.eye(4, dtype=torch.float64), torch.eye(3))
        self.assertEqual(target[0, 0].item(), 1)  # depth=2 -> bin=1
        self.assertEqual((target >= 0).sum().item(), 1)

    def test_current_and_state_loading_preserve_geometry_and_avoid_unused_images(self):
        from resworld.nuscenes_io import CAMERAS
        with tempfile.TemporaryDirectory() as directory:
            Image.new('RGB', (16, 16), (32, 64, 96)).save(Path(directory)/'frame.png')
            dataset = object.__new__(DrivingDataset)
            dataset.root, dataset.image_size = Path(directory), (16, 16)
            dataset.depth_bins, dataset.depth_min, dataset.depth_max = 4, 1., 10.
            dataset.entries = ['current']
            tokens = ['old', 'previous', 'current']
            poses = {t:dict(rotation=[1., 0., 0., 0.], translation=[float(i), 0., 0.])
                     for i,t in enumerate(tokens)}
            calibration = dict(rotation=[1., 0., 0., 0.], translation=[0., 0., 1.],
                               camera_intrinsic=[[8., 0., 8.], [0., 8., 8.], [0., 0., 1.]])
            dataset.calibrations, dataset.poses = {'calib':calibration}, poses
            dataset.samples = {t:dict(timestamp=(i+1)*500000, scene_token='scene') for i,t in enumerate(tokens)}
            dataset.scene_names = {'scene':'test-scene'}
            dataset.records = {t:{c:dict(filename='frame.png', calibrated_sensor_token='calib', ego_pose_token=t)
                                   for c in CAMERAS} for t in tokens}
            dataset.history_tokens = lambda _: tokens
            dataset.reference_pose = lambda t: poses[t]
            dataset.lidar_global_points = lambda _: None
            dataset.labels = lambda _: (torch.zeros(6, 2), torch.ones(6, dtype=torch.bool), 1)
            dataset.input_mode = 'temporal'
            with patch('resworld.driving_data.Image.open', wraps=Image.open) as opened:
                temporal = dataset[0]
                self.assertEqual(opened.call_count, 18)
            dataset.input_mode = 'current'
            with patch('resworld.driving_data.Image.open', wraps=Image.open) as opened:
                current = dataset[0]
                self.assertEqual(opened.call_count, 6)
            dataset.input_mode = 'state'
            with patch('resworld.driving_data.Image.open', side_effect=AssertionError('state loading read RGB')):
                state = dataset[0]
            self.assertEqual(state['images'].numel(), 0)
            torch.testing.assert_close(current['images'], temporal['images'][-1:])
            for key in ('motion', 'history_dt', 'current2history', 'gt_positions', 'future_valid', 'command'):
                torch.testing.assert_close(current[key], temporal[key], rtol=0, atol=0)
                torch.testing.assert_close(state[key], temporal[key], rtol=0, atol=0)

    def test_analytical_camera_ray_splats_to_metric_cell(self):
        cfg = ModelConfig(channels=4, depth_bins=1, depth_min=5., depth_max=5.,
                          x_min=-10, x_max=10, y_min=-10, y_max=10,
                          bev_height=21, bev_width=21)
        encoder = CameraLiftSplat(cfg)
        encoder.backbone = OnePixel()
        encoder.context_head = nn.Identity()
        encoder.refine = nn.Identity()
        transform = torch.eye(4)[None, None]
        # Camera right/down/forward -> ego -left/-up/forward.
        transform[0, 0, :3, :3] = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        transform[0, 0, :3, 3] = torch.tensor([0., 2., 1.5])
        bev, _ = encoder(torch.zeros(1, 1, 3, 8, 8), torch.eye(3)[None, None], transform)
        # Principal ray has camera xyz=(0,0,5), ego xyz=(5,2,1.5).
        expected = torch.zeros_like(bev)
        expected[:, :, 5, 8] = 1
        torch.testing.assert_close(bev, expected)

    def test_masked_losses_ignore_padded_future(self):
        initial = torch.zeros(1, 3, 3, 2, requires_grad=True)
        final = torch.zeros_like(initial, requires_grad=True)
        outputs = dict(init_deltas=initial, final_deltas=final,
                       depth_logits=torch.zeros(1, 3, 6, 4, 1, 1, requires_grad=True))
        batch = dict(gt_positions=torch.tensor([[[1., 0.], [2., 0.], [999., 999.]]]),
                     future_valid=torch.tensor([[True, True, False]]), command=torch.tensor([1]))
        terms = planning_loss(outputs, batch)
        torch.testing.assert_close(terms['planning'], torch.tensor(1.))
        terms['total'].backward()
        self.assertTrue((final.grad[:, :, 2] == 0).all())
        self.assertTrue((final.grad[:, [0, 2]] == 0).all())
        self.assertGreater(final.grad[:, 1, :2].abs().sum().item(), 0)

    def test_metrics_exclude_missing_horizons(self):
        batch = dict(gt_positions=torch.zeros(2, 6, 2), future_valid=torch.tensor(
            [[True]*6, [True, True, False, False, False, False]]))
        prediction = torch.ones(2, 6, 2)
        result = trajectory_metrics(prediction, batch)
        self.assertEqual(result['ade_count'], 8)
        self.assertEqual(result['l2_1s_count'], 2)
        self.assertEqual(result['l2_3s_count'], 1)
        self.assertAlmostEqual(result['fde_sum'], 2**.5, places=5)

    def test_identical_history_has_zero_residual_and_invisible_history_is_masked(self):
        cfg = ModelConfig(channels=8, tokens=2, bev_height=8, bev_width=8,
                          depth_bins=4, future_steps=2)
        model = DrivingResWorld(cfg).eval()
        frame = torch.rand(1, 1, 6, 3, 16, 16)
        k = torch.tensor([[8., 0., 8.], [0., 8., 8.], [0., 0., 1.]])
        batch = dict(images=frame.expand(-1, 3, -1, -1, -1, -1).contiguous(),
                     intrinsics=k.expand(1, 3, 6, 3, 3),
                     camera_to_ego=torch.eye(4).expand(1, 3, 6, 4, 4),
                     motion=torch.zeros(1, 3, 3), command=torch.tensor([1]),
                     current2history=torch.eye(3).expand(1, 2, 3, 3).clone())
        with torch.no_grad():
            output = model(batch)
            torch.testing.assert_close(output['residual_tokens'], torch.zeros_like(output['residual_tokens']), atol=1e-6, rtol=0)
            batch['current2history'][:, :, 0, 2] = 1000
            output = model(batch)
        self.assertFalse(output['overlap_mask'].any())
        self.assertTrue(torch.isfinite(output['final_traj']).all())
        self.assertEqual(output['residual_weights'].abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
