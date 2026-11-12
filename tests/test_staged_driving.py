"""Modality isolation, stage boundaries and independently known target geometry."""
from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from resworld.driving_model import DrivingResWorld, ModelConfig
from resworld.perception_targets import footprint_target
from resworld.staged_driving import EgoTrajectoryBranch, StagedDrivingModel, staged_loss
from scripts.train_staged_driving import transfer_checkpoint


class StagedDrivingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def config(self, **kwargs):
        return ModelConfig(channels=8, tokens=2, heads=2, bev_height=9, bev_width=9,
                           depth_bins=4, x_min=-4, x_max=4, y_min=-4, y_max=4, **kwargs)

    def batch(self):
        k = torch.tensor([[8., 0., 8.], [0., 8., 8.], [0., 0., 1.]])
        extrinsic = torch.eye(4)
        extrinsic[:3, :3] = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        extrinsic[2, 3] = 1.5
        transform = torch.eye(3).repeat(1, 2, 1, 1)
        transform[:, 0, 0, 2] = 4
        transform[:, 1, 0, 2] = 2
        return dict(images=torch.rand(1, 3, 6, 3, 16, 16),
            intrinsics=k.repeat(1, 3, 6, 1, 1), camera_to_ego=extrinsic.repeat(1, 3, 6, 1, 1),
            camera_to_ego_static=extrinsic.repeat(1, 3, 6, 1, 1),
            current2history=transform, history_dt=torch.tensor([.5]), motion=torch.rand(1, 3, 3),
            command=torch.tensor([1]), gt_positions=torch.zeros(1, 6, 2),
            future_valid=torch.ones(1, 6, dtype=torch.bool),
            depth_targets=torch.ones(1, 3, 6, 2, 2, dtype=torch.long),
            bev_target=torch.zeros(1, 9, 9, dtype=torch.long))

    def test_no_explicit_state_does_not_read_motion_or_time(self):
        model = DrivingResWorld(self.config(use_ego_state=False)).eval()
        batch = self.batch()
        first = model(batch)['final_traj']
        del batch['motion'], batch['history_dt']
        second = model(batch)['final_traj']
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.config(use_ego_state=False, kinematic_prior=True)

    def test_strict_visual_does_not_read_past_images_poses_or_compensated_calibration(self):
        model = DrivingResWorld(self.config(use_ego_state=False, temporal_context='single_frame',
                                            use_world_delta=False)).eval()
        batch = self.batch()
        first = model(batch)
        batch['images'][:, :2] = float('nan')
        del batch['motion'], batch['current2history'], batch['history_dt'], batch['camera_to_ego']
        second = model(batch)
        torch.testing.assert_close(first['final_traj'], second['final_traj'], rtol=0, atol=0)
        self.assertEqual(second['depth_logits'].shape[1], 1)
        self.assertEqual(second['temporal_residual_bev'].abs().sum().item(), 0)
        # Verify that the isolated branch still has a usable image gradient.
        model.zero_grad()
        second['final_traj'].sum().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.camera_bev.parameters()
                               if p.grad is not None), 0)

    def test_ego_only_needs_no_camera_input_and_starts_at_constant_velocity(self):
        model = EgoTrajectoryBranch(8, 6)
        batch = self.batch()
        output = model({k: batch[k] for k in ('motion', 'history_dt', 'current2history')})
        expected = torch.stack((torch.arange(1, 7)*2., torch.zeros(6)), -1)
        torch.testing.assert_close(output['final_traj'][0, 1], expected)

    def test_frozen_encoders_stay_fixed_while_fusion_learns(self):
        model = StagedDrivingModel(self.config(), 'fusion', ego_dropout=0)
        batch = self.batch()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
        before = model.fusion_head[-1].weight.clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            staged_loss(model(batch), batch, 'fusion')['total'].backward()
            optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
                self.assertIsNone(parameter.grad)
        self.assertFalse(torch.equal(before, model.fusion_head[-1].weight))
        model.set_stage('joint')
        self.assertTrue(all(p.requires_grad for p in model.ego.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.visual.camera_bev.parameters()))

    def test_perception_pretraining_reaches_spatial_attention(self):
        model = StagedDrivingModel(self.config(perception='bev_query'), 'perception')
        loss = staged_loss(model(self.batch()), self.batch(), 'perception')['total']
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.visual.camera_bev.spatial.parameters()
                               if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in model.ego.parameters()))
        self.assertTrue(all(p.grad is None for p in model.visual.init_head.parameters()))

    def test_fusion_initialization_and_complete_explicit_state_dropout(self):
        model = StagedDrivingModel(self.config(), 'fusion', ego_dropout=.3).eval()
        batch = self.batch()
        torch.testing.assert_close(model(batch)['final_traj'], model.ego(batch)['final_traj'])
        # Once learned, both the ego embedding and ego trajectory must be masked.
        torch.nn.init.normal_(model.fusion_head[-1].weight)
        torch.nn.init.normal_(model.ego.head.weight)
        model.train()
        with patch('torch.rand_like', side_effect=lambda x: torch.zeros_like(x)):
            first = model(batch)['final_traj']
            batch['motion'] += 100
            second = model(batch)['final_traj']
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_transfer_checks_provenance_before_loading(self):
        model = StagedDrivingModel(self.config(), 'perception')
        manifest = dict(model_config=asdict(model.config), version='v1.0-mini', selection_sha256='a',
                        image_size=[16, 16], split_sha256='b', stage='perception')
        checkpoint = dict(model=model.state_dict(), manifest=manifest)
        other = StagedDrivingModel(self.config(), 'fusion')
        transfer_checkpoint(other, checkpoint, manifest, 'visual')
        torch.testing.assert_close(other.visual.init_head.weight, model.visual.init_head.weight)
        # Historical checkpoints predate this field, whose canonical default is tiny.
        legacy={**manifest,'model_config':dict(manifest['model_config'])}
        legacy['model_config'].pop('image_backbone')
        transfer_checkpoint(other,{**checkpoint,'manifest':legacy},manifest,'visual')
        incompatible={**manifest,'model_config':{**manifest['model_config'],'image_backbone':'resnet50',
                                                'perception':'bev_query'}}
        with self.assertRaisesRegex(ValueError,'model_config'):
            transfer_checkpoint(other,checkpoint,incompatible,'visual')
        changed = {**manifest, 'selection_sha256':'different'}
        with self.assertRaisesRegex(ValueError, 'selection_sha256'):
            transfer_checkpoint(other, checkpoint, changed, 'visual')
        with self.assertRaisesRegex(ValueError, 'ego-only'):
            transfer_checkpoint(other, checkpoint, manifest, 'ego')

    def test_state_transfer_across_camera_configs_preserves_visual_and_checks_state_contract(self):
        source = StagedDrivingModel(self.config(), 'ego')
        manifest = dict(model_config=asdict(source.config), version='v1.0-mini',
                        selection_sha256='a', split_sha256='b', image_size=[16, 16], stage='ego')
        with torch.no_grad():
            source.ego.head.bias.fill_(.25)
        checkpoint = dict(model=source.state_dict(), manifest=manifest)
        target = StagedDrivingModel(self.config(perception='bev_query', image_backbone='resnet50'), 'fusion')
        requested = {**manifest, 'stage':'fusion', 'model_config':asdict(target.config),
                     'image_size':[32, 64]}
        camera_before = target.visual.camera_bev.backbone.body.conv1.weight.clone()
        transfer_checkpoint(target, checkpoint, requested, 'ego')
        batch = {key:value for key,value in self.batch().items()
                 if key in ('motion', 'history_dt', 'current2history')}
        torch.testing.assert_close(target.ego(batch)['final_traj'], source.ego(batch)['final_traj'],
                                   rtol=0, atol=0)
        torch.testing.assert_close(target.visual.camera_bev.backbone.body.conv1.weight,
                                   camera_before, rtol=0, atol=0)
        before = {name:value.clone() for name,value in target.ego.state_dict().items()}
        for key in ('version', 'selection_sha256', 'split_sha256'):
            with self.assertRaisesRegex(ValueError, key):
                transfer_checkpoint(target, checkpoint, {**requested, key:'changed'}, 'ego')
        for key in ('channels', 'future_steps'):
            changed = {**requested, 'model_config':{**requested['model_config'],
                       key:requested['model_config'][key]*2}}
            with self.assertRaisesRegex(ValueError, 'model_config'):
                transfer_checkpoint(target, checkpoint, changed, 'ego')
        for name,value in target.ego.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_footprints_use_width_length_yaw_and_ignore_other_categories(self):
        pose = dict(rotation=[1., 0., 0., 0.], translation=[10., 20., 0.])
        box = dict(rotation=[1., 0., 0., 0.], translation=[12., 20., 0.], size=[2., 4., 1.], label=1)
        grid = self.config().grid()
        target = footprint_target([box], pose, grid)
        self.assertEqual(target[grid.metric_to_row_col(4., 1.)].item(), 1)
        self.assertEqual(target[grid.metric_to_row_col(2., 2.)].item(), 0)
        root_half = 2**-.5
        box['rotation'] = [root_half, 0., 0., root_half]
        box['label'] = -1
        target = footprint_target([box], pose, grid)
        self.assertEqual(target[grid.metric_to_row_col(2., 2.)].item(), 255)
        self.assertEqual(target[grid.metric_to_row_col(4., 0.)].item(), 0)


if __name__ == '__main__':
    unittest.main()
