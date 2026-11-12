"""Geometry, visibility, and temporal-residual contracts of the new perception."""
import unittest

import torch
from torch import nn

from resworld.bev_attention import SpatialCrossAttention, TemporalBEVAttention, project_anchors
from resworld.driving_model import DrivingResWorld, ModelConfig
from resworld.temporal_alignment import BEVGrid


class BEVAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_known_ego_point_projects_to_principal_pixel(self):
        transform = torch.eye(4)[None, None]
        transform[0, 0, :3, :3] = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        transform[0, 0, :3, 3] = torch.tensor([0., 2., 1.5])
        intrinsics = torch.tensor([[10., 0., 8.], [0., 10., 8.], [0., 0., 1.]])[None, None]
        points = torch.tensor([[[5., 2., 1.5]], [[-5., 2., 1.5]]])
        uv, depth = project_anchors(points, intrinsics, transform)
        torch.testing.assert_close(uv[0, 0, 0, 0], torch.tensor([8., 8.]))
        self.assertEqual(depth[0, 0, 0, 0].item(), 5)
        self.assertLess(depth[0, 0, 1, 0].item(), 0)

    def test_invisible_cameras_cannot_affect_spatial_query(self):
        layer = SpatialCrossAttention(8, 2, 2, 2, chunk_size=2).eval()
        query = torch.randn(1, 3, 8)
        features = [torch.randn(1, 6, 8, 4, 4), torch.randn(1, 6, 8, 2, 2)]
        uv = torch.zeros(1, 6, 3, 2, 2)
        visible = torch.zeros(1, 6, 3, 2, dtype=torch.bool)
        first, mask = layer(query, features, uv, visible, (8, 16))
        second, _ = layer(query, [f*1000 for f in features], uv, visible, (8, 16))
        torch.testing.assert_close(first, second)
        self.assertFalse(mask.any())
        self.assertTrue(torch.isfinite(first).all())

    def test_spatial_sampling_has_image_and_offset_gradients(self):
        torch.manual_seed(42)
        layer = SpatialCrossAttention(8, 2, 2, 2, chunk_size=2)
        query = torch.randn(1, 3, 8)
        features = [torch.randn(1, 6, 8, 4, 4, requires_grad=True),
                    torch.randn(1, 6, 8, 2, 2, requires_grad=True)]
        uv = torch.full((1, 6, 3, 2, 2), 12.)
        visible = torch.ones(1, 6, 3, 2, dtype=torch.bool)
        output, mask = layer(query, features, uv, visible, (8, 16))
        (output * torch.randn_like(output)).sum().backward()
        self.assertTrue(mask.all())
        for gradient in [features[0].grad, features[1].grad, layer.offsets.weight.grad,
                         layer.weights.weight.grad, layer.view_weights.weight.grad]:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_invalid_history_does_not_leak_into_temporal_attention(self):
        layer = TemporalBEVAttention(8, 2, chunk_size=5).eval()
        current = torch.randn(1, 8, 4, 4)
        masks = [torch.zeros(1, 1, 4, 4, dtype=torch.bool)]*2 + [torch.ones(1, 1, 4, 4, dtype=torch.bool)]
        grid = BEVGrid(-2, 2, -2, 2, 4, 4)
        a = layer(current, [current*0, current*0, current], masks, grid)
        b = layer(current, [current*1000, current*-999, current], masks, grid)
        torch.testing.assert_close(a, b)

    def test_temporal_search_radius_is_consistent_in_meters(self):
        layer = TemporalBEVAttention(4, 1, points=1, max_offset_meters=4.)
        layer.output, layer.norm, layer.ffn_norm = nn.Identity(), nn.Identity(), nn.Identity()
        layer.ffn = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            layer.ffn.weight.zero_()
            layer.weights.weight.zero_()
            layer.weights.bias.zero_()
            # Normalized row offset +0.5 radius -> 2m backwards in ego x.
            bias = torch.zeros(1, 3, 1, 2)
            bias[..., 1] = torch.atanh(torch.tensor(.5))
            layer.offsets.bias.copy_(bias.flatten())
        for resolution in (9, 17):
            grid = BEVGrid(-4, 4, -4, 4, resolution, resolution)
            points = grid.target_metric_points(1, 'cpu', torch.float32)
            current = points[..., 0][:, None].expand(-1, 4, -1, -1)
            masks = [torch.ones(1, 1, resolution, resolution, dtype=torch.bool)]*3
            output = layer(current, [current]*3, masks, grid)
            torch.testing.assert_close(output[0, :, resolution//2, resolution//2], torch.full((4,), -2.))

    def test_visual_residual_is_not_contaminated_by_motion_condition(self):
        cfg = ModelConfig(channels=8, tokens=2, heads=2, future_steps=2,
                          bev_height=6, bev_width=6, depth_bins=4,
                          x_min=-2, x_max=10, y_min=-4, y_max=4,
                          perception='bev_query', temporal_fusion='attention',
                          residual_source='visual', query_chunk_size=8)
        model = DrivingResWorld(cfg).eval()
        frame = torch.randn(1, 1, 6, 3, 32, 32)
        k = torch.tensor([[16., 0., 16.], [0., 16., 16.], [0., 0., 1.]])
        transform = torch.eye(4).repeat(1, 3, 6, 1, 1)
        transform[..., :3, :3] = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        transform[..., 2, 3] = 1.5
        batch = dict(images=frame.expand(-1, 3, -1, -1, -1, -1).contiguous(),
                     intrinsics=k.expand(1, 3, 6, 3, 3), camera_to_ego=transform,
                     current2history=torch.eye(3).expand(1, 2, 3, 3),
                     motion=torch.randn(1, 3, 3), command=torch.tensor([1]))
        with torch.no_grad():
            output = model(batch)
        self.assertTrue(output['overlap_mask'].any())
        torch.testing.assert_close(output['residual_tokens'], torch.zeros_like(output['residual_tokens']), atol=1e-6, rtol=0)
        self.assertTrue(torch.isfinite(output['final_traj']).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
