"""Deterministic checks for the manually implemented FGTR sampling step."""

import unittest

import torch

from resworld.fgtr import (
    FutureGuidedTrajectoryRefinement,
    WaypointDeformableSampler,
    sample_future_waypoints,
    sample_waypoint_features,
)


class SamplingTests(unittest.TestCase):
    def test_corners_and_center(self):
        bev = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
        references = torch.tensor([[[-1., -1.], [0., 0.], [1., 1.]]])
        sampled = sample_waypoint_features(bev, references)
        self.assertEqual(tuple(sampled.shape), (1, 3, 1))
        torch.testing.assert_close(sampled[0, :, 0], torch.tensor([0., 12., 24.]))

    def test_bilinear_interpolation(self):
        bev = torch.tensor([[[[0., 2.], [4., 6.]]]])
        sampled = sample_waypoint_features(bev, torch.tensor([[[0., 0.]]]))
        # At the center, all four corner pixels contribute equally.
        torch.testing.assert_close(sampled, torch.tensor([[[3.]]]))

    def test_batches_channels_and_non_square_grid(self):
        bev = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        bev = torch.cat((bev, bev + 10), dim=1)
        bev = torch.cat((bev, bev + 100), dim=0)
        references = torch.tensor([
            [[-1., -1.], [1., 1.]],
            [[1., -1.], [-1., 1.]],
        ])
        sampled = sample_waypoint_features(bev, references)
        torch.testing.assert_close(sampled, torch.tensor([
            [[0., 10.], [5., 15.]],
            [[102., 112.], [103., 113.]],
        ]))

    def test_offset_sampling_and_weighted_sum(self):
        # These are fixed inputs, not learned offset/weight heads.  A 3x5
        # raster checks that horizontal and vertical scales are not swapped.
        bev = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
        center = torch.tensor([[[0., 0.]]])
        pixel_offsets = torch.tensor([[[[-1., 0.], [1., 0.]]]])
        scale = torch.tensor([2. / (5 - 1), 2. / (3 - 1)])
        locations = center.unsqueeze(2) + pixel_offsets * scale
        sampled = sample_waypoint_features(bev, locations.reshape(1, 2, 2))
        sampled = sampled.reshape(1, 1, 2, 1)
        torch.testing.assert_close(sampled, torch.tensor([[[[6.], [8.]]]]))
        weights = torch.tensor([[[0., 0.]]]).softmax(dim=-1)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 1))
        context = (sampled * weights.unsqueeze(-1)).sum(dim=2)
        torch.testing.assert_close(context, torch.tensor([[[7.]]]))

    def test_gradients_to_bev_and_sample_coordinates(self):
        bev = torch.tensor([[[[0., 2.], [4., 6.]]]], requires_grad=True)
        references = torch.tensor([[[0., 0.]]], requires_grad=True)
        sample_waypoint_features(bev, references).sum().backward()
        torch.testing.assert_close(bev.grad, torch.full_like(bev, 0.25))
        # This raster's horizontal slope is 1 and vertical slope is 2 in
        # normalized grid coordinates, so offsets can receive useful gradients.
        torch.testing.assert_close(references.grad, torch.tensor([[[1., 2.]]]))


class DeformableSamplerTests(unittest.TestCase):
    def make_sampler(self):
        sampler = WaypointDeformableSampler(3, num_points=2, max_offset_pixels=2.0)
        with torch.no_grad():
            for parameter in sampler.parameters():
                parameter.zero_()
        return sampler

    def test_bounded_offsets_and_border_padding(self):
        sampler = self.make_sampler()
        with torch.no_grad():
            sampler.offset_head.bias.copy_(torch.tensor([100., -100., 100., -100.]))
        bev = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
        reference = torch.tensor([[[1., -1.]]])
        context, weights, locations = sampler(torch.zeros(1, 1, 3), bev, reference)
        pixel_offsets = (locations - reference.unsqueeze(2)) / torch.tensor([0.5, 1.])
        torch.testing.assert_close(pixel_offsets, torch.tensor([[[[2., -2.], [2., -2.]]]]))
        # Both points leave the top-right corner and must read its value (4).
        torch.testing.assert_close(context, torch.tensor([[[4.]]]))
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 1))

    def test_future_batch_time_and_mode_correspondence(self):
        sampler = self.make_sampler()
        batch, modes, steps = 2, 2, 3
        raster = torch.arange(15, dtype=torch.float32).reshape(1, 1, 1, 3, 5)
        time_values = torch.tensor([[0., 10., 20.], [100., 110., 120.]])
        bev = raster + time_values[:, :, None, None, None]
        references = torch.zeros(batch, modes, steps, 2)
        references[:, 0, :, :] = torch.tensor([-1., -1.])
        references[:, 1, :, :] = torch.tensor([1., 1.])
        context, weights, locations = sample_future_waypoints(
            sampler, torch.zeros(batch, modes, steps, 3), bev, references
        )
        expected = torch.stack((time_values, time_values + 14), dim=1).unsqueeze(-1)
        torch.testing.assert_close(context, expected)
        torch.testing.assert_close(weights.sum(-1), torch.ones(batch, modes, steps))
        torch.testing.assert_close(locations, references.unsqueeze(-2).expand(-1, -1, -1, 2, -1))

    def test_gradients_through_learned_offsets_and_weights(self):
        sampler = self.make_sampler()
        with torch.no_grad():
            sampler.offset_head.bias.copy_(torch.tensor([-0.2, 0.1, 0.2, -0.1]))
        query = torch.ones(1, 1, 3, requires_grad=True)
        bev = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5).requires_grad_()
        reference = torch.zeros(1, 1, 2, requires_grad=True)
        context, _, _ = sampler(query, bev, reference)
        context.sum().backward()
        for gradient in (sampler.offset_head.weight.grad,
                         sampler.weight_head.weight.grad, bev.grad, reference.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_invalid_inputs(self):
        sampler = self.make_sampler()
        query = torch.zeros(1, 2, 3)
        bev = torch.zeros(1, 1, 3, 5)
        reference = torch.zeros(1, 2, 2)
        cases = [
            (query.squeeze(0), bev, reference),
            (query, bev.unsqueeze(0), reference),
            (query[..., :2], bev, reference),
            (query, bev.expand(2, -1, -1, -1), reference),
            (query, bev, reference[:, :1]),
            (query, bev[:, :, :1], reference),
            (query, bev.double(), reference),
        ]
        for inputs in cases:
            with self.subTest(shapes=[tuple(value.shape) for value in inputs]):
                with self.assertRaises(ValueError):
                    sampler(*inputs)
        with self.assertRaises(ValueError):
            sample_future_waypoints(sampler, torch.zeros(1, 2, 3, 3),
                                    torch.zeros(1, 2, 1, 3, 5), torch.zeros(1, 2, 3, 2))
        for kwargs in ({"num_points": 0}, {"max_offset_pixels": 0},
                       {"max_offset_pixels": float("nan")}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    WaypointDeformableSampler(3, **kwargs)


class TrajectoryRefinementTests(unittest.TestCase):
    def test_fixed_correction_and_output_shapes(self):
        model = FutureGuidedTrajectoryRefinement(channels=4, num_points=2)
        prior = torch.tensor([[
            [[0., 0.], [0.2, 0.], [0.4, 0.]],
            [[0., 0.], [0.2, 0.1], [0.4, 0.2]],
        ]])
        bev = torch.zeros(1, 3, 4, 5, 5)
        with torch.no_grad():
            model.delta_head[-1].weight.zero_()
            model.delta_head[-1].bias.copy_(torch.tensor([0.05, -0.02]))
            # Zero offsets make the returned sampling locations equal to references.
            model.sampler.offset_head.weight.zero_()
            model.sampler.offset_head.bias.zero_()
        output = model(prior, bev)
        expected_correction = torch.tensor([0.05, -0.02]).expand_as(prior)
        torch.testing.assert_close(output["correction"], expected_correction)
        torch.testing.assert_close(output["refined_trajectory"], prior + expected_correction)
        torch.testing.assert_close(output["locations"], prior.unsqueeze(-2).expand(-1, -1, -1, 2, -1))
        self.assertEqual(tuple(output["context"].shape), (1, 2, 3, 4))
        self.assertEqual(tuple(output["weights"].shape), (1, 2, 3, 2))

    def test_correction_clipped_at_grid_boundary(self):
        model = FutureGuidedTrajectoryRefinement(channels=1)
        prior = torch.tensor([[[[0.98, -0.99]]]])
        with torch.no_grad():
            model.delta_head[-1].weight.zero_()
            model.delta_head[-1].bias.copy_(torch.tensor([0.05, -0.02]))
        output = model(prior, torch.zeros(1, 1, 1, 3, 3))
        torch.testing.assert_close(output["refined_trajectory"], torch.tensor([[[[1., -1.]]]]))
        torch.testing.assert_close(output["correction"], torch.tensor([[[[0.05, -0.02]]]]))

    def test_future_environment_affects_corresponding_waypoints_and_gradients(self):
        model = FutureGuidedTrajectoryRefinement(channels=1, num_points=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            # Select only the context half of [query, context]. Positive context
            # passes through ReLU, then produces dx=0.1*context, dy=0.2*context.
            model.delta_head[0].weight.copy_(torch.tensor([[0., 1.]]))
            model.delta_head[-1].weight.copy_(torch.tensor([[0.1], [0.2]]))

        prior = torch.zeros(2, 2, 3, 2)
        time_values = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        bev = time_values[:, :, None, None, None].expand(-1, -1, 1, 5, 5).clone()
        bev.requires_grad_()
        output = model(prior, bev)
        expected_context = time_values[:, None, :, None].expand(-1, 2, -1, -1)
        expected_correction = expected_context * torch.tensor([0.1, 0.2])
        torch.testing.assert_close(output["context"], expected_context)
        torch.testing.assert_close(output["refined_trajectory"], expected_correction)

        # Change only sample 0, future step 1. Only its two modes should change.
        changed_bev = bev.detach().clone()
        changed_bev[0, 1] += 0.1
        changed = model(prior, changed_bev)
        expected_change = torch.zeros_like(prior)
        expected_change[0, :, 1] = torch.tensor([0.01, 0.02])
        torch.testing.assert_close(
            changed["refined_trajectory"] - output["refined_trajectory"],
            expected_change,
        )

        output["refined_trajectory"].sum().backward()
        for gradient in (bev.grad, model.delta_head[-1].weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
