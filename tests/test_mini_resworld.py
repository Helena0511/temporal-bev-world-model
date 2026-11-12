"""Checks for the restored metric-coordinate MiniResWorld training path."""

import unittest

import torch

from resworld.mini_resworld import MiniResWorld, planning_loss
from resworld.temporal_alignment import BEVGrid


class MiniResWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_metric_grid_axes_and_round_trip(self):
        grid = BEVGrid(-15, 15, -30, 30, 6, 8)
        points = torch.tensor([[15., 30.], [0., 0.], [-15., -30.], [7.5, -15.]])
        normalized = grid.source_points_to_grid(points)
        torch.testing.assert_close(normalized, torch.tensor([
            [-1., -1.], [0., 0.], [1., 1.], [0.5, -0.5],
        ]))
        torch.testing.assert_close(grid.grid_points_to_metric(normalized), points)

    def test_command_loss_value_and_selected_gradients(self):
        final = torch.full((2, 3, 2, 2), 0.1, requires_grad=True)
        initial = torch.full((2, 3, 2, 2), 0.2, requires_grad=True)
        command = torch.tensor([0, 2])
        loss = planning_loss({"final_traj": final, "init_traj": initial},
                             torch.zeros(2, 2, 2), command)
        torch.testing.assert_close(loss, torch.tensor(0.015))
        loss.backward()
        selected = torch.zeros(2, 3, dtype=torch.bool)
        selected[torch.arange(2), command] = True
        for prediction in (final, initial):
            self.assertTrue((prediction.grad[~selected] == 0).all())
            self.assertTrue((prediction.grad[selected].abs().sum((-2, -1)) > 0).all())

    def test_integrated_future_bev_and_backward(self):
        torch.manual_seed(12)
        model = MiniResWorld(bev_dim=8, num_tokens=3, future_steps=3,
                             num_heads=2, transformer_layers=1)
        current = torch.randn(3, 4, 5, 8)
        history = torch.randn_like(current)
        command = torch.tensor([0, 1, 2])
        output = model(current, history, command)
        self.assertEqual(tuple(output["future_bev"].shape), (3, 3, 8, 4, 5))
        self.assertEqual(tuple(output["final_traj"].shape), (3, 3, 3, 2))
        self.assertEqual(tuple(output["sampling_locations"].shape), (3, 3, 3, 4, 2))
        torch.testing.assert_close(output["init_traj"], output["init_deltas"].cumsum(2))
        grid = BEVGrid(-15, 15, -30, 30, 4, 5)
        torch.testing.assert_close(output["reference_grid"], grid.source_points_to_grid(output["init_traj"]))
        torch.testing.assert_close(output["future_bev"],
                                   current.permute(0, 3, 1, 2)[:, None] + output["future_delta_bev"])
        torch.testing.assert_close(output["sampling_weights"].sum(-1), torch.ones(3, 3, 3))
        target = torch.tensor([[[1., 0.], [2., 0.], [3., 0.]]]).expand(3, -1, -1)
        loss = planning_loss(output, target, command)
        loss.backward()
        # Confirm planning supervision reaches both prediction and world branches.
        for parameter in (model.init_traj_head[-1].weight,
                          model.action_film.condition[0].weight,
                          model.token_fuser.token_proj.weight,
                          model.future_time_embedding,
                          model.fgtr.sampler.offset_head.weight,
                          model.fgtr.sampler.weight_head.weight,
                          model.fgtr.delta_head[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_invalid_command_and_target(self):
        model = MiniResWorld(bev_dim=8, num_tokens=2, future_steps=2,
                             num_heads=2, transformer_layers=1)
        bev = torch.zeros(1, 3, 4, 8)
        for command in (torch.tensor([3]), torch.tensor([-1]), torch.tensor([1.])):
            with self.assertRaises(ValueError):
                model(bev, bev, command)
        outputs = {"final_traj": torch.zeros(1, 3, 2, 2),
                   "init_traj": torch.zeros(1, 3, 2, 2)}
        with self.assertRaises(ValueError):
            planning_loss(outputs, torch.zeros(1, 3, 2), torch.tensor([1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
