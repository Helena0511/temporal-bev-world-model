import unittest

import torch

from resworld.driving_model import DrivingResWorld, ModelConfig, planning_loss


class KinematicPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_batch(self):
        transform = torch.eye(3).repeat(1, 2, 1, 1)
        transform[:, 0, 0, 2] = 4
        transform[:, 1, 0, 2] = 2
        return dict(images=torch.rand(1, 3, 6, 3, 16, 16),
                    intrinsics=torch.eye(3).repeat(1, 3, 6, 1, 1),
                    camera_to_ego=torch.eye(4).repeat(1, 3, 6, 1, 1),
                    current2history=transform, history_dt=torch.tensor([.5]),
                    motion=torch.zeros(1, 3, 3), command=torch.tensor([1]))

    def test_zero_heads_reproduce_constant_velocity_in_meters(self):
        config = ModelConfig(channels=8, tokens=2, heads=2, bev_height=8, bev_width=8,
                             future_steps=3, depth_bins=4, kinematic_prior=True)
        model = DrivingResWorld(config).eval()
        batch = self.make_batch()
        output = model(batch)
        expected = torch.tensor([[[[2., 0.], [4., 0.], [6., 0.]]]]).expand(1, 3, -1, -1)
        torch.testing.assert_close(output['init_traj'], expected)
        torch.testing.assert_close(output['final_traj'], expected)
        # Supervision can change the residual heads while the physical prior stays fixed.
        batch.update(gt_positions=torch.zeros(1, 3, 2), future_valid=torch.ones(1, 3, dtype=torch.bool))
        planning_loss(output, batch)['total'].backward()
        self.assertGreater(model.init_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.refine_head[-1].weight.grad.abs().sum().item(), 0)

    def test_invalid_time_interval_is_rejected(self):
        model = DrivingResWorld(ModelConfig(channels=8, tokens=2, heads=2, bev_height=8,
                                bev_width=8, depth_bins=4, kinematic_prior=True))
        batch = self.make_batch()
        batch['history_dt'] = torch.tensor([0.])
        with self.assertRaises(ValueError):
            model(batch)

    def test_velocity_is_expressed_in_rotated_current_frame(self):
        model = DrivingResWorld(ModelConfig(channels=8, tokens=2, heads=2, bev_height=8,
                                bev_width=8, future_steps=3, depth_bins=4, kinematic_prior=True)).eval()
        batch = self.make_batch()
        # Current ego: global (0,2), yaw=90 degrees. Previous ego: global origin.
        # Motion is therefore +2m along current forward, not along current left.
        batch['current2history'][:, 1] = torch.tensor([[0., -1., 0.], [1., 0., 2.], [0., 0., 1.]])
        with torch.no_grad():
            output = model(batch)
        torch.testing.assert_close(output['final_traj'][0, 1], torch.tensor([[2., 0.], [4., 0.], [6., 0.]]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
