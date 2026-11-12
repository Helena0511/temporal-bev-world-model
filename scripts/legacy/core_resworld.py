"""Minimal, data-independent reproduction of ResWorld's two core ideas.

Inputs are already-aligned BEV features.  This deliberately excludes camera
encoders, ego-pose warping, nuScenes reading, detection, and map heads.

The two implemented equations are:
    R_t = B_t - B_{t-1}
    B_hat[t+k] = B_t + R_hat[t+k]

The FGTR block then uses each prior waypoint as a reference point to sample
the predicted future BEV and regress a waypoint correction.  The sampler has
learned offsets and learned weights, i.e. the essential deformable-attention
operation used by FGTR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualWorld(nn.Module):
    """Predict future BEV residuals from an aligned adjacent-frame residual."""

    def __init__(self, channels: int, future_steps: int):
        super().__init__()
        self.future_steps = future_steps
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual_head = nn.Conv2d(channels, future_steps * channels, 1)

    def forward(self, current_bev, previous_bev):
        """Return temporal residual and predicted future BEVs.

        Both inputs are ``[B, C, H, W]`` and must use the *same current-ego
        coordinate system*.  Alignment is intentionally outside this core
        implementation.
        """
        if current_bev.shape != previous_bev.shape or current_bev.ndim != 4:
            raise ValueError("current_bev and previous_bev must have identical [B,C,H,W] shapes")
        batch, channels, height, width = current_bev.shape
        temporal_residual = current_bev - previous_bev
        latent = self.encoder(temporal_residual)
        future_residual = self.residual_head(latent).view(
            batch, self.future_steps, channels, height, width
        )
        future_bev = current_bev.unsqueeze(1) + future_residual
        return temporal_residual, future_residual, future_bev


class PriorTrajectoryHead(nn.Module):
    """Produce multi-modal prior trajectories from the current scene feature."""

    def __init__(self, channels: int, num_modes: int, future_steps: int):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.query = nn.Parameter(torch.randn(num_modes, future_steps, channels) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, 2)
        )

    def forward(self, current_bev):
        scene = current_bev.mean(dim=(-2, -1))                    # [B, C]
        query = scene[:, None, None, :] + self.query[None]        # [B, M, T, C]
        deltas = self.head(query)
        # Waypoints are normalized grid coordinates in [-1, 1].  Metric-unit
        # conversion belongs to a BEV geometry adapter, not this core module.
        return deltas.cumsum(dim=2).tanh()


class FutureGuidedTrajectoryRefinement(nn.Module):
    """FGTR: deformably sample future BEV around prior trajectory waypoints."""

    def __init__(self, channels: int, num_points: int = 4):
        super().__init__()
        self.num_points = num_points
        self.query_embed = nn.Linear(2, channels)
        self.offsets = nn.Linear(channels, num_points * 2)
        self.weights = nn.Linear(channels, num_points)
        self.delta_head = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.ReLU(inplace=True), nn.Linear(channels, 2)
        )

    def forward(self, prior_trajectory, future_bev):
        """Refine ``[B,M,T,2]`` prior waypoints with ``[B,T,C,H,W]`` BEVs."""
        if prior_trajectory.ndim != 4 or prior_trajectory.shape[-1] != 2:
            raise ValueError("prior_trajectory must be [B, M, T, 2]")
        if future_bev.ndim != 5:
            raise ValueError("future_bev must be [B, T, C, H, W]")
        batch, modes, steps, _ = prior_trajectory.shape
        if future_bev.shape[:2] != (batch, steps):
            raise ValueError("trajectory and future_bev must have matching B and T")
        _, _, channels, height, width = future_bev.shape

        query = self.query_embed(prior_trajectory)                 # [B,M,T,C]
        offsets = self.offsets(query).view(batch, modes, steps, self.num_points, 2)
        # Offsets are normalized coordinates and bounded to a local BEV neighbourhood.
        sample_locations = (prior_trajectory.unsqueeze(-2) + 0.15 * offsets.tanh()).clamp(-1, 1)
        attention = self.weights(query).softmax(dim=-1)             # [B,M,T,P]

        # grid_sample operates on one image per future timestep.  Flatten B,T
        # and query all modes/points in that corresponding future BEV feature.
        bev_for_sampling = future_bev.reshape(batch * steps, channels, height, width)
        grid = sample_locations.permute(0, 2, 1, 3, 4).reshape(
            batch * steps, modes * self.num_points, 1, 2
        )
        sampled = F.grid_sample(
            bev_for_sampling, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).view(
            batch, steps, modes, self.num_points, channels
        ).permute(0, 2, 1, 3, 4)                                    # [B,M,T,P,C]
        future_context = (sampled * attention.unsqueeze(-1)).sum(dim=-2)
        correction = self.delta_head(torch.cat((query, future_context), dim=-1))
        return (prior_trajectory + correction).clamp(-1, 1)


class CoreResWorld(nn.Module):
    """TR-World plus FGTR, without non-core perception and data plumbing."""

    def __init__(self, channels=64, future_steps=6, num_modes=3, deformable_points=4):
        super().__init__()
        self.world = TemporalResidualWorld(channels, future_steps)
        self.prior = PriorTrajectoryHead(channels, num_modes, future_steps)
        self.fgtr = FutureGuidedTrajectoryRefinement(channels, deformable_points)

    def forward(self, current_bev, previous_bev):
        temporal_residual, future_residual, future_bev = self.world(current_bev, previous_bev)
        prior_trajectory = self.prior(current_bev)
        refined_trajectory = self.fgtr(prior_trajectory, future_bev)
        return {
            "temporal_residual": temporal_residual,
            "future_residual": future_residual,
            "future_bev": future_bev,
            "prior_trajectory": prior_trajectory,
            "refined_trajectory": refined_trajectory,
        }


def run_smoke_test():
    torch.manual_seed(0)
    model = CoreResWorld(channels=32, future_steps=6, num_modes=3)
    current = torch.randn(2, 32, 20, 20)
    previous = torch.randn(2, 32, 20, 20)
    output = model(current, previous)
    expected = {
        "temporal_residual": (2, 32, 20, 20),
        "future_residual": (2, 6, 32, 20, 20),
        "future_bev": (2, 6, 32, 20, 20),
        "prior_trajectory": (2, 3, 6, 2),
        "refined_trajectory": (2, 3, 6, 2),
    }
    for name, shape in expected.items():
        assert output[name].shape == shape, f"{name}: {output[name].shape}, expected {shape}"
    assert torch.isfinite(output["refined_trajectory"]).all()
    print("core ResWorld smoke test passed")
    for name, shape in expected.items():
        print(f"{name}: {shape}")


if __name__ == "__main__":
    run_smoke_test()
