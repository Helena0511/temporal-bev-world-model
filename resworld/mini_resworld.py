"""Standalone minimal ResWorld-style planning smoke test.

This starts from BEV features rather than raw images and implements:
BEV -> scene/residual tokens -> action-conditioned residual -> future BEV
-> multi-modal trajectory.
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fgtr import FutureGuidedTrajectoryRefinement
from .temporal_alignment import BEVGrid


# Project conventions used by this reproduction.
#
# Ego coordinate system (right-handed):
#   x: forward, y: left, z: up.
# Heads predict per-step displacements; init_traj and final_traj contain
# accumulated positions in meters in the fixed current-ego frame.
# Inputs must already be ego-motion aligned, with row 0 furthest forward and
# column 0 furthest left (the BEVGrid convention).
# Command IDs are deliberately fixed so data generation, training and
# evaluation cannot silently disagree:
#   0 = left, 1 = straight, 2 = right.
# The six future steps are interpreted as 0.5 s apart (3 s horizon).
COMMAND_LEFT = 0
COMMAND_STRAIGHT = 1
COMMAND_RIGHT = 2
NUM_COMMANDS = 3
FUTURE_STEP_SECONDS = 0.5

# Geometry convention inherited from the checked-in ResWorld configuration.
# Bounds are used by the trajectory/grid adapter. Resolution values describe
# the reference raster; this small demo uses a coarser H,W over the same bounds.
BEV_X_RANGE_METERS = (-15.0, 15.0)
BEV_Y_RANGE_METERS = (-30.0, 30.0)
BEV_X_RESOLUTION_METERS = 0.15
BEV_Y_RESOLUTION_METERS = 0.30


class TokenLearner(nn.Module):
    """Compress [B, N, C] spatial features into [B, K, C] tokens."""

    def __init__(self, in_dim: int, num_tokens: int, hidden_dim: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens),
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"TokenLearner expects [B, N, C], got {tuple(x.shape)}")
        scores = self.score(x)                       # [B, N, K]
        weights = scores.transpose(1, 2).softmax(-1)  # [B, K, N]
        tokens = torch.einsum("bkn,bnc->bkc", weights, x)
        return tokens, weights


class TokenFuser(nn.Module):
    """Fuse [B, K, C] tokens back into [B, N, C] BEV features."""

    def __init__(self, dim: int, num_tokens: int):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens),
            nn.Sigmoid(),
        )
        self.token_proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, base_bev: torch.Tensor):
        if tokens.ndim != 3 or base_bev.ndim != 3:
            raise ValueError("TokenFuser expects tokens [B,K,C] and BEV [B,N,C]")
        weights = self.weight_net(base_bev)            # [B, N, K]
        projected_tokens = self.token_proj(tokens)     # [B, K, C]
        return torch.einsum("bnk,bkc->bnc", weights, projected_tokens)


class ActionFiLM(nn.Module):
    """Condition residual tokens on the selected initial trajectory."""

    def __init__(self, action_dim: int, dim: int):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(action_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * 2),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, action: torch.Tensor):
        gamma, beta = self.condition(action).chunk(2, dim=-1)
        return (1.0 + gamma.unsqueeze(1)) * self.norm(x) + beta.unsqueeze(1)


class MiniResWorld(nn.Module):
    """Core ResWorld-style model operating on precomputed BEV features."""

    def __init__(
        self,
        bev_dim: int = 128,
        num_tokens: int = 16,
        num_modes: int = 3,
        future_steps: int = 6,
        num_commands: int = 3,
        num_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()
        if bev_dim % num_heads != 0:
            raise ValueError("bev_dim must be divisible by num_heads")
        if num_modes != num_commands:
            raise ValueError("This smoke test assumes one trajectory mode per command")

        self.bev_dim = bev_dim
        self.num_tokens = num_tokens
        self.num_modes = num_modes
        self.future_steps = future_steps
        if future_steps <= 0:
            raise ValueError("future_steps must be positive")

        self.command_embedding = nn.Embedding(num_commands, bev_dim)
        self.bev_condition = nn.Sequential(
            nn.Linear(bev_dim, bev_dim),
            nn.ReLU(),
            nn.Linear(bev_dim, bev_dim * 2),
        )

        self.scene_tokenizer = TokenLearner(bev_dim, num_tokens)
        self.residual_tokenizer = TokenLearner(bev_dim, num_tokens)

        scene_layer = nn.TransformerEncoderLayer(
            d_model=bev_dim, nhead=num_heads,
            batch_first=True, norm_first=True,
        )
        self.scene_transformer = nn.TransformerEncoder(
            scene_layer, num_layers=transformer_layers
        )

        self.waypoint_query = nn.Parameter(
            torch.randn(num_modes * future_steps, bev_dim) * 0.02
        )
        self.waypoint_attention = nn.MultiheadAttention(
            embed_dim=bev_dim, num_heads=num_heads, batch_first=True
        )
        self.init_traj_head = nn.Sequential(
            nn.LayerNorm(bev_dim),
            nn.Linear(bev_dim, bev_dim),
            nn.ReLU(),
            nn.Linear(bev_dim, 2),
        )

        self.action_film = ActionFiLM(future_steps * 2, bev_dim)

        residual_layer = nn.TransformerEncoderLayer(
            d_model=bev_dim, nhead=num_heads,
            batch_first=True, norm_first=True,
        )
        self.residual_transformer = nn.TransformerEncoder(
            residual_layer, num_layers=transformer_layers
        )
        self.token_fuser = TokenFuser(bev_dim, num_tokens)
        self.future_time_embedding = nn.Parameter(
            torch.randn(future_steps, bev_dim) * 0.02
        )
        self.fgtr = FutureGuidedTrajectoryRefinement(bev_dim)

    def apply_command_condition(self, bev: torch.Tensor, command: torch.Tensor):
        """FiLM-style command conditioning for [B, N, C] features."""
        command_feat = self.command_embedding(command)
        gamma, beta = self.bev_condition(command_feat).chunk(2, dim=-1)
        return (1.0 + gamma.unsqueeze(1)) * bev + beta.unsqueeze(1)

    def forward(self, current_bev, history_bev, command):
        if current_bev.ndim != 4 or history_bev.ndim != 4:
            raise ValueError("BEV inputs must have shape [B, H, W, C]")
        if current_bev.shape != history_bev.shape:
            raise ValueError("current_bev and history_bev must have identical shapes")
        if command.ndim != 1 or command.shape[0] != current_bev.shape[0]:
            raise ValueError("command must have shape [B]")
        if command.dtype != torch.long:
            raise ValueError("command must have dtype torch.long")
        if command.device != current_bev.device or history_bev.device != current_bev.device:
            raise ValueError("BEVs and command must be on the same device")
        if ((command < 0) | (command >= self.num_modes)).any():
            raise ValueError("command is outside the trajectory mode range")

        batch_size, height, width, channels = current_bev.shape
        if channels != self.bev_dim:
            raise ValueError(f"Expected C={self.bev_dim}, got C={channels}")
        grid = BEVGrid(*BEV_X_RANGE_METERS, *BEV_Y_RANGE_METERS, height, width)

        # Keep the geometric BEV representation separate from the planning
        # condition. The paper computes temporal residuals in the aligned BEV
        # coordinate system, then adds the predicted dynamic residual back to
        # the current BEV.
        current = current_bev.reshape(batch_size, height * width, channels)
        history = history_bev.reshape(batch_size, height * width, channels)

        # The command conditions the prior-trajectory branch.
        conditioned_current = self.apply_command_condition(current, command)
        scene_tokens, scene_weights = self.scene_tokenizer(conditioned_current)
        scene_tokens = self.scene_transformer(scene_tokens)

        # Use one spatial attention mask for both timestamps. This is the
        # important alignment rule: the subtraction compares the same sparse
        # spatial positions at t and t-1.
        current_res_tokens, residual_weights = self.residual_tokenizer(current)
        history_res_tokens = torch.einsum(
            "bkn,bnc->bkc", residual_weights, history
        )
        residual_tokens = current_res_tokens - history_res_tokens

        # Initial multi-modal trajectory.
        query = self.waypoint_query.unsqueeze(0).expand(batch_size, -1, -1)
        waypoint_features, _ = self.waypoint_attention(
            query=query, key=scene_tokens, value=scene_tokens
        )
        init_delta = self.init_traj_head(waypoint_features)
        init_deltas = init_delta.view(
            batch_size, self.num_modes, self.future_steps, 2
        )
        init_traj = init_deltas.cumsum(dim=2)

        # Use the command-selected initial trajectory as the action condition.
        batch_index = torch.arange(batch_size, device=current.device)
        selected_init = init_traj[batch_index, command]
        action = selected_init.reshape(batch_size, self.future_steps * 2)

        residual_tokens = self.action_film(residual_tokens, action)
        # Each horizon gets its own token features and BEV residual.
        future_tokens = residual_tokens[:, None] + self.future_time_embedding[None, :, None]
        future_tokens = self.residual_transformer(
            future_tokens.reshape(batch_size * self.future_steps, self.num_tokens, channels)
        )
        current_per_step = current[:, None].expand(-1, self.future_steps, -1, -1)
        future_delta = self.token_fuser(
            future_tokens,
            current_per_step.reshape(batch_size * self.future_steps, height * width, channels),
        )
        future_delta = future_delta.reshape(batch_size, self.future_steps, height * width, channels)
        future_bev_flat = current[:, None] + future_delta
        future_bev = future_bev_flat.reshape(
            batch_size, self.future_steps, height, width, channels
        ).permute(0, 1, 4, 2, 3).contiguous()

        # Metric (forward,left) is converted to grid (column,row) for sampling.
        reference_grid = grid.source_points_to_grid(init_traj)
        refinement = self.fgtr(reference_grid, future_bev)
        final_traj = grid.grid_points_to_metric(refinement["refined_trajectory"])

        return {
            "final_traj": final_traj,
            "init_traj": init_traj,
            "init_deltas": init_deltas,
            "future_bev": future_bev,
            "future_delta_bev": future_delta.reshape(
                batch_size, self.future_steps, height, width, channels
            ).permute(0, 1, 4, 2, 3).contiguous(),
            "reference_grid": reference_grid,
            "sampling_weights": refinement["weights"],
            "sampling_locations": refinement["locations"],
            "future_context": refinement["context"],
            "trajectory_correction": final_traj - init_traj,
            "scene_tokens": scene_tokens,
            "residual_tokens": residual_tokens,
            "scene_weights": scene_weights,
            "residual_weights": residual_weights,
        }


def planning_loss(outputs, gt_traj, command):
    """Smooth L1 on metric positions of the command-selected trajectory mode."""
    final_traj = outputs["final_traj"]
    init_traj = outputs["init_traj"]
    if final_traj.ndim != 4 or final_traj.shape[-1] != 2 or init_traj.shape != final_traj.shape:
        raise ValueError("init_traj and final_traj must have matching [B,M,T,2] shapes")
    if gt_traj.shape != (final_traj.shape[0], final_traj.shape[2], 2):
        raise ValueError("gt_traj must be [B,T,2]")
    if command.shape != (final_traj.shape[0],) or command.dtype != torch.long:
        raise ValueError("command must be a torch.long tensor shaped [B]")
    if ((command < 0) | (command >= final_traj.shape[1])).any():
        raise ValueError("command is outside the trajectory mode range")
    batch_index = torch.arange(gt_traj.shape[0], device=gt_traj.device)

    pred_final = final_traj[batch_index, command]
    pred_init = init_traj[batch_index, command]

    loss_final = F.smooth_l1_loss(pred_final, gt_traj)
    loss_init = F.smooth_l1_loss(pred_init, gt_traj)
    return loss_final + 0.5 * loss_init


def make_synthetic_batch(batch_size, height, width, channels, future_steps, device):
    """Create a deliberately simple command-conditioned smoke-test task."""
    current_bev = torch.randn(batch_size, height, width, channels, device=device)
    history_bev = torch.randn(batch_size, height, width, channels, device=device)
    command = torch.randint(0, NUM_COMMANDS, (batch_size,), device=device)

    # [dx, dy] in the fixed current-ego frame:
    # x is forward and y is left, so left has positive dy and right negative dy.
    directions = torch.tensor(
        [[1.0, 0.5], [1.0, 0.0], [1.0, -0.5]],
        dtype=current_bev.dtype,
        device=device,
    )
    gt_delta = directions[command].unsqueeze(1).repeat(1, future_steps, 1)
    gt_traj = gt_delta.cumsum(dim=1)
    return current_bev, history_bev, command, gt_traj


def run_smoke_test(steps=20, seed=0, device="cpu"):
    """Fit a fixed synthetic batch to validate optimization, not driving quality."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    batch_size, height, width, channels = 6, 6, 8, 16
    future_steps = 6
    model = MiniResWorld(
        bev_dim=channels, num_tokens=4, num_modes=3,
        future_steps=future_steps,
        num_commands=NUM_COMMANDS,
        transformer_layers=1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )

    current_bev, history_bev, command, _ = make_synthetic_batch(
        batch_size, height, width, channels, future_steps, device
    )
    # Include every command in the fixed toy batch.
    command = torch.arange(batch_size, device=device) % NUM_COMMANDS
    directions = current_bev.new_tensor([[1., 0.5], [1., 0.], [1., -0.5]])
    gt_traj = directions[command, None].expand(-1, future_steps, -1).cumsum(1)
    model.eval()
    with torch.no_grad():
        initial_loss = planning_loss(model(current_bev, history_bev, command), gt_traj, command).item()
    model.train()
    for step in range(steps):
        outputs = model(current_bev, history_bev, command)
        loss = planning_loss(outputs, gt_traj, command)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            print(f"step={step + 1:03d}, loss={loss.item():.6f}")

    assert outputs["init_traj"].shape == (batch_size, 3, future_steps, 2)
    assert outputs["final_traj"].shape == (batch_size, 3, future_steps, 2)
    assert outputs["future_bev"].shape == (batch_size, future_steps, channels, height, width)
    model.eval()
    with torch.no_grad():
        outputs = model(current_bev, history_bev, command)
        final_loss = planning_loss(outputs, gt_traj, command).item()
        batch_index = torch.arange(batch_size, device=device)
        initial_ade = (outputs["init_traj"][batch_index, command] - gt_traj).norm(dim=-1).mean().item()
        refined_ade = (outputs["final_traj"][batch_index, command] - gt_traj).norm(dim=-1).mean().item()
    print(f"fixed-batch eval loss: {initial_loss:.6f} -> {final_loss:.6f}")
    print(f"toy ADE (meters): initial={initial_ade:.6f}, refined={refined_ade:.6f}")
    print("smoke test passed")
    return {"initial_loss": initial_loss, "final_loss": final_loss,
            "initial_ade": initial_ade, "refined_ade": refined_ade}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniResWorld synthetic training check")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run_smoke_test(steps=args.steps, seed=args.seed, device=args.device)
