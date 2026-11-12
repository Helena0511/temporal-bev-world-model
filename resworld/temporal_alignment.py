"""Metric BEV warping used before computing ResWorld temporal residuals.

The test at the bottom is deliberately independent of the image encoder.  It
checks the geometric invariant needed by TR-World: after ego-motion alignment,
the same world point occupies the same BEV cell in two adjacent frames.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class BEVGrid:
    """Metric convention for a feature map shaped ``[B, C, H, W]``.

    ``x`` is forward and ``y`` is left.  Row 0 is the largest x (furthest in
    front of the ego vehicle); column 0 is the largest y (furthest left).
    """

    def __init__(self, x_min, x_max, y_min, y_max, height, width):
        if not (x_max > x_min and y_max > y_min and height > 1 and width > 1):
            raise ValueError("BEV bounds must increase and H/W must be at least 2")
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.height, self.width = int(height), int(width)

    def metric_to_row_col(self, x, y):
        """Return nearest raster cell for a metric point in the ego frame."""
        row = round((self.x_max - float(x)) * (self.height - 1) / (self.x_max - self.x_min))
        col = round((self.y_max - float(y)) * (self.width - 1) / (self.y_max - self.y_min))
        return int(row), int(col)

    def target_metric_points(self, batch_size, device, dtype):
        """Return homogeneous target-frame coordinates: ``[B, H, W, 3]``."""
        x = torch.linspace(self.x_max, self.x_min, self.height, device=device, dtype=dtype)
        y = torch.linspace(self.y_max, self.y_min, self.width, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        ones = torch.ones_like(xx)
        points = torch.stack((xx, yy, ones), dim=-1)
        return points.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def source_points_to_grid(self, source_points):
        """Convert source-frame metric xy points to ``grid_sample`` coordinates."""
        x, y = source_points[..., 0], source_points[..., 1]
        col = (self.y_max - y) * (self.width - 1) / (self.y_max - self.y_min)
        row = (self.x_max - x) * (self.height - 1) / (self.x_max - self.x_min)
        grid_x = 2.0 * col / (self.width - 1) - 1.0
        grid_y = 2.0 * row / (self.height - 1) - 1.0
        return torch.stack((grid_x, grid_y), dim=-1)

    def grid_points_to_metric(self, grid_points):
        """Invert normalized (column,row) coordinates to metric (forward,left).

        Uses the same endpoint pixel-center convention as source_points_to_grid.
        """
        grid_x, grid_y = grid_points[..., 0], grid_points[..., 1]
        x = self.x_max - (grid_y + 1.0) * 0.5 * (self.x_max - self.x_min)
        y = self.y_max - (grid_x + 1.0) * 0.5 * (self.y_max - self.y_min)
        return torch.stack((x, y), dim=-1)


def warp_bev_to_target(source_bev, target_to_source, grid):
    """Express a source-ego BEV in the target ego coordinate system.

    Args:
        source_bev: Features from the earlier frame, ``[B, C, H, W]``.
        target_to_source: Batched planar homogeneous transform, ``[B, 3, 3]``.
            It maps a point expressed in the target ego frame into the source
            ego frame.  For nuScenes poses this is ``inverse(E_source) @
            E_target``, where E maps ego coordinates to global coordinates.
        grid: The metric BEV raster convention shared by both tensors.
    """
    if source_bev.ndim != 4:
        raise ValueError("source_bev must have shape [B, C, H, W]")
    batch_size, _, height, width = source_bev.shape
    if (height, width) != (grid.height, grid.width):
        raise ValueError("source_bev shape does not match BEVGrid")
    if target_to_source.shape != (batch_size, 3, 3):
        raise ValueError("target_to_source must have shape [B, 3, 3]")

    target_points = grid.target_metric_points(batch_size, source_bev.device, source_bev.dtype)
    source_points = torch.einsum("bij,bhwj->bhwi", target_to_source, target_points)
    sample_grid = grid.source_points_to_grid(source_points)
    return F.grid_sample(
        source_bev, sample_grid, mode="nearest", padding_mode="zeros", align_corners=True
    )


def _translation(tx, ty):
    matrix = torch.eye(3).unsqueeze(0)
    matrix[:, 0, 2] = tx
    matrix[:, 1, 2] = ty
    return matrix


def run_alignment_test():
    """A static point at global x=10 moves from x=10 to x=8 after ego moves +2m."""
    grid = BEVGrid(x_min=0, x_max=20, y_min=-10, y_max=10, height=21, width=21)
    source = torch.zeros(1, 1, 21, 21)
    source_row, source_col = grid.metric_to_row_col(x=10, y=0)
    source[0, 0, source_row, source_col] = 1.0

    # E_source = identity; E_target translates target-ego coordinates by +2 m
    # into global coordinates.  Hence target x=8 is source x=10.
    target_to_source = _translation(tx=2, ty=0)
    aligned = warp_bev_to_target(source, target_to_source, grid)
    peak_row, peak_col = torch.nonzero(aligned[0, 0] == aligned.max(), as_tuple=False)[0].tolist()
    expected_row, expected_col = grid.metric_to_row_col(x=8, y=0)

    assert (peak_row, peak_col) == (expected_row, expected_col), (
        f"expected static point at {(expected_row, expected_col)}, got {(peak_row, peak_col)}"
    )
    print("alignment test passed")
    print("source static point (x=10, y=0):", (source_row, source_col))
    print("aligned point in target frame (x=8, y=0):", (peak_row, peak_col))


if __name__ == "__main__":
    run_alignment_test()
