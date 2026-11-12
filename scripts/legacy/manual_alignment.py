import torch

from resworld.nuscenes_io import NuScenesTwoHistory, get_current2history
from resworld.temporal_alignment import BEVGrid,warp_bev_to_target

history_ego_pose={
    "translation":[0.0,0.0,0.0],
    "rotation":[1.0,0.0,0.0,0.0],
}

current_ego_pose={
    "translation":[2.0,0.0,0.0],
    "rotation":[1.0,0.0,0.0,0.0],
}

current2history=get_current2history(current_ego_pose,history_ego_pose)["current2history"]

print("current2history:", current2history)


grid=BEVGrid(
    x_min=0,
    x_max=20,
    y_min=-10,
    y_max=10,
    height=21,
    width=21,
)

history_bev=torch.zeros(1,1,21,21)
current_bev=torch.zeros_like(history_bev)

history_row,history_col=grid.metric_to_row_col(x=10,y=0)
history_bev[0,0,history_row,history_col]=1.0

current_row,current_col=grid.metric_to_row_col(x=8,y=0)
current_bev[0,0,current_row,current_col]=1.0


aligned_history_bev=warp_bev_to_target(
    source_bev=history_bev,
    target_to_source=current2history.unsqueeze(0),
    grid=grid
)

raw_residual=current_bev-history_bev
aligned_residual=current_bev-aligned_history_bev

print("未对齐的残差绝对值之和:", raw_residual.abs().sum().item())
print("对齐后的残差绝对值之和:", aligned_residual.abs().sum().item())

torch.testing.assert_close(aligned_history_bev,current_bev)
torch.testing.assert_close(aligned_residual,torch.zeros_like(current_bev))


# 读取真实三帧位姿，顺序为 [t-2, t-1, t]。
dataset = NuScenesTwoHistory("/mnt/ssd0/datasets/nuscenes")
current_token = dataset.find_sample_with_two_history()
pose_t2, pose_t1, pose_t = dataset.get_three_frame_poses(current_token)

current2history_t2 = get_current2history(pose_t, pose_t2)["current2history"]
current2history_t1 = get_current2history(pose_t, pose_t1)["current2history"]

print("当前帧 → t-2 坐标系:\n", current2history_t2)
print("当前帧 → t-1 坐标系:\n", current2history_t1)

# 每个矩阵将当前帧的点转换到对应历史帧，供 BEV 对齐时采样。
current2history = torch.stack(
    [current2history_t2, current2history_t1], dim=0
)
print("两帧历史变换形状:", current2history.shape)
assert current2history.shape == (2, 3, 3)
assert torch.isfinite(current2history).all()


# 使用真实自车运动，生成同一个静态场景在三个自车坐标系下的 BEV。
# 这里用解析场景验证几何；数值来自构造场景，不是真实图像编码特征。
real_grid = BEVGrid(
    x_min=-20, x_max=20, y_min=-20, y_max=20,
    height=401, width=401,
)
current_points = real_grid.target_metric_points(1, "cpu", torch.float32)


def static_scene(points):
    """在固定的当前帧坐标系中定义三个平滑的静态地标。"""
    x, y = points[..., 0], points[..., 1]
    feature = torch.zeros_like(x)
    for cx, cy, amplitude in ((5, 0, 1.0), (-3, 5, 0.8), (0, -6, 0.6)):
        feature += amplitude * torch.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * 1.5 ** 2)
        )
    return feature.unsqueeze(1)  # [B, C=1, H, W]


static_current_bev = static_scene(current_points)
history_features = []
for transform in current2history:
    # 历史网格点先转换到当前坐标系，再查询同一个静态场景。
    history_points_in_current = torch.einsum(
        "ij,bhwj->bhwi", torch.linalg.inv(transform), current_points
    )
    history_features.append(static_scene(history_points_in_current))

history_features = torch.stack(history_features, dim=1)  # [B, T=2, C, H, W]
aligned_features = []
valid_masks = []
for index, transform in enumerate(current2history):
    source = history_features[:, index]
    aligned_features.append(warp_bev_to_target(
        source_bev=source,
        target_to_source=transform.unsqueeze(0),
        grid=real_grid,
    ))
    # 越界位置没有历史观测，不应算作场景变化。
    valid_masks.append(warp_bev_to_target(
        source_bev=torch.ones_like(source),
        target_to_source=transform.unsqueeze(0),
        grid=real_grid,
    ) > 0.5)

aligned_features = torch.stack(aligned_features, dim=1)
valid_masks = torch.stack(valid_masks, dim=1)
raw_residuals = static_current_bev[:, None] - history_features
aligned_residuals = static_current_bev[:, None] - aligned_features

print("历史 BEV [B,T,C,H,W]:", history_features.shape)
print("对齐后 BEV:", aligned_features.shape)
for index, label in enumerate(("t-2", "t-1")):
    valid = valid_masks[:, index]
    raw_error = raw_residuals[:, index][valid].abs().mean()
    aligned_error = aligned_residuals[:, index][valid].abs().mean()
    print(
        f"{label}: 有效覆盖率={valid.float().mean().item():.1%}, "
        f"未对齐 MAE={raw_error.item():.6f}, "
        f"对齐后 MAE={aligned_error.item():.6f}"
    )
    # 真实位姿包含非整数平移和旋转，栅格采样后允许少量误差。
    assert aligned_error < 0.1 * raw_error

# 当前帧额外出现一个局部变化：对齐应消除自车运动，但保留场景变化。
change = 0.5 * torch.exp(
    -((current_points[..., 0] - 2) ** 2 + (current_points[..., 1] - 2) ** 2)
    / (2 * 1.0 ** 2)
)
change = change.unsqueeze(1)
changed_current_bev = static_current_bev + change
changed_residuals = changed_current_bev[:, None] - aligned_features
torch.testing.assert_close(
    changed_residuals - aligned_residuals,
    change[:, None].expand_as(changed_residuals),
    rtol=1e-5, atol=1e-7,
)
print("两帧历史对齐通过；当前帧新增的局部变化保留在残差中。")
