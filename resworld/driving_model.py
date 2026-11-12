"""Dependency-light ResWorld implementation; see REPRODUCTION.md for differences."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .temporal_alignment import BEVGrid, warp_bev_to_target
from .mini_resworld import TokenLearner, TokenFuser, ActionFiLM
from .fgtr import WaypointDeformableSampler, sample_future_waypoints
from .bev_attention import BEVQueryEncoder, TemporalBEVAttention


@dataclass
class ModelConfig:
    channels: int = 32
    tokens: int = 8
    heads: int = 4
    layers: int = 1
    bev_height: int = 32
    bev_width: int = 48
    x_min: float = -15.
    x_max: float = 15.
    y_min: float = -30.
    y_max: float = 30.
    depth_bins: int = 16
    depth_min: float = 1.
    depth_max: float = 35.
    future_steps: int = 6
    perception: str = 'lift_splat'
    temporal_fusion: str = 'conv'
    residual_source: str = 'conditioned'
    spatial_layers: int = 1
    spatial_points: int = 2
    anchor_heights: tuple = (-1., 0., 1., 2.)
    query_chunk_size: int = 256
    use_world_delta: bool = True
    temporal_offset_meters: float = 4.
    kinematic_prior: bool = False
    use_ego_state: bool = True
    temporal_context: str = 'aligned'
    image_backbone: str = 'tiny'

    def __post_init__(self):
        if self.perception not in ('lift_splat', 'bev_query'):
            raise ValueError('unknown perception encoder')
        if self.temporal_fusion not in ('conv', 'attention'):
            raise ValueError('unknown temporal fusion')
        if self.residual_source not in ('conditioned', 'visual'):
            raise ValueError('unknown residual source')
        if min(self.spatial_layers, self.spatial_points, self.query_chunk_size) < 1:
            raise ValueError('attention layer, point, and chunk counts must be positive')
        if not self.anchor_heights:
            raise ValueError('height anchors cannot be empty')
        if self.temporal_offset_meters <= 0:
            raise ValueError('temporal offset radius must be positive')
        if self.temporal_context not in ('aligned', 'single_frame'):
            raise ValueError('unknown temporal context')
        if self.kinematic_prior and not self.use_ego_state:
            raise ValueError('a no-ego model cannot use a kinematic prior')
        if self.image_backbone not in ('tiny','resnet50'):
            raise ValueError('unknown image backbone')
        if self.image_backbone == 'resnet50' and self.perception != 'bev_query':
            raise ValueError('ResNet50-FPN is currently supported by the BEV query encoder')

    def grid(self):
        return BEVGrid(self.x_min, self.x_max, self.y_min, self.y_max,
                       self.bev_height, self.bev_width)


class CameraLiftSplat(nn.Module):
    """Predict categorical depths, lift rays with K^-1, then splat into metric BEV."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.channels
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, c, 3, stride=2, padding=1), nn.GroupNorm(4, c), nn.SiLU(),
        )
        self.depth_head = nn.Conv2d(c, config.depth_bins, 1)
        self.context_head = nn.Conv2d(c, c, 1)
        self.refine = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(4, c),
                                    nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))
        self.register_buffer('depths', torch.linspace(config.depth_min, config.depth_max,
                                                     config.depth_bins))

    def forward(self, images, intrinsics, camera_to_ego):
        # The independent camera projection already compensates exposure timing.
        batch, views, _, image_h, image_w = images.shape
        cfg = self.config
        feature = self.backbone(images.reshape(batch * views, 3, image_h, image_w))
        logits = self.depth_head(feature)
        depth = logits.softmax(1)
        context = self.context_head(feature)
        _, channels, height, width = context.shape
        # Conv stride 8 with padding 1 has feature centres at input pixels 0,8,...
        yy, xx = torch.meshgrid(torch.arange(height, device=images.device) * 8.,
                                torch.arange(width, device=images.device) * 8., indexing='ij')
        pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
        rays = torch.einsum('bvij,pj->bvpi', torch.linalg.inv(intrinsics), pixels)
        points = rays[:, :, None] * self.depths[None, None, :, None, None]
        points = torch.einsum('bvij,bvdpj->bvdpi', camera_to_ego[..., :3, :3], points)
        points = points + camera_to_ego[..., :3, 3][:, :, None, None]
        # Endpoint pixel-centre convention, identical to BEVGrid and grid_sample.
        row = torch.round((cfg.x_max - points[..., 0]) / (cfg.x_max-cfg.x_min)
                          * (cfg.bev_height-1)).long()
        col = torch.round((cfg.y_max - points[..., 1]) / (cfg.y_max-cfg.y_min)
                          * (cfg.bev_width-1)).long()
        valid = ((row >= 0) & (row < cfg.bev_height) & (col >= 0) & (col < cfg.bev_width)
                 & (points[..., 2] >= -5) & (points[..., 2] <= 3))
        lifted = context[:, :, None] * depth[:, None]
        lifted = lifted.reshape(batch, views, channels, cfg.depth_bins, height * width)
        lifted = lifted.permute(0, 2, 1, 3, 4).flatten(2)
        indices = (row * cfg.bev_width + col).flatten(1)
        indices = indices.clamp(0, cfg.bev_height * cfg.bev_width - 1)
        keep = valid.flatten(1).to(lifted.dtype)
        bev = lifted.new_zeros(batch, channels, cfg.bev_height * cfg.bev_width)
        bev.scatter_add_(2, indices[:, None].expand(-1, channels, -1), lifted * keep[:, None])
        count = lifted.new_zeros(batch, 1, cfg.bev_height * cfg.bev_width)
        count.scatter_add_(2, indices[:, None], keep[:, None])
        bev = bev / count.clamp_min(1) * cfg.depth_bins
        bev = self.refine(bev.reshape(batch, channels, cfg.bev_height, cfg.bev_width))
        return bev, logits.reshape(batch, views, cfg.depth_bins, height, width)


class DrivingResWorld(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        c, k, steps = cfg.channels, cfg.tokens, cfg.future_steps
        if c % 4 or c % cfg.heads:
            raise ValueError('channels must be divisible by 4 and attention heads')
        self.camera_bev = (BEVQueryEncoder(cfg) if cfg.perception == 'bev_query'
                           else CameraLiftSplat(cfg))
        self.state_condition = nn.Sequential(nn.Linear(3, c), nn.SiLU(), nn.Linear(c, c*2))
        self.fusion = nn.Sequential(nn.Conv2d(c*3, c, 3, padding=1), nn.GroupNorm(4, c), nn.SiLU())
        if cfg.temporal_fusion == 'attention':
            self.temporal_attention = TemporalBEVAttention(c, cfg.heads, chunk_size=cfg.query_chunk_size,
                                                           max_offset_meters=cfg.temporal_offset_meters)
        self.command_embedding = nn.Embedding(3, c)
        self.position = nn.Linear(2, c)
        self.scene_tokenizer = TokenLearner(c, k)
        self.residual_tokenizer = TokenLearner(c, k)
        def transformer():
            return nn.TransformerEncoder(nn.TransformerEncoderLayer(
                c, cfg.heads, dim_feedforward=c*4, dropout=0., batch_first=True,
                norm_first=True), cfg.layers, enable_nested_tensor=False)
        self.scene_transformer = transformer()
        self.waypoint_query = nn.Parameter(torch.randn(3*steps, c) * .02)
        self.waypoint_attention = nn.MultiheadAttention(c, cfg.heads, batch_first=True)
        self.init_head = nn.Linear(c, 2)
        if cfg.kinematic_prior:
            # Start exactly from past-motion extrapolation; learn displacement corrections.
            nn.init.zeros_(self.init_head.weight)
            nn.init.zeros_(self.init_head.bias)
        self.action_condition = ActionFiLM(steps*2, c)
        self.residual_transformer = transformer()
        self.token_fuser = TokenFuser(c, k)
        self.sampler = WaypointDeformableSampler(c, num_points=4)
        self.refine_head = nn.Sequential(nn.Linear(c*2, c), nn.SiLU(), nn.Linear(c, 2))
        # Refined head predicts a displacement correction; start with prior trajectory.
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)

    def forward(self, batch):
        images = batch['images']
        b, times, views, _, ih, iw = images.shape
        if times != 3:
            raise ValueError('expected chronological [t-2,t-1,t] images')
        cfg = self.config
        c, h, w = cfg.channels, cfg.bev_height, cfg.bev_width
        single = cfg.temporal_context == 'single_frame'
        encoded_times = 1 if single else times
        camera_images = images[:, -1:] if single else images
        intrinsics = batch['intrinsics'][:, -1:] if single else batch['intrinsics']
        # Static calibration has no exposure-time ego-pose compensation. This is
        # required for the strict single-frame control; never silently fall back.
        extrinsics = (batch['camera_to_ego_static'][:, -1:] if single
                      else batch['camera_to_ego'])
        bev, depth_logits = self.camera_bev(
            camera_images.reshape(b*encoded_times, views, 3, ih, iw),
            intrinsics.reshape(b*encoded_times, views, 3, 3),
            extrinsics.reshape(b*encoded_times, views, 4, 4))
        bev = bev.reshape(b, encoded_times, c, h, w)
        if single:
            bev = bev.expand(-1, times, -1, -1, -1)
        if cfg.use_ego_state:
            gamma, beta = self.state_condition(batch['motion']).chunk(2, -1)
            conditioned = (1 + gamma[..., None, None]) * bev + beta[..., None, None]
        else:
            conditioned = bev
        aligned, aligned_visual, masks = [], [], []
        visibility = (self.camera_bev.last_visibility.reshape(b, encoded_times, 1, h, w).expand(-1, times, -1, -1, -1)
                      if cfg.perception == 'bev_query' else torch.ones_like(bev[:, :, :1], dtype=torch.bool))
        for t in range(2):
            transform = (torch.eye(3, device=bev.device, dtype=bev.dtype)[None].expand(b, -1, -1)
                         if single else batch['current2history'][:, t].to(bev))
            aligned.append(warp_bev_to_target(conditioned[:, t], transform, cfg.grid()))
            if cfg.residual_source == 'visual':
                aligned_visual.append(warp_bev_to_target(bev[:, t], transform, cfg.grid()))
            masks.append(warp_bev_to_target(visibility[:, t].to(bev),
                                            transform, cfg.grid()) > .5)
        aligned.append(conditioned[:, 2])
        if single:
            fused_map = conditioned[:, 2]
        elif cfg.temporal_fusion == 'attention':
            fused_map = self.temporal_attention(conditioned[:, 2], aligned,
                                                masks + [visibility[:, 2]], cfg.grid())
        else:
            fused_map = self.fusion(torch.cat(aligned, dim=1))
        fused = fused_map.flatten(2).transpose(1, 2)
        points = cfg.grid().target_metric_points(1, bev.device, bev.dtype)[..., :2]
        scale = bev.new_tensor([max(abs(cfg.x_min), abs(cfg.x_max)),
                                max(abs(cfg.y_min), abs(cfg.y_max))])
        position = self.position((points / scale).reshape(1, h*w, 2))
        scene = fused + position + self.command_embedding(batch['command'])[:, None]
        scene_tokens, scene_weights = self.scene_tokenizer(scene)
        scene_tokens = self.scene_transformer(scene_tokens)
        # Adjacent t - (t-1), both in current coordinates. The older frame enters fusion.
        # Use identical, overlap-normalized spatial weights for current and history.
        _, weights = self.residual_tokenizer(scene)
        overlap_mask = masks[1] & visibility[:, 2]
        overlap = overlap_mask.flatten(2).to(weights.dtype)
        weights = weights * overlap
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        # Preserve per-frame observations: do not subtract temporally smoothed scene BEVs.
        residual_bev = (bev[:, 2] - aligned_visual[1] if cfg.residual_source == 'visual'
                        else aligned[2] - aligned[1])
        residual = torch.einsum('bkn,bnc->bkc', weights, residual_bev.flatten(2).transpose(1, 2))
        query = self.waypoint_query[None].expand(b, -1, -1)
        wp, _ = self.waypoint_attention(query, scene_tokens, scene_tokens)
        init_deltas = self.init_head(wp).reshape(b, 3, cfg.future_steps, 2)
        if cfg.kinematic_prior:
            dt = batch['history_dt']
            if not torch.isfinite(dt).all() or (dt <= 0).any():
                raise ValueError('history_dt must be finite and positive for the kinematic prior')
            previous_origin = torch.linalg.inv(batch['current2history'][:, 1])[:, :2, 2]
            velocity = -previous_origin / dt[:, None]
            init_deltas = init_deltas + velocity[:, None, None] * .5
        init_traj = init_deltas.cumsum(2)
        selected = init_deltas[torch.arange(b, device=bev.device), batch['command']].flatten(1)
        predicted_tokens = self.residual_transformer(self.action_condition(residual, selected))
        # Single future BEV, matching the local reference rather than independent horizons.
        future_delta = self.token_fuser(predicted_tokens, fused)
        if not cfg.use_world_delta:
            future_delta = torch.zeros_like(future_delta)
        future = (fused + future_delta).transpose(1, 2).reshape(b, c, h, w)
        references = cfg.grid().source_points_to_grid(init_traj)
        context, sample_weights, locations = sample_future_waypoints(
            self.sampler, wp.reshape(b, 3, cfg.future_steps, c),
            future[:, None].expand(-1, cfg.future_steps, -1, -1, -1), references)
        waypoint_features = torch.cat((wp.reshape(b, 3, cfg.future_steps, c), context), -1)
        correction = self.refine_head(waypoint_features)
        final_deltas = init_deltas + correction
        return dict(init_traj=init_traj, final_traj=final_deltas.cumsum(2),
                    init_deltas=init_deltas, final_deltas=final_deltas,
                    future_bev=future, future_delta_bev=future_delta,
                    residual_tokens=residual, residual_weights=weights,
                    overlap_mask=overlap_mask, scene_weights=scene_weights,
                    temporal_residual_bev=residual_bev * overlap_mask.to(bev),
                    sampling_weights=sample_weights, sampling_locations=locations,
                    current_bev=bev[:, 2], waypoint_features=waypoint_features,
                    depth_logits=depth_logits.reshape(b, encoded_times, *depth_logits.shape[1:]))


def planning_loss(outputs, batch):
    """Masked L1 on per-step displacements, supervising both trajectory heads."""
    positions = batch['gt_positions']
    target = torch.diff(positions, dim=1, prepend=torch.zeros_like(positions[:, :1]))
    mask = batch['future_valid'].clone()
    mask[:, 1:] &= batch['future_valid'][:, :-1]
    weights = mask[..., None].to(positions.dtype)
    index = torch.arange(positions.shape[0], device=positions.device)
    terms = {}
    for name in ('init', 'final'):
        prediction = outputs[f'{name}_deltas'][index, batch['command']]
        terms[name] = ((prediction-target).abs() * weights).sum() / (2*weights.sum()).clamp_min(1)
    logits = outputs.get('depth_logits')
    target_depth = batch.get('depth_targets')
    terms['depth'] = terms['final'] * 0
    if logits is not None:
        terms['depth'] = logits.sum() * 0
    if target_depth is not None and logits is not None:
        target_depth = target_depth[:, -logits.shape[1]:]
        logits_flat = logits.reshape(-1, logits.shape[3], *logits.shape[-2:])
        targets_flat = target_depth.reshape(-1, *target_depth.shape[-2:])
        count = (targets_flat >= 0).sum()
        if count > 0:
            terms['depth'] = F.cross_entropy(logits_flat, targets_flat,
                                             ignore_index=-1, reduction='sum') / count
    terms['planning'] = terms['init'] + terms['final']
    terms['total'] = terms['planning'] + .1 * terms['depth']
    return terms
