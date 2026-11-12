"""Independent visual/ego training and late fusion for causality diagnostics."""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .driving_model import DrivingResWorld, planning_loss
from .perception_objectives import bev_segmentation_loss


STAGES = ('perception', 'visual', 'ego', 'fusion', 'joint')


class EgoTrajectoryBranch(nn.Module):
    """Past state only: a constant-velocity base plus learned corrections."""
    def __init__(self, channels, steps):
        super().__init__()
        self.steps = steps
        self.encoder = nn.Sequential(nn.Linear(12, channels), nn.SiLU(),
                                     nn.Linear(channels, channels), nn.SiLU())
        self.head = nn.Linear(channels, 3*steps*2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, batch):
        dt = batch['history_dt']
        if not torch.isfinite(dt).all() or (dt <= 0).any():
            raise ValueError('history_dt must be finite and positive')
        relative = torch.linalg.inv(batch['current2history'][:, 1])
        velocity = -relative[:, :2, 2] / dt[:, None]
        feature = self.encoder(torch.cat((batch['motion'].flatten(1), velocity, dt[:, None]), -1))
        deltas = self.head(feature).reshape(-1, 3, self.steps, 2) + velocity[:, None, None]*.5
        return dict(init_deltas=deltas, final_deltas=deltas, init_traj=deltas.cumsum(2),
                    final_traj=deltas.cumsum(2), ego_features=feature)


class StagedDrivingModel(nn.Module):
    def __init__(self, config, stage='visual', ego_dropout=.3):
        super().__init__()
        # The visual branch must never receive state FiLM or a motion prior.
        self.config = replace(config, use_ego_state=False, kinematic_prior=False)
        c = config.channels
        self.visual = DrivingResWorld(self.config)
        self.ego = EgoTrajectoryBranch(c, config.future_steps)
        self.perception_head = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(4, c),
                                             nn.SiLU(), nn.Conv2d(c, 3, 1))
        self.fusion_head = nn.Sequential(nn.Linear(3*c+5, c), nn.SiLU(), nn.Linear(c, 2))
        nn.init.zeros_(self.fusion_head[-1].weight)
        nn.init.zeros_(self.fusion_head[-1].bias)
        if not 0 <= ego_dropout < 1:
            raise ValueError('ego dropout must be in [0,1)')
        self.ego_dropout = ego_dropout
        self.set_stage(stage)

    def set_stage(self, stage):
        if stage not in STAGES:
            raise ValueError('unknown training stage')
        self.stage = stage
        for name, parameter in self.named_parameters():
            if stage == 'perception':
                active = name.startswith(('visual.camera_bev.', 'perception_head.'))
            elif stage == 'ego':
                active = name.startswith('ego.')
            elif stage == 'visual':
                active = name.startswith(('visual.', 'perception_head.'))
            elif stage == 'fusion':
                active = name.startswith(('visual.', 'fusion_head.')) and not name.startswith('visual.camera_bev.')
            else:
                active = True
            # These parameters are structurally unused by the visual branch.
            if name.startswith('visual.state_condition.'):
                active = False
            if self.config.temporal_context == 'single_frame' and name.startswith(
                    ('visual.fusion.', 'visual.temporal_attention.')):
                active = False
            parameter.requires_grad_(active)
        backbone = self.visual.camera_bev.backbone
        if hasattr(backbone,'apply_freezing'):
            backbone.apply_freezing()

    def forward(self, batch):
        if self.stage == 'ego':
            return self.ego(batch)
        if self.stage == 'perception':
            static = self.config.temporal_context == 'single_frame'
            bev, depth = self.visual.camera_bev(batch['images'][:, -1], batch['intrinsics'][:, -1],
                batch['camera_to_ego_static' if static else 'camera_to_ego'][:, -1])
            return dict(bev_logits=self.perception_head(bev), depth_logits=depth[:, None])
        visual = self.visual(batch)
        visual['bev_logits'] = self.perception_head(visual['current_bev'])
        if self.stage == 'visual':
            return visual
        ego = self.ego(batch)
        b = ego['final_deltas'].shape[0]
        keep = torch.ones(b, 1, 1, 1, device=ego['final_deltas'].device)
        if self.training:
            keep = (torch.rand_like(keep) >= self.ego_dropout).to(keep)
        shape = visual['final_deltas'].shape[:-1]
        ego_feature = ego['ego_features'][:, None, None].expand(*shape, -1) * keep
        ego_delta = ego['final_deltas'] * keep
        keep_expanded = keep.expand(*shape, 1)
        feature = torch.cat((visual['waypoint_features'], ego_feature,
                             visual['final_deltas'], ego_delta, keep_expanded), -1)
        base = keep*ego['final_deltas'] + (1-keep)*visual['final_deltas']
        final = base + self.fusion_head(feature)
        return dict(init_deltas=base, init_traj=base.cumsum(2), final_deltas=final,
                    final_traj=final.cumsum(2), depth_logits=visual['depth_logits'],
                    bev_logits=visual['bev_logits'], visual_outputs=visual)


def perception_loss(outputs, batch, bev_loss='balanced'):
    logits = outputs['depth_logits']
    targets = batch['depth_targets'][:, -logits.shape[1]:]
    count = (targets >= 0).sum()
    depth = logits.sum()*0
    if count > 0:
        depth = F.cross_entropy(logits.flatten(0, 2), targets.flatten(0, 2),
                                ignore_index=-1, reduction='sum') / count
    target = batch['bev_target']
    bev = bev_segmentation_loss(outputs['bev_logits'],target,bev_loss)
    return dict(depth=depth, bev=bev, total=.1*depth+bev)


def staged_loss(outputs, batch, stage, visual_weight=.5, bev_loss='balanced'):
    if stage == 'perception':
        return perception_loss(outputs, batch, bev_loss)
    terms = planning_loss(outputs, batch)
    if stage != 'ego':
        auxiliary = perception_loss(outputs, batch, bev_loss)
        terms['bev'] = auxiliary['bev']
        terms['total'] = terms['planning'] + auxiliary['total']
    if stage in ('fusion', 'joint'):
        # Keep a trajectory-supervised visual path during state dropout/fusion.
        terms['visual_planning'] = planning_loss(outputs['visual_outputs'], batch)['planning']
        terms['total'] = terms['total'] + visual_weight*terms['visual_planning']
    return terms
