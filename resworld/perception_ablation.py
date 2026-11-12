"""Isolated perception candidates; historical training modules stay unchanged."""
from dataclasses import asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from .staged_driving import StagedDrivingModel, perception_loss
from .perception_objectives import bev_segmentation_loss


VARIANTS = ('baseline', 'spatial3', 'resolution384', 'focal_dice', 'channels64', 'center_aux')


def initialize_candidate(model, initialization, allowed_differences=()):
    """Copy every compatible tensor from one fresh, ImageNet-initialized control.

    This is experimental initialization, not resuming a trained checkpoint.
    Shape changes/new layers must be explicitly allowed and are recorded.
    """
    previous = initialization['model_config']
    requested = asdict(model.core.config)
    changed = {key for key in requested if requested[key] != previous[key]}
    if changed != set(allowed_differences):
        raise ValueError(f'unexpected initialization config changes: {changed}')
    source = initialization['model']
    destination = model.core.state_dict()
    copied = {key:value for key,value in source.items()
              if key in destination and value.shape == destination[key].shape}
    missing = sorted(set(destination)-set(copied))
    if not changed and missing:
        raise ValueError('identical configuration must load every core tensor')
    body_prefix = 'visual.camera_bev.backbone.body.'
    if any(key.startswith(body_prefix) for key in missing):
        raise ValueError('all ImageNet body tensors must be identical')
    model.core.load_state_dict(copied, strict=False)
    return dict(config_changes=sorted(changed), copied_tensors=len(copied),
                new_or_shape_changed_tensors=missing, trained_checkpoint=False)


class PerceptionCandidate(nn.Module):
    def __init__(self, config, center_aux=False):
        super().__init__()
        self.core = StagedDrivingModel(config, 'perception')
        self.config = self.core.config
        self.center_aux = center_aux
        if center_aux:
            c = config.channels
            self.center_head = nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(4,c),
                                             nn.SiLU(),nn.Conv2d(c,6,1))
            with torch.no_grad():
                self.center_head[-1].bias[:2].fill_(-2.197224577)

    def forward(self, batch):
        # Exactly the same current-frame path as historical perception training.
        bev, depth = self.core.visual.camera_bev(batch['images'][:,-1],
            batch['intrinsics'][:,-1],batch['camera_to_ego'][:,-1])
        result = dict(bev_logits=self.core.perception_head(bev), depth_logits=depth[:,None])
        if self.center_aux:
            center = self.center_head(bev)
            result.update(center_logits=center[:,:2], center_offsets=center[:,2:].reshape(
                center.shape[0],2,2,*center.shape[-2:]))
        return result


def focal_dice_loss(logits, target, gamma=2.):
    valid = target != 255
    if not valid.any():
        return logits.sum()*0
    ce = F.cross_entropy(logits,target,ignore_index=255,reduction='none')
    focal = ((1-ce[valid].neg().exp()).pow(gamma)*ce[valid]).mean()
    # Keep precisely the historical foreground Dice, changing only CE to focal CE.
    return bev_segmentation_loss(logits,target,'ce_dice')-ce[valid].mean()+focal


def make_center_targets(objects, grid, valid_mask):
    """Gaussian center heatmaps and fractional row/column offsets.

    Same-class centers sharing a cell cannot be represented independently; use
    the center nearest the grid point and record collisions rather than averaging.
    """
    h,w = grid.height,grid.width
    heatmap = torch.zeros(2,h,w)
    offsets = torch.zeros(2,2,h,w)
    mask = torch.zeros(2,h,w,dtype=torch.bool)
    best_distance = torch.full((2,h,w),float('inf'))
    axis = torch.arange(-3,4,dtype=torch.float32)
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    gaussian = (-(xx.square()+yy.square())/2).exp()
    collisions = 0
    for row in objects:
        x,y = row['xy']
        if not (grid.x_min<=x<=grid.x_max and grid.y_min<=y<=grid.y_max):
            continue
        c = row['label']-1
        rf = (grid.x_max-x)*(h-1)/(grid.x_max-grid.x_min)
        cf = (grid.y_max-y)*(w-1)/(grid.y_max-grid.y_min)
        r,col = int(round(rf)),int(round(cf))
        r0,r1,c0,c1 = max(0,r-3),min(h,r+4),max(0,col-3),min(w,col+4)
        heatmap[c,r0:r1,c0:c1] = torch.maximum(heatmap[c,r0:r1,c0:c1],
            gaussian[r0-r+3:r1-r+3,c0-col+3:c1-col+3])
        distance = (rf-r)**2+(cf-col)**2
        if mask[c,r,col]: collisions += 1
        if distance < best_distance[c,r,col]:
            offsets[c,:,r,col] = torch.tensor([rf-r,cf-col])
            best_distance[c,r,col] = distance
        mask[c,r,col] = True
    valid = valid_mask[None].expand(2,-1,-1).clone() | (heatmap == 1)
    return dict(center_heatmap=heatmap, center_target_offsets=offsets,
                center_mask=mask, center_valid=valid,
                center_collisions=torch.tensor(collisions))


def center_loss(outputs, batch):
    logits = outputs['center_logits']
    probability = logits.sigmoid()
    target = batch['center_heatmap']
    positive = target == 1
    negative = target < 1
    weights = (1-target).pow(4)
    terms = (-F.logsigmoid(logits)*(1-probability).pow(2)*positive
             -F.logsigmoid(-logits)*probability.pow(2)*weights*negative)
    heatmap = (terms*batch['center_valid']).sum()/positive.sum().clamp_min(1)
    mask = batch['center_mask'][:,:,None].expand_as(outputs['center_offsets'])
    offsets = F.smooth_l1_loss(outputs['center_offsets'][mask],
        batch['center_target_offsets'][mask],reduction='sum')/mask.sum().clamp_min(1)
    return heatmap,offsets


def candidate_loss(outputs, batch, focal=False, center_weight=.1):
    terms = perception_loss(outputs,batch,'ce_dice')
    if focal:
        replacement = focal_dice_loss(outputs['bev_logits'],batch['bev_target'])
        terms['total'] = terms['total']-terms['bev']+replacement
        terms['bev'] = replacement
    if 'center_logits' in outputs:
        heatmap,offsets = center_loss(outputs,batch)
        terms.update(center_heatmap=heatmap,center_offsets=offsets)
        terms['total'] = terms['total']+center_weight*(heatmap+offsets)
    return terms
