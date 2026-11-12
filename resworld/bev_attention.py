"""BEVFormer-inspired sparse spatial/temporal attention in native PyTorch.

This is a scaled research variant, not a checkpoint-compatible BEVFormer port.
Spatial queries project 3D height anchors into calibrated cameras. Temporal
attention samples locally from already ego-motion-aligned BEVs. No N*N BEV
attention matrix is allocated. Source: https://arxiv.org/abs/2203.17270
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from .image_backbone import ResNet50FPN


def masked_softmax(logits, valid, dim=-1):
    weights = logits.masked_fill(~valid, -1e4).softmax(dim) * valid.to(logits.dtype)
    return weights / weights.sum(dim, keepdim=True).clamp_min(1e-8)


def project_anchors(points, intrinsics, camera_to_ego):
    """Ego xyz [N,A,3] -> input-image uv/depth [B,V,N,A,...]."""
    ego_to_camera = torch.linalg.inv(camera_to_ego)
    camera = torch.einsum('bvij,naj->bvnai', ego_to_camera[..., :3, :3], points)
    camera = camera + ego_to_camera[..., :3, 3][:, :, None, None]
    pixels = torch.einsum('bvij,bvnaj->bvnai', intrinsics, camera)
    depth = camera[..., 2]
    uv = pixels[..., :2] / depth.clamp_min(1e-5)[..., None]
    return uv, depth


class SpatialCrossAttention(nn.Module):
    def __init__(self, channels, heads, levels, anchors, points=4, chunk_size=256):
        super().__init__()
        self.heads, self.levels = heads, levels
        self.anchors, self.points = anchors, points
        self.chunk_size = chunk_size
        self.offsets = nn.Linear(channels, heads*levels*anchors*points*2)
        self.weights = nn.Linear(channels, heads*levels*anchors*points)
        self.view_weights = nn.Linear(channels, 6)
        self.output = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels*2), nn.GELU(), nn.Linear(channels*2, channels))
        self.ffn_norm = nn.LayerNorm(channels)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        nn.init.zeros_(self.view_weights.weight)
        nn.init.zeros_(self.view_weights.bias)
        angles = torch.arange(heads) * (2*math.pi/heads)
        directions = torch.stack((angles.cos(), angles.sin()), -1)
        radial = torch.linspace(.1, .8, points)
        bias = directions[:, None, None, None] * radial[None, None, None, :, None]
        bias = bias.expand(heads, levels, anchors, points, 2).clamp(-.9, .9).atanh()
        with torch.no_grad():
            self.offsets.bias.copy_(bias.reshape(-1))

    def forward(self, query, features, uv, anchor_visible, strides):
        b, n, c = query.shape
        heads, levels, anchors, points = self.heads, self.levels, self.anchors, self.points
        views = features[0].shape[1]
        dim = c // heads
        chunks, coverage = [], []
        for start in range(0, n, self.chunk_size):
            q = query[:, start:start+self.chunk_size]
            size = q.shape[1]
            offset = self.offsets(q).reshape(b, size, heads, levels, anchors, points, 2).tanh()*2
            logits = self.weights(q).reshape(b, size, heads, levels, anchors, points)
            view_logits = self.view_weights(q)[:, :, None, :, None, None, None]
            logits = logits[:, :, :, None] + view_logits
            samples, valids = [], []
            for level, (feature, stride) in enumerate(zip(features, strides)):
                fh, fw = feature.shape[-2:]
                scale = q.new_tensor([max(1, fw-1), max(1, fh-1)])
                reference = uv[:, :, start:start+size] / stride
                reference = reference.permute(0, 2, 1, 3, 4)[:, :, None, :, :, None]
                locations = (reference + offset[:, :, :, None, level]) / scale * 2 - 1
                valid = anchor_visible[:, :, start:start+size].permute(0, 2, 1, 3)
                valid = valid[:, :, None, :, :, None] & (locations.abs() <= 1).all(-1)
                valids.append(valid)
                grid = locations.permute(0, 3, 2, 1, 4, 5, 6).reshape(b*views*heads, size, anchors*points, 2)
                values = feature.reshape(b, views, heads, dim, fh, fw).reshape(b*views*heads, dim, fh, fw)
                sampled = F.grid_sample(values, grid, align_corners=True, padding_mode='zeros')
                samples.append(sampled.reshape(b, views, heads, dim, size, anchors, points)
                               .permute(0, 4, 2, 1, 5, 6, 3))
            valid = torch.stack(valids, dim=4)  # B,N,H,V,L,A,P
            attention = masked_softmax(logits.flatten(3), valid.flatten(3)).reshape_as(valid)
            values = torch.stack(samples, dim=4)  # B,N,H,V,L,A,P,D
            context = (values * attention[..., None]).sum((3, 4, 5, 6)).reshape(b, size, c)
            visible = valid.flatten(2).any(-1)
            # Learned projection bias must not inject camera observations into unseen cells.
            update = self.output(context) * visible[..., None]
            q = self.norm(q + update)
            q = self.ffn_norm(q + self.ffn(q))
            chunks.append(q)
            coverage.append(visible)
        return torch.cat(chunks, 1), torch.cat(coverage, 1)


class BEVQueryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.channels
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, c, 3, stride=2, padding=1), nn.GroupNorm(4, c), nn.SiLU())
        self.pyramid = nn.Sequential(nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GroupNorm(4, c), nn.SiLU())
        if config.image_backbone == 'resnet50':
            self.backbone = ResNet50FPN(c)
            self.pyramid = nn.Identity()
        self.depth_head = nn.Conv2d(c, config.depth_bins, 1)  # optional auxiliary supervision
        self.camera_embedding = nn.Parameter(torch.randn(6, c)*.02)
        self.queries = nn.Parameter(torch.randn(config.bev_height*config.bev_width, c)*.02)
        self.position = nn.Linear(2, c)
        self.spatial = nn.ModuleList([
            SpatialCrossAttention(c, config.heads, 2, len(config.anchor_heights),
                                  config.spatial_points, config.query_chunk_size)
            for _ in range(config.spatial_layers)])

    def forward(self, images, intrinsics, camera_to_ego):
        b, views, _, ih, iw = images.shape
        if views != 6:
            raise ValueError('BEV query encoder expects six calibrated cameras')
        cfg = self.config
        backbone_output = self.backbone(images.reshape(b*views, 3, ih, iw))
        feature = backbone_output[0] if cfg.image_backbone == 'resnet50' else backbone_output
        depth_logits = self.depth_head(feature)
        pyramid = list(backbone_output) if cfg.image_backbone == 'resnet50' else [feature,self.pyramid(feature)]
        features = [f.reshape(b, views, cfg.channels, *f.shape[-2:])
                    + self.camera_embedding[None, :, :, None, None] for f in pyramid]
        xy = cfg.grid().target_metric_points(1, images.device, images.dtype)[0, ..., :2].reshape(-1, 2)
        heights = images.new_tensor(cfg.anchor_heights)
        xyz = torch.cat((xy[:, None].expand(-1, len(heights), -1),
                         heights[None, :, None].expand(len(xy), -1, -1)), -1)
        uv, depth = project_anchors(xyz, intrinsics, camera_to_ego)
        visible = ((depth > .1) & (uv[..., 0] >= 0) & (uv[..., 0] < iw)
                   & (uv[..., 1] >= 0) & (uv[..., 1] < ih))
        scale = images.new_tensor([max(abs(cfg.x_min), abs(cfg.x_max)),
                                  max(abs(cfg.y_min), abs(cfg.y_max))])
        query = (self.queries + self.position(xy / scale))[None].expand(b, -1, -1)
        for layer in self.spatial:
            query, coverage = layer(query, features, uv, visible, (8, 16))
        self.last_visibility = coverage.reshape(b, 1, cfg.bev_height, cfg.bev_width)
        bev = query.transpose(1, 2).reshape(b, cfg.channels, cfg.bev_height, cfg.bev_width)
        return bev, depth_logits.reshape(b, views, cfg.depth_bins, *feature.shape[-2:])


class TemporalBEVAttention(nn.Module):
    """Local multihead attention over current plus two ego-aligned history BEVs."""
    def __init__(self, channels, heads, points=4, chunk_size=512, max_offset_meters=4.):
        super().__init__()
        self.heads, self.points, self.chunk_size = heads, points, chunk_size
        self.max_offset_meters = max_offset_meters
        self.offsets = nn.Linear(channels, heads*3*points*2)
        self.weights = nn.Linear(channels, heads*3*points)
        self.output = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels*2), nn.GELU(), nn.Linear(channels*2, channels))
        self.ffn_norm = nn.LayerNorm(channels)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def forward(self, current, aligned, masks, grid):
        b, c, h, w = current.shape
        heads, points, dim = self.heads, self.points, c//self.heads
        query = current.flatten(2).transpose(1, 2)
        metric = grid.target_metric_points(b, current.device, current.dtype)
        reference = grid.source_points_to_grid(metric).reshape(b, h*w, 2)
        values = torch.stack(aligned, 1).reshape(b, 3, heads, dim, h, w)
        values = values.permute(0, 2, 1, 3, 4, 5).reshape(b*heads*3, dim, h, w)
        visibility = torch.stack(masks, 1).float()  # B,T,1,H,W
        visibility = visibility[:, None].expand(-1, heads, -1, -1, -1, -1).reshape(b*heads*3, 1, h, w)
        chunks = []
        for start in range(0, h*w, self.chunk_size):
            q = query[:, start:start+self.chunk_size]
            n = q.shape[1]
            offsets = self.offsets(q).reshape(b, n, heads, 3, points, 2).tanh()
            # Bound the search in meters so changing BEV resolution preserves its reach.
            scale = q.new_tensor([2*self.max_offset_meters/(grid.y_max-grid.y_min),
                                  2*self.max_offset_meters/(grid.x_max-grid.x_min)])
            locations = reference[:, start:start+n, None, None, None] + offsets*scale
            sample_grid = locations.permute(0, 2, 3, 1, 4, 5).reshape(b*heads*3, n, points, 2)
            sampled = F.grid_sample(values, sample_grid, align_corners=True)
            sampled = sampled.reshape(b, heads, 3, dim, n, points).permute(0, 4, 1, 2, 5, 3)
            visible = F.grid_sample(visibility, sample_grid, mode='nearest', align_corners=True)
            visible = visible.reshape(b, heads, 3, n, points).permute(0, 3, 1, 2, 4) > .5
            visible = visible & (locations.abs() <= 1).all(-1)
            weights = masked_softmax(self.weights(q).reshape(b, n, heads, 3*points), visible.flatten(3))
            context = (sampled.flatten(3, 4) * weights[..., None]).sum(3).reshape(b, n, c)
            any_visible = visible.flatten(2).any(-1)
            q = self.norm(q + self.output(context)*any_visible[..., None])
            chunks.append(self.ffn_norm(q + self.ffn(q)))
        return torch.cat(chunks, 1).transpose(1, 2).reshape(b, c, h, w)
