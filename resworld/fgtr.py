import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sample_waypoint_features(future_bev,reference_grid, padding_mode="zeros"):

    grid=reference_grid.unsqueeze(2)

    sampled=F.grid_sample(
        future_bev,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True
    )

    return sampled.squeeze(-1).transpose(1,2)

class WaypointDeformableSampler(nn.Module):

    def __init__(self,dim,num_points=4,max_offset_pixels=2.0):

        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if not isinstance(num_points, int) or isinstance(num_points, bool) or num_points <= 0:
            raise ValueError("num_points must be a positive integer")
        if not math.isfinite(max_offset_pixels) or max_offset_pixels <= 0:
            raise ValueError("max_offset_pixels must be finite and positive")
        self.dim=dim
        self.num_points=num_points
        self.max_offset_pixels=float(max_offset_pixels)

        self.offset_head=nn.Linear(dim,num_points*2)
        self.weight_head=nn.Linear(dim,num_points)


    def forward(self,query,future_bev,reference_grid):
        """Sample [B,N,D] queries from a single [B,C,H,W] BEV per sample.

        References and returned locations use normalized (x,y) coordinates.
        Each offset component is bounded in BEV pixels. Locations may leave
        [-1,1]; border padding then reads the nearest edge feature.
        """
        if query.ndim != 3:
            raise ValueError("query must be [B, N, D]")
        if future_bev.ndim != 4:
            raise ValueError("future_bev must be [B, C, H, W]")

        batch,num_queries,dim=query.shape
        bev_batch,channels,height,width=future_bev.shape
        if dim != self.dim:
            raise ValueError("query feature dimension must match dim")
        if bev_batch != batch:
            raise ValueError("query and future_bev must have matching batch sizes")
        if reference_grid.shape != (batch,num_queries,2):
            raise ValueError("reference_grid must be [B, N, 2]")
        if height <= 1 or width <= 1:
            raise ValueError("BEV height and width must both be greater than 1")
        if not (query.device == future_bev.device == reference_grid.device):
            raise ValueError("all inputs must be on the same device")
        if not (query.dtype == future_bev.dtype == reference_grid.dtype):
            raise ValueError("all inputs must have the same dtype")
        if not query.is_floating_point():
            raise ValueError("inputs must be floating-point tensors")
        points=self.num_points

        raw_offsets=self.offset_head(query).reshape(batch,num_queries,points,2)
        offsets=raw_offsets.tanh()*self.max_offset_pixels
        weights=self.weight_head(query).softmax(dim=-1)

        scale=query.new_tensor([
            2.0/(width-1),
            2.0/(height-1),
        ])

        locations=(reference_grid.unsqueeze(2)+offsets*scale)

        sampled=sample_waypoint_features(
            future_bev,
            locations.reshape(batch,num_queries*points,2),
            padding_mode="border",
        )

        sampled=sampled.reshape(batch,num_queries,points,channels)

        context=(sampled*weights.unsqueeze(-1)).sum(dim=2)

        return context,weights,locations


def sample_future_waypoints(sampler, query, future_bev, reference_grid):
    """Match each waypoint to its future timestep's BEV.

    Inputs: query [B,M,T,D], BEV [B,T,C,H,W], references [B,M,T,2].
    Returns: context [B,M,T,C], weights [B,M,T,P], locations [B,M,T,P,2].
    M is the number of trajectory modes; T is the number of future steps.
    """
    if query.ndim != 4:
        raise ValueError("query must be [B, M, T, D]")
    if future_bev.ndim != 5:
        raise ValueError("future_bev must be [B, T, C, H, W]")
    batch,modes,steps,dim=query.shape
    bev_batch,bev_steps,channels,height,width=future_bev.shape
    if (bev_batch,bev_steps) != (batch,steps):
        raise ValueError("query and future_bev must have matching B and T")
    if reference_grid.shape != (batch,modes,steps,2):
        raise ValueError("reference_grid must be [B, M, T, 2]")

    # Move time before modes so flattening B,T preserves BEV correspondence.
    query_flat=query.permute(0,2,1,3).reshape(batch*steps,modes,dim)
    reference_flat=reference_grid.permute(0,2,1,3).reshape(batch*steps,modes,2)
    bev_flat=future_bev.reshape(batch*steps,channels,height,width)
    context,weights,locations=sampler(query_flat,bev_flat,reference_flat)
    points=sampler.num_points
    context=context.reshape(batch,steps,modes,channels).permute(0,2,1,3)
    weights=weights.reshape(batch,steps,modes,points).permute(0,2,1,3)
    locations=locations.reshape(batch,steps,modes,points,2).permute(0,2,1,3,4)
    return context,weights,locations


class FutureGuidedTrajectoryRefinement(nn.Module):

    def __init__(self,channels,num_points=4):

        super().__init__()

        self.query_embed=nn.Linear(2,channels)

        self.sampler=WaypointDeformableSampler(
            dim=channels,
            num_points=num_points,
            max_offset_pixels=2.0
        )

        self.delta_head=nn.Sequential(
            nn.Linear(channels*2,channels),
            nn.ReLU(),
            nn.Linear(channels,2),
        )


    def forward(self,prior_trajectory,future_bev):
        """Refine normalized [B,M,T,2] waypoints with [B,T,C,H,W] BEVs.

        The initial trajectory itself supplies the reference grid. Coordinates
        are already normalized (x,y); no metric-to-grid conversion occurs here.
        """

        query=self.query_embed(prior_trajectory)

        context,weights,locations=sample_future_waypoints(
            sampler=self.sampler,
            query=query,
            future_bev=future_bev,
            reference_grid=prior_trajectory
        )

        features=torch.cat((query,context),dim=-1)
        correction=self.delta_head(features)

        refined_trajectory=(prior_trajectory+correction).clamp(-1.0,1.0)

        return {
            "refined_trajectory":refined_trajectory,
            "correction":correction,
            "context":context,
            "weights":weights,
            "locations":locations
        }
    
if __name__=="__main__":

    bev=torch.arange(25,dtype=torch.float32).reshape(1,1,5,5)

    references=torch.tensor([[
        [-1.,-1.],
        [0.,0.],
        [1.,1.]
    ]])

    features=sample_waypoint_features(bev,references)

    print("shape:",features.shape)
    print("values:",features[0,:,0])

    assert features.shape==(1,3,1)
    assert torch.allclose(
        features[0,:,0],
        torch.tensor([0.,12.,24.])
    )
