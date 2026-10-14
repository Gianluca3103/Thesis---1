"""Training-only physical-radius radar/LiDAR correspondence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from .config import VoxelGrid
from .sparse import SparseSites


@dataclass
class Neighborhood:
    indices: torch.Tensor  # [R,K], -1 denotes padding
    distances_m: torch.Tensor  # [R,K], inf at padding

    @property
    def valid(self) -> torch.Tensor:
        return self.indices >= 0


def local_neighbors(radar: SparseSites, lidar: SparseSites, grid: VoxelGrid, radius_m: float, max_neighbors: int) -> Neighborhood:
    """Batched KD-tree radius lookup in physical XYZ, never global attention."""
    if radar.stride != lidar.stride or radius_m <= 0 or max_neighbors < 1:
        raise ValueError("Incompatible scale or local neighborhood settings")
    return local_neighbors_xyz(radar.centers_xyz(grid),radar.coords[:,0],lidar,grid,
                               radius_m,max_neighbors)


def local_neighbors_xyz(query_xyz: torch.Tensor, query_batch: torch.Tensor,
                        lidar: SparseSites, grid: VoxelGrid, radius_m: float,
                        max_neighbors: int) -> Neighborhood:
    """Physical clean-surface neighbors for arbitrary radar-pattern query sites."""
    if (query_xyz.ndim != 2 or query_xyz.shape[1] != 3 or
            query_batch.shape != (len(query_xyz),) or radius_m <= 0 or max_neighbors < 1):
        raise ValueError("Invalid physical query neighborhood settings")
    idx = torch.full((len(query_xyz),max_neighbors),-1,dtype=torch.long,device=query_xyz.device)
    distances = query_xyz.new_full(idx.shape,float("inf"))
    if not len(query_xyz) or not len(lidar.coords):
        return Neighborhood(idx,distances)
    rxyz = query_xyz.detach().cpu().numpy()
    lxyz = lidar.centers_xyz(grid).detach().cpu().numpy()
    rb = query_batch.detach().cpu().numpy()
    lb = lidar.coords[:,0].detach().cpu().numpy()
    for batch_id in np.unique(rb):
        ri = np.flatnonzero(rb==batch_id)
        li = np.flatnonzero(lb==batch_id)
        if not len(li):
            continue
        tree = cKDTree(lxyz[li])
        delta,local = tree.query(rxyz[ri],k=max_neighbors,distance_upper_bound=radius_m,workers=-1)
        delta=np.asarray(delta).reshape(len(ri),max_neighbors)
        local=np.asarray(local).reshape(len(ri),max_neighbors)
        present=np.isfinite(delta)
        mapped=np.full(local.shape,-1,dtype=np.int64)
        mapped[present]=li[local[present]]
        idx[torch.as_tensor(ri,device=idx.device)] = torch.as_tensor(mapped,device=idx.device)
        distances[torch.as_tensor(ri,device=idx.device)] = torch.as_tensor(delta,device=distances.device,dtype=distances.dtype)
    return Neighborhood(idx,distances)


class LocalCorrespondence(nn.Module):
    """Train radar/LiDAR feature alignment with local geometric proxy labels."""

    def __init__(self, radar_channels: int, lidar_channels: int, dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.query = nn.Linear(radar_channels,dim,bias=False)
        self.key_value = nn.Linear(lidar_channels,dim,bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, radar_z: SparseSites, lidar: SparseSites, grid: VoxelGrid, neighborhood: Neighborhood,
                radius_m: float, positive_radius_m: float, temperature: float = 0.1,
                negative_strategy: str = "nearest", num_negatives: int = 16,
                *, defer_diagnostics: bool = False) -> dict:
        n,k = neighborhood.indices.shape
        if n != len(radar_z.coords):
            raise ValueError("Neighborhood and radar site counts differ")
        if n == 0:
            zero=radar_z.features.sum()*0
            return {"loss":zero,"valid_queries":0,"positive_queries":0,"contrastive_queries":0,"weights":zero.new_zeros((0,k)),"logits":zero.new_zeros((0,k)),"positive_mask":torch.zeros((0,k),dtype=torch.bool,device=zero.device)}
        q=F.normalize(self.query(self.dropout(radar_z.features)),dim=-1)
        all_k=F.normalize(self.key_value(self.dropout(lidar.features)),dim=-1)
        if len(all_k)==0:
            zero=radar_z.features.sum()*0
            return {"loss":zero,"valid_queries":0,"positive_queries":0,"contrastive_queries":0,"weights":zero.new_zeros((n,k)),"logits":zero.new_full((n,k),-1e4),"positive_mask":torch.zeros((n,k),dtype=torch.bool,device=zero.device)}
        valid=neighborhood.valid
        cand=all_k[neighborhood.indices.clamp_min(0)]
        similarity=(q[:,None,:]*cand).sum(-1)
        # XYZ selects candidates and defines supervision, but cannot solve the
        # feature-ranking task through a learned distance/position shortcut.
        logits=similarity/temperature
        logits=logits.masked_fill(~valid,-1e4)
        weights=torch.softmax(logits,dim=-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        positive=valid & (neighborhood.distances_m <= positive_radius_m)
        negatives=valid & ~positive
        if negative_strategy == "nearest":
            priority=-neighborhood.distances_m
        elif negative_strategy == "random_local":
            if self.training:
                priority=torch.rand_like(neighborhood.distances_m)
            else:
                # Same local-negative strategy with reproducible validation.
                rows=torch.arange(n,device=neighborhood.indices.device)[:,None]
                hashed=(neighborhood.indices.clamp_min(0)*1103515245 + rows*12345) % 2147483647
                priority=hashed.to(neighborhood.distances_m.dtype)/2147483647
        else:
            raise ValueError(f"Unknown negative strategy: {negative_strategy}")
        priority=priority.masked_fill(~negatives,-float("inf"))
        selected=torch.zeros_like(negatives)
        selected.scatter_(1,priority.topk(min(num_negatives,k),dim=1).indices,True)
        selected &= negatives
        eligible=positive|selected
        use=positive.any(-1) & selected.any(-1)
        positive_logprob=torch.logsumexp(logits.masked_fill(~positive,-1e4),dim=-1)-torch.logsumexp(logits.masked_fill(~eligible,-1e4),dim=-1)
        corr=(-positive_logprob*use).sum()/use.sum().clamp_min(1)
        valid_queries=valid.any(-1).sum()
        positive_queries=positive.any(-1).sum()
        contrastive_queries=use.sum()
        if not defer_diagnostics:
            valid_queries,positive_queries,contrastive_queries=map(int,(valid_queries,positive_queries,contrastive_queries))
        return {"loss":corr,"valid_queries":valid_queries,"positive_queries":positive_queries,"contrastive_queries":contrastive_queries,"weights":weights,"logits":logits,"positive_mask":positive}
