"""L2 — pose-dependent non-rigid deformation (shared over time, a function of pose).

f(x_canonical, p_t) -> (dx(3), dq(4, residual), dlog_s(3), dcolor(3)); pure-PyTorch multires hash grid
Spatial encoding: multi-res dense grids via grid_sample (default) or a pure-PyTorch hash grid
(tinycudann not installed in sr4d; the torch hash grid was ~5x slower; docs/DECISIONS.md D-006).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PRIMES = (1, 2654435761, 805459861)


class HashGrid(nn.Module):
    def __init__(self, bbox_min, bbox_max, levels=12, log2_size=18, feat=2, base_res=16, max_res=1024):
        super().__init__()
        self.levels, self.feat = levels, feat
        self.size = 2 ** log2_size
        b = math.exp((math.log(max_res) - math.log(base_res)) / max(levels - 1, 1))
        self.res = [int(base_res * b ** l) for l in range(levels)]
        self.register_buffer("bmin", torch.as_tensor(bbox_min, dtype=torch.float32))
        self.register_buffer("bmax", torch.as_tensor(bbox_max, dtype=torch.float32))
        self.table_ = nn.Parameter(torch.empty(levels, self.size, feat).uniform_(-1e-4, 1e-4))
        corners = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.long)
        self.register_buffer("corners", corners)

    @property
    def out_dim(self):
        return self.levels * self.feat

    @property
    def table(self):
        return [self.table_]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = ((x - self.bmin) / (self.bmax - self.bmin)).clamp(0, 1)
        feats = []
        for l, r in enumerate(self.res):
            p = u * r
            p0 = p.floor().long()
            w = p - p0
            c = p0[:, None, :] + self.corners[None]  # (N,8,3)
            if (r + 1) ** 3 <= self.size:
                idx = (c[..., 0] * (r + 1) + c[..., 1]) * (r + 1) + c[..., 2]
            else:
                idx = (c[..., 0] * PRIMES[0]) ^ (c[..., 1] * PRIMES[1]) ^ (c[..., 2] * PRIMES[2])
            idx = idx % self.size
            f = self.table_[l][idx]  # (N,8,F)
            cw = torch.where(self.corners[None].bool(), w[:, None, :], 1 - w[:, None, :]).prod(-1)  # (N,8)
            feats.append((f * cw[..., None]).sum(1))
        return torch.cat(feats, -1)


class DenseGrid(nn.Module):
    """Multi-resolution dense 3D feature grids sampled with F.grid_sample (fast CUDA path)."""

    def __init__(self, bbox_min, bbox_max, resolutions=(32, 64, 128), feat=8):
        super().__init__()
        self.register_buffer("bmin", torch.as_tensor(bbox_min, dtype=torch.float32))
        self.register_buffer("bmax", torch.as_tensor(bbox_max, dtype=torch.float32))
        self.grids = nn.ParameterList([nn.Parameter(torch.empty(1, feat, r, r, r).uniform_(-1e-4, 1e-4)) for r in resolutions])
        self.feat = feat

    @property
    def out_dim(self):
        return self.feat * len(self.grids)

    @property
    def table(self):
        return list(self.grids)

    def forward(self, x):
        u = ((x - self.bmin) / (self.bmax - self.bmin)).clamp(0, 1) * 2 - 1
        g = u.flip(-1)[None, :, None, None, :]  # grid_sample expects (x,y,z) = (W,H,D)
        return torch.cat([F.grid_sample(gr, g, mode="bilinear", align_corners=True)[0, :, :, 0, 0].T for gr in self.grids], -1)


class PoseNonRigid(nn.Module):
    def __init__(self, bbox_min, bbox_max, pose_in: int, cfg):
        super().__init__()
        if cfg.get("encoding", "dense") == "hash":
            self.grid = HashGrid(bbox_min, bbox_max, cfg.hash_levels, cfg.hash_log2, 2, cfg.hash_base_res, cfg.hash_max_res)
        else:
            self.grid = DenseGrid(bbox_min, bbox_max, tuple(cfg.get("dense_res", [32, 64, 128])), cfg.get("dense_feat", 8))
        self.pose_mlp = nn.Sequential(nn.Linear(pose_in, 64), nn.SiLU(), nn.Linear(64, cfg.pose_dim))
        h = cfg.hidden
        self.mlp = nn.Sequential(nn.Linear(self.grid.out_dim + cfg.pose_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 12))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, pose_feat: torch.Tensor):
        """x (N,3) canonical, pose_feat (P,) -> dict of residuals."""
        code = self.pose_mlp(pose_feat[None])  # (1,D)
        h = torch.cat([self.grid(x), code.expand(x.shape[0], -1)], -1)
        o = self.mlp(h)
        dq = torch.cat([torch.ones_like(o[:, :1]), o[:, 3:6]], -1)  # (1 + small) -> normalised later
        return {"dx": o[:, 0:3] * 0.05, "dq": dq, "dlog_s": o[:, 6:9] * 0.1, "dcolor": o[:, 9:12] * 0.1}
