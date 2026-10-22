"""L4 — view absorber: per-view appearance code -> colour affine; optional per-view camera refinement
(6DoF, bounded) and per-view low-res pixel-confidence logits (generated views only).
Discarded at render time: the mean code is used and no camera refinement / confidence is applied.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sr4d.utils.geometry import rotvec_to_rotmat


class ViewAbsorber(nn.Module):
    def __init__(self, n_views: int, cfg, image_hw=None):
        super().__init__()
        self.codes = nn.Parameter(torch.zeros(n_views, cfg.code_dim))
        self.mlp = nn.Sequential(nn.Linear(cfg.code_dim, 64), nn.SiLU(), nn.Linear(64, 12))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.color = bool(cfg.get("color", True))  # D-024: per-view colour affine (A-stage only)
        self.camera_refine = cfg.camera_refine
        self.confidence = cfg.confidence
        self.conf_norm = bool(cfg.get("conf_norm", False))
        if cfg.camera_refine:
            self.cam_delta = nn.Parameter(torch.zeros(n_views, 6))
            self.max_rot = math.radians(cfg.max_cam_rot_deg)
            self.max_trans = cfg.max_cam_trans_m
        if cfg.confidence:
            self.conf_logits = nn.Parameter(torch.zeros(n_views, 1, cfg.conf_res, cfg.conf_res))

    def affine(self, view: int | None):
        if not self.color:
            dev = self.codes.device
            return torch.eye(3, device=dev), torch.zeros(3, device=dev)
        code = self.codes.mean(0) if view is None else self.codes[view]
        o = self.mlp(code)
        return torch.eye(3, device=o.device) + o[:9].reshape(3, 3) * 0.1, o[9:12] * 0.1

    def refine_w2c(self, view: int, w2c: torch.Tensor) -> torch.Tensor:
        if not self.camera_refine:
            return w2c
        d = self.cam_delta[view]
        R = rotvec_to_rotmat(torch.tanh(d[:3]) * self.max_rot)
        t = torch.tanh(d[3:]) * self.max_trans
        D = torch.eye(4, device=w2c.device)
        D[:3, :3] = R
        D[:3, 3] = t
        return D @ w2c

    def confidence_map(self, view: int, H: int, W: int) -> torch.Tensor | None:
        if not self.confidence:
            return None
        l = F.interpolate(self.conf_logits[view:view + 1], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        w = torch.exp(l.clamp(-3, 3))
        if self.conf_norm:  # D-034: per-image mean 1, so w only redistributes weight (saturation made it a global x20)
            w = w / w.mean().clamp_min(1e-6)
        return w

    def reg_loss(self) -> torch.Tensor:
        l = (self.codes ** 2).sum(-1).mean()
        if self.camera_refine:
            l = l + (self.cam_delta ** 2).sum(-1).mean()
        return l
