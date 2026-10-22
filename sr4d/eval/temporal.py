"""Temporal consistency metrics.

temporal_error (CLAUDE.md 8, CSV column): mean |(R_t - R_{t-1}) - (G_t - G_{t-1})| over the foreground.
  NOTE: a completely static render scores exactly mean|G_t - G_{t-1}|, so this metric favours renders that
  under-move (measured on ZJU-387: 4DGS |dR|=0.021 vs GT |dG|=0.041). Report it together with warp_error.
tof (JSON only): temporal optical-flow error = mean endpoint error between the flow estimated on the RENDER
  sequence and on the GT sequence (torchvision RAFT-large), over the foreground. Unlike the two-frame
  difference and plain warping error (both measured to favour blurry / under-moving renders), a static render
  scores |F_G| and a flickering one scores > 0; only correct motion scores ~0. Computed between frames t-k and t
  (metrics.py --tof_stride, default 5): on ZJU-387 (50 fps) the GT moves only ~0.45 px/frame, so stride-1 values
  are dominated by noise (a static render would score 0.45 px; measured SR4D 0.45, 4DGS 0.29).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_error(R_t, G_t, R_p, G_p, mask) -> float:
    d = ((R_t - R_p) - (G_t - G_p)).abs().mean(-1)
    return float((d * mask).sum() / mask.sum().clamp_min(1))


class FlowWarp:
    def __init__(self, device="cuda"):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.w = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.w).to(device).eval()
        self.device = device

    @torch.no_grad()
    def flow_backward(self, G_prev, G_cur):
        """Flow from current to previous frame (for backward warping prev -> cur). Inputs (H,W,3) in [0,1]."""
        H, W = G_cur.shape[:2]
        H8, W8 = (H + 7) // 8 * 8, (W + 7) // 8 * 8
        a = F.interpolate(G_cur.permute(2, 0, 1)[None], (H8, W8), mode="bilinear", align_corners=False) * 2 - 1
        b = F.interpolate(G_prev.permute(2, 0, 1)[None], (H8, W8), mode="bilinear", align_corners=False) * 2 - 1
        fl = self.model(a, b)[-1]
        fl = F.interpolate(fl, (H, W), mode="bilinear", align_corners=False)
        fl[:, 0] *= W / W8
        fl[:, 1] *= H / H8
        return fl[0]  # (2,H,W)

    @staticmethod
    def warp(img_prev, flow):
        H, W = img_prev.shape[:2]
        ys, xs = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device), indexing="ij")
        gx = (xs + flow[0]) / (W - 1) * 2 - 1
        gy = (ys + flow[1]) / (H - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        return F.grid_sample(img_prev.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=True)[0].permute(1, 2, 0)

    def tof(self, R_t, R_p, G_t, G_p, mask) -> float:
        fr = self.flow_backward(R_p, R_t)
        fg = self.flow_backward(G_p, G_t)
        epe = (fr - fg).norm(dim=0)
        m = mask.float()
        return float((epe * m).sum() / m.sum().clamp_min(1))
