"""Losses (CLAUDE.md 7.6)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def photometric(pred: torch.Tensor, gt: torch.Tensor, lam: float = 0.2, weight: torch.Tensor | None = None):
    """(1-lam) L1 + lam D-SSIM; pred/gt (H,W,3). Optional per-pixel weight (H,W) on the L1 term."""
    l1 = (pred - gt).abs().mean(-1)
    if weight is not None:
        l1 = l1 * weight
    l1 = l1.mean()
    s = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0, size_average=True)
    return (1 - lam) * l1 + lam * (1 - s)


def mask_bbox(mask: torch.Tensor, pad: int = 16):
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    H, W = mask.shape
    if ys.numel() == 0:
        return 0, H, 0, W
    return max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, H), max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, W)


def lpips_loss(fn, pred, gt, mask):
    y0, y1, x0, x1 = mask_bbox(mask)
    a = pred[y0:y1, x0:x1].permute(2, 0, 1)[None] * 2 - 1
    b = gt[y0:y1, x0:x1].permute(2, 0, 1)[None] * 2 - 1
    return fn(a, b).mean()


def mask_loss(alpha, mask):
    return (alpha - mask).abs().mean()


def normal_consistency(normal_p, depth_normal, alpha):
    """1 - cos between alpha-composited normal and normal from rendered depth (2DGS-style)."""
    n = F.normalize(normal_p, dim=-1)
    d = F.normalize(depth_normal, dim=-1)
    w = (alpha > 0.5).float().detach()
    return ((1 - (n * d).sum(-1)) * w).sum() / w.sum().clamp_min(1)


def anisotropy(scales: torch.Tensor, r: float = 4.0):
    """Penalty on max/min scale ratio above r (only the 2 in-plane axes for surfels)."""
    ratio = scales.max(-1).values / scales.min(-1).values.clamp_min(1e-8)
    return F.relu(ratio - r).mean()


def opacity_entropy(opac: torch.Tensor):
    o = opac.clamp(1e-4, 1 - 1e-4)
    return (-(o * o.log() + (1 - o) * (1 - o).log())).mean()


def body_alignment(offset: torch.Tensor, normal_dot: torch.Tensor):
    """SAGA-style: keep detached surfels near the mesh (position) and parallel to it (normal)."""
    return (offset ** 2).sum(-1).mean(), (1 - normal_dot.abs()).mean()


def l2_magnitude(d: dict):
    return (d["dx"] ** 2).sum(-1).mean() * 100 + (d["dq"][:, 1:] ** 2).sum(-1).mean() + (d["dlog_s"] ** 2).sum(-1).mean() \
        + (d["dcolor"] ** 2).sum(-1).mean()


def l2_smoothness(l2_module, x: torch.Tensor, pose_feat: torch.Tensor, n: int = 4096, eps: float = 0.01):
    """Spatial smoothness: outputs at x and x+noise (eps m) should agree."""
    idx = torch.randint(0, x.shape[0], (min(n, x.shape[0]),), device=x.device)
    xs = x[idx].detach()
    a = l2_module(xs, pose_feat)
    b = l2_module(xs + torch.randn_like(xs) * eps, pose_feat)
    return sum(((a[k] - b[k]) ** 2).sum(-1).mean() for k in ("dx", "dlog_s", "dcolor")) * 100


def confidence_loss(res: torch.Tensor, w: torch.Tensor):
    """w * residual - log w (uncertainty; generated views only)."""
    return (w * res - torch.log(w)).mean()
