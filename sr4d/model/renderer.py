"""gsplat-based renderer. Body surfels and free Gaussians are rendered together as 3DGS in ONE pass
(gsplat's 2DGS rasterizer cannot be composited with 3DGS in the same pass; surfels are flattened instead:
docs/DECISIONS.md D-004). Outputs premultiplied RGB (black background), alpha, expected depth and
alpha-composited world normals. Colour affines (L3 per-frame, L4 per-view) are applied to the premultiplied
foreground before compositing onto the background, so the background colour is never altered.
"""
from __future__ import annotations

import gsplat
import torch
from gsplat.utils import depth_to_normal

from sr4d.model.canonical import sh_num


def apply_filter3d(covars, opac, filt):
    if filt is None or filt.numel() != covars.shape[0]:
        return covars, opac
    eye = torch.eye(3, device=covars.device)
    cf = covars + (filt ** 2)[:, None, None] * eye
    det0 = torch.linalg.det(covars).clamp_min(1e-30)
    det1 = torch.linalg.det(cf).clamp_min(1e-30)
    return cf, opac * torch.sqrt(det0 / det1)


def render(g: dict, K: torch.Tensor, w2c: torch.Tensor, W: int, H: int, sh_degree: int, filter3d=None, with_normal: bool = True):
    means, covars, opac, shs = g["means"], g["covars"], g["opacities"], g["shs"]
    c2w = torch.linalg.inv(w2c)
    campos = c2w[:3, 3]
    dirs = means - campos
    rgb = gsplat.spherical_harmonics(sh_degree, dirs, shs[:, : sh_num(sh_degree)])
    rgb = torch.clamp_min(rgb + 0.5, 0.0)
    covars, opac = apply_filter3d(covars, opac, filter3d)
    feats = [rgb]
    if with_normal:
        n = g["normals"]
        n = torch.where(((n * dirs).sum(-1, keepdim=True) > 0), -n, n)  # face the camera
        feats.append(n)
    feats = torch.cat(feats, -1)
    out, alpha, meta = gsplat.rasterization(
        means, None, None, opac, feats, w2c[None], K[None], W, H,
        covars=covars, packed=False, render_mode="RGB+ED", rasterize_mode="antialiased", absgrad=False,
    )
    out = out[0]
    res = {"rgb_p": out[..., :3], "alpha": alpha[0, ..., 0], "depth": out[..., -1], "meta": meta}
    if with_normal:
        res["normal"] = out[..., 3:6]
        dn = depth_to_normal(out[None, ..., -1:], c2w[None], K[None])[0]
        res["depth_normal"] = dn
    return res


def composite(rgb_p, alpha, bg, affines: list | None = None):
    x = rgb_p
    for A, b in affines or []:
        x = x @ A.T + alpha[..., None] * b
    return x + bg * (1 - alpha[..., None])


@torch.no_grad()
def compute_filter3d(means: torch.Tensor, cams: list, kernel: float = 0.2) -> torch.Tensor:
    """Mip-Splatting 3D smoothing filter std per Gaussian from the training cameras' max sampling rate."""
    rate = torch.zeros(means.shape[0], device=means.device)
    for cam in cams:
        pc = means @ cam.w2c[:3, :3].T + cam.w2c[:3, 3]
        z = pc[:, 2]
        uv = pc[:, :2] / z.clamp_min(1e-6)[:, None] * torch.stack([cam.K[0, 0], cam.K[1, 1]]) + cam.K[:2, 2]
        inside = (z > 0.05) & (uv[:, 0] >= -0.15 * cam.W) & (uv[:, 0] <= 1.15 * cam.W) & (uv[:, 1] >= -0.15 * cam.H) & (uv[:, 1] <= 1.15 * cam.H)
        r = torch.where(inside, cam.K[0, 0] / z.clamp_min(1e-6), torch.zeros_like(z))
        rate = torch.maximum(rate, r)
    rate = torch.where(rate > 0, rate, rate[rate > 0].min() if (rate > 0).any() else torch.ones_like(rate))
    return (kernel ** 0.5) / rate

