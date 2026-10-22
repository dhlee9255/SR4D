"""MCMC-style densification (Kheradmand et al. 2024, via gsplat's relocation kernel) adapted to the two
Gaussian sets. Each set keeps its own count; growth fills the total cap proportionally to current sizes.
Body surfels inherit the source triangle (barycentrics jittered); free Gaussians get MCMC position noise.
"""
from __future__ import annotations

import math

import torch
from gsplat.relocation import compute_relocation

from sr4d.model.canonical import GaussianSet
from sr4d.utils.geometry import quat_to_rotmat

N_MAX = 51
_BINOMS = None


def binoms(device):
    global _BINOMS
    if _BINOMS is None or _BINOMS.device != torch.device(device):
        b = torch.zeros(N_MAX, N_MAX, device=device)
        for n in range(N_MAX):
            for k in range(n + 1):
                b[n, k] = math.comb(n, k)
        _BINOMS = b
    return _BINOMS


def _full_scales(gs: GaussianSet, eps: float):
    s = gs["log_scales"].exp()
    if s.shape[1] == 2:
        s = torch.cat([s, torch.full_like(s[:, :1], eps)], -1)
    return s


def _apply_new(gs: GaussianSet, src: torch.Tensor, new_op: torch.Tensor, new_s: torch.Tensor):
    vals = gs.gather(src)
    vals["opacity"] = torch.logit(new_op)[:, None]
    vals["log_scales"] = torch.log(new_s[:, : gs["log_scales"].shape[1]])
    if "bary_logits" in vals:  # spread duplicated surfels over their triangle
        vals["bary_logits"] = vals["bary_logits"] + torch.randn_like(vals["bary_logits"]) * 0.3
    return vals


@torch.no_grad()
def relocate(gs: GaussianSet, min_opacity: float, eps_scale: float) -> int:
    op = torch.sigmoid(gs["opacity"][:, 0])
    dead = op <= min_opacity
    n = int(dead.sum())
    if n == 0 or n == len(gs):
        return 0
    alive = (~dead).nonzero()[:, 0]
    src = alive[torch.multinomial(op[alive], n, replacement=True)]
    ratios = torch.bincount(src, minlength=len(gs))[src] + 1
    new_op, new_s = compute_relocation(op[src], _full_scales(gs, eps_scale)[src], ratios, binoms(op.device))
    new_op = new_op.clamp(min_opacity, 1 - 1e-6)
    vals = _apply_new(gs, src, new_op, new_s)
    # source rows get the same reduced opacity/scale, then dead rows become copies
    gs.set_rows(src, {"opacity": vals["opacity"], "log_scales": vals["log_scales"]})
    gs.set_rows(dead.nonzero()[:, 0], vals)
    return n


@torch.no_grad()
def grow(gs: GaussianSet, n_add: int, min_opacity: float, eps_scale: float) -> int:
    if n_add <= 0:
        return 0
    op = torch.sigmoid(gs["opacity"][:, 0])
    src = torch.multinomial(op, n_add, replacement=True)
    ratios = torch.bincount(src, minlength=len(gs))[src] + 1
    new_op, new_s = compute_relocation(op[src], _full_scales(gs, eps_scale)[src], ratios, binoms(op.device))
    new_op = new_op.clamp(min_opacity, 1 - 1e-6)
    vals = _apply_new(gs, src, new_op, new_s)
    gs.set_rows(src, {"opacity": vals["opacity"], "log_scales": vals["log_scales"]})
    gs.append(vals)
    return n_add


@torch.no_grad()
def inject_noise_free(gs: GaussianSet, scaler: float):
    op = torch.sigmoid(gs["opacity"][:, 0])
    R = quat_to_rotmat(gs["quats"])
    s = gs["log_scales"].exp()
    cov = R @ torch.diag_embed(s * s) @ R.transpose(-1, -2)
    k = 1 / (1 + torch.exp(-100 * ((1 - op) - 0.995)))
    noise = torch.randn_like(gs["means"]) * k[:, None] * scaler
    gs["means"].add_(torch.einsum("bij,bj->bi", cov, noise))


@torch.no_grad()
def unseen_free_mask(model, data, replay: bool = True) -> torch.Tensor:
    """D-029 (user 2026-10-12): free Gaussians whose DEFORMED centre is outside the image of EVERY training camera in
    EVERY frame (bool, len = n_free). Body surfels (attached to SMPL-X) are never tested. One 'in frame' anywhere keeps it."""
    nb, nf = model.n_body, model.n_free
    seen = torch.zeros(nf, dtype=torch.bool, device=model.free["means"].device)
    cams = [data.cameras[c] for c in data.train_ids]
    for t in range(data.num_frames):
        x = model.deform(t, replay=replay, need_aux=False)["means"][nb:nb + nf]
        for cam in cams:
            pc = x @ cam.w2c[:3, :3].T + cam.w2c[:3, 3]
            z = pc[:, 2]
            u = pc[:, 0] / z.clamp_min(1e-6) * cam.K[0, 0] + cam.K[0, 2]
            v = pc[:, 1] / z.clamp_min(1e-6) * cam.K[1, 1] + cam.K[1, 2]
            seen |= (z > 1e-3) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
    return ~seen


@torch.no_grad()
def kill_rows(gs: GaussianSet, mask: torch.Tensor, min_opacity: float) -> int:
    """Mark rows dead (opacity just below the MCMC threshold) so the next relocate() reuses them where the scene is
    observed; the Gaussian count stays fixed (MCMC). Returns the number killed."""
    idx = mask.nonzero()[:, 0]
    if len(idx):
        o = torch.full((len(idx), 1), float(torch.logit(torch.tensor(min_opacity * 0.5))), device=gs["opacity"].device)
        gs.set_rows(idx, {"opacity": o})
    return len(idx)
