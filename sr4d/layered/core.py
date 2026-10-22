"""2-layer model (docs/DECISIONS.md D-040): frozen SR4D inner layer (cached per frame) + free FTGS++-style outer layer,
rasterised together in ONE gsplat call. A 4th feature channel (1 = inner, 0 = outer) gives the inner layer's share of the
accumulated alpha.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gsplat
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "ftgs_outer"))
from outer import TimeGaussians, cat_states, init_dynamic  # noqa: E402

from sr4d.model.canonical import sh_num  # noqa: E402
from sr4d.model.renderer import apply_filter3d  # noqa: E402

FPS = 30.0


class InnerCache:
    """Per-frame deformed Gaussians of a frozen SR4D checkpoint (replay mode, L4 discarded, L3 colour affine folded in)."""

    @torch.no_grad()
    def __init__(self, ckpt: Path, data):
        from sr4d.train.trainer import load_model
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(st["cfg"])
        m = load_model(Path(ckpt), data, cfg)
        m.eval()
        m.use_l2 = m.l2 is not None
        m.use_l3 = m.l3 is not None
        m.use_l4 = m.l4 is not None
        m.body_adhere = False
        self.sh_degree = m.sh_degree
        self.frames = []
        for t in range(data.num_frames):
            g = m.deform(t, replay=True, need_aux=False)
            cov, op = apply_filter3d(g["covars"], g["opacities"], m.filter3d)
            A, b = torch.eye(3, device=cov.device), torch.zeros(3, device=cov.device)
            if g["l3"] is not None:
                A, b = g["l3"]["color_A"], g["l3"]["color_b"]
            if m.l4 is not None:  # mean code (protocol); identity for colour-off L4
                A4, b4 = m.l4.affine(None)
                A, b = A4 @ A, A4 @ b + b4
            self.frames.append(dict(means=g["means"].contiguous(), covars=cov.contiguous(), opac=op.contiguous(),
                                    shs=g["shs"][:, : sh_num(self.sh_degree)].contiguous(), A=A, b=b))
        self.n = self.frames[0]["means"].shape[0]
        del m
        torch.cuda.empty_cache()

    def colors(self, t: int, campos):
        f = self.frames[t]
        rgb = gsplat.spherical_harmonics(self.sh_degree, f["means"] - campos, f["shs"])
        rgb = torch.clamp_min(rgb + 0.5, 0.0)
        return rgb @ f["A"].T + f["b"]


def render2(inner: InnerCache | None, outer: TimeGaussians | None, t: int, K, w2c, W, H, sh_outer: int,
            use_inner=True, use_outer=True, bg=1.0, outer_opac_mul=None, outer_probe=None):
    """Joint rasterisation. Returns rgb (H,W,3) on bg, alpha (H,W), inner alpha share (H,W), meta, n_inner.
    outer_opac_mul: optional (N_outer,) multiplier on the outer opacities (EXP-004 view-coverage fade).
    outer_probe: optional (N_outer,) all-zero tensor with requires_grad, rendered as a 5th channel (inner 0) into
    meta["probe"]; adding meta["probe"].sum() to the loss changes neither the loss nor any other gradient and gives
    outer_probe.grad = per-Gaussian contribution sum_px T*alpha (EXP-005 view counting)."""
    c2w = torch.linalg.inv(w2c)
    campos = c2w[:3, 3]
    means, covs, opac, feats = [], [], [], []
    n_in = 0
    if inner is not None and use_inner:
        f = inner.frames[t]
        means.append(f["means"]); covs.append(f["covars"]); opac.append(f["opac"])
        fi = [inner.colors(t, campos), torch.ones_like(f["opac"])[:, None]]
        if outer_probe is not None and outer is not None and use_outer:
            fi.append(torch.zeros_like(f["opac"])[:, None])
        feats.append(torch.cat(fi, -1))
        n_in = f["means"].shape[0]
    if outer is not None and use_outer:
        ts = t / FPS
        mo = outer.means_t(ts)
        rgb = gsplat.spherical_harmonics(sh_outer, mo - campos, outer.shs()[:, : sh_num(sh_outer)] if sh_outer < 3 else outer.shs())
        rgb = torch.clamp_min(rgb + 0.5, 0.0)
        oo = outer.opacities_t(ts)[:, 0]
        means.append(mo); covs.append(outer.covars()); opac.append(oo if outer_opac_mul is None else oo * outer_opac_mul)
        fo = [rgb, torch.zeros_like(rgb[:, :1])] + ([outer_probe[:, None]] if outer_probe is not None else [])
        feats.append(torch.cat(fo, -1))
    out, alpha, meta = gsplat.rasterization(
        torch.cat(means), None, None, torch.cat(opac), torch.cat(feats), w2c[None], K[None], W, H,
        covars=torch.cat(covs), packed=False, rasterize_mode="antialiased", absgrad=False)
    out, alpha = out[0], alpha[0, ..., 0]
    rgb = out[..., :3] + bg * (1 - alpha[..., None])
    if out.shape[-1] > 4:
        meta["probe"] = out[..., 4]
    return rgb, alpha, out[..., 3], meta, n_in


@torch.no_grad()
def project_colors(P: torch.Tensor, data, t: int):
    """median colour of 3D points over the training views whose mask contains them (visual-hull points)."""
    cols, valid = [], []
    for cid in data.train_ids:
        cam = data.cameras[cid]
        pc = P @ cam.w2c[:3, :3].T + cam.w2c[:3, 3]
        z = pc[:, 2].clamp_min(1e-6)
        u = (pc[:, 0] / z * cam.K[0, 0] + cam.K[0, 2]).long()
        v = (pc[:, 1] / z * cam.K[1, 1] + cam.K[1, 2]).long()
        ok = (pc[:, 2] > 0.05) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        uu, vv = u.clamp(0, cam.W - 1), v.clamp(0, cam.H - 1)
        ok &= data.masks[cid][t][vv, uu] > 127
        cols.append(data.images[cid][t][vv, uu].float() / 255)
        valid.append(ok)
    C = torch.stack(cols)  # (V,N,3)
    Vm = torch.stack(valid)
    C = torch.where(Vm[..., None], C, torch.nan)
    med = torch.nanmedian(C, 0).values
    return torch.nan_to_num(med, 0.5)


@torch.no_grad()
def init_outer(inner: InnerCache, data, n_total=500_000, stride=10, vh_voxel=0.01, inner_frac=0.5, seed=0, log=print):
    """Keyframes every `stride` frames (+ last): points = inner layer posed at that frame + visual-hull surface points;
    then FTGS++ init_dynamic between consecutive keyframes (last keyframe paired backwards)."""
    from sr4d.model.build import carve_frame
    from sr4d.model.canonical import C0
    T = data.num_frames
    keys = list(range(0, T, stride))
    if keys[-1] != T - 1:
        keys.append(T - 1)
    gen = torch.Generator().manual_seed(seed)
    n_key = n_total // len(keys)
    pts = []
    for t in keys:
        f = inner.frames[t]
        keep = f["opac"] > 0.05
        xi = f["means"][keep]
        ci = (f["shs"][keep, 0] * C0 + 0.5).clamp(0, 1) @ f["A"].T + f["b"]
        pad = 0.25
        P = carve_frame(data, t, xi.min(0).values - pad, xi.max(0).values + pad, vh_voxel)
        cp = project_colors(P, data, t)
        ni = min(int(n_key * inner_frac), xi.shape[0])
        nv = min(n_key - ni, P.shape[0])
        si = torch.randperm(xi.shape[0], generator=gen)[:ni]
        sv = torch.randperm(P.shape[0], generator=gen)[:nv]
        xyz = torch.cat([xi[si.to(xi.device)], P[sv.to(P.device)]]).cpu()
        rgb = torch.cat([ci[si.to(ci.device)], cp[sv.to(cp.device)]]).clamp(0, 1).cpu()
        log(f"keyframe {t}: inner {ni} + visual hull {nv} (of {P.shape[0]}) points")
        pts.append((t, xyz, rgb))
    parts = []
    for i, (t, xyz, rgb) in enumerate(pts):
        j = i + 1 if i + 1 < len(pts) else i - 1
        t1, xyz1, _ = pts[j]
        parts.append(init_dynamic(len(xyz), xyz, rgb, xyz1, t / FPS, (t1 - t) / FPS, gen=gen))
    return TimeGaussians(max_duration=1.0, **cat_states(parts))


def save_png(path, x):
    import cv2
    a = x.clamp(0, 1).cpu().numpy()
    if a.ndim == 3:
        cv2.imwrite(str(path), cv2.cvtColor((a * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), (a * 255 + 0.5).astype(np.uint8))
