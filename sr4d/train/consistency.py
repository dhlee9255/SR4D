"""D-036: multi-view consistency confidence for generated views (model.l4.conf_mode = consistency).

For every training view v and frame t: render the current model's depth (quarter resolution), back-project the GT
foreground pixels, re-project them into the n nearest training views (by camera-centre angle), keep the projections
that are visible there (|z - depth_n| < 2 cm + 2 %), and compare GT colours. Agreement per pixel =
mean_n exp(-||c_v - c_n||^2 / (2 sigma^2)); pixels with no visible neighbour stay neutral (1). The map is normalised to
mean 1 over the foreground, floored at `floor`, background = 1. Maps are fixed between refreshes (no residual updates).
Returns {(cid, t): (h, w) cpu tensor}.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sr4d.model.renderer import render


@torch.no_grad()
def nearest_views(data, ids, n):
    C = torch.stack([data.cameras[c].center for c in ids])
    d = C - C.mean(0)
    d = d / d.norm(dim=1, keepdim=True)
    sim = d @ d.T
    sim.fill_diagonal_(-2)
    return {ids[i]: [ids[j] for j in sim[i].topk(n).indices.tolist()] for i in range(len(ids))}


@torch.no_grad()
def compute(model, data, div: int = 4, n_nb: int = 4, sigma: float = 0.1, floor: float = 0.1):
    ids = list(data.train_ids)
    nbs = nearest_views(data, ids, n_nb)
    prev = (model.use_l2, model.use_l3, model.use_l4)
    model.use_l4 = False
    out = {}
    for t in range(data.num_frames):
        g = model.deform(t, replay=model.use_l3, need_aux=False)
        D, G, M, cams = {}, {}, {}, {}
        for c in ids:
            cam = data.cameras[c]
            W, H = cam.W // div, cam.H // div
            K = cam.K.clone(); K[:2] /= div
            r = render(g, K, cam.w2c, W, H, model.sh_degree, model.filter3d, with_normal=False)
            D[c] = torch.where(r["alpha"] > 0.5, r["depth"], torch.zeros_like(r["depth"]))  # gsplat "ED" is already depth/alpha
            gt, mask, raw = data.get(c, t)
            G[c] = F.interpolate(raw.permute(2, 0, 1)[None], size=(H, W), mode="area")[0]       # (3,h,w)
            M[c] = F.interpolate(mask[None, None], size=(H, W), mode="area")[0, 0] > 0.5
            cams[c] = (K, cam.w2c, W, H)
        for c in ids:
            K, w2c, W, H = cams[c]
            ys, xs = torch.nonzero(M[c] & (D[c] > 0), as_tuple=True)
            w_map = torch.ones(H, W, device=D[c].device)
            if len(ys) == 0:
                out[(c, t)] = w_map.cpu(); continue
            z = D[c][ys, xs]
            pc = torch.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, (ys + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
            c2w = torch.linalg.inv(w2c)
            X = pc @ c2w[:3, :3].T + c2w[:3, 3]
            col = G[c][:, ys, xs].T
            acc = torch.zeros(len(ys), device=X.device); cnt = torch.zeros_like(acc)
            for nb in nbs[c]:
                Kn, w2cn, Wn, Hn = cams[nb]
                q = X @ w2cn[:3, :3].T + w2cn[:3, 3]
                u = q[:, 0] / q[:, 2].clamp_min(1e-6) * Kn[0, 0] + Kn[0, 2] - 0.5
                v = q[:, 1] / q[:, 2].clamp_min(1e-6) * Kn[1, 1] + Kn[1, 2] - 0.5
                ok = (q[:, 2] > 1e-3) & (u >= 0) & (u <= Wn - 1) & (v >= 0) & (v <= Hn - 1)
                ui, vi = u.round().long().clamp(0, Wn - 1), v.round().long().clamp(0, Hn - 1)
                dn = D[nb][vi, ui]
                ok &= (dn > 0) & ((q[:, 2] - dn).abs() < 0.02 + 0.02 * dn) & M[nb][vi, ui]
                grid = torch.stack([u / (Wn - 1) * 2 - 1, v / (Hn - 1) * 2 - 1], -1)[None, :, None]
                cn = F.grid_sample(G[nb][None], grid, mode="bilinear", align_corners=True)[0, :, :, 0].T
                agree = torch.exp(-((col - cn) ** 2).sum(-1) / (2 * sigma ** 2))
                acc += torch.where(ok, agree, torch.zeros_like(agree)); cnt += ok.float()
            a = torch.where(cnt > 0, acc / cnt.clamp_min(1), torch.ones_like(acc))
            a = a / a.mean().clamp_min(1e-6)
            w_map[ys, xs] = a.clamp_min(floor)
            out[(c, t)] = w_map.cpu()
    model.use_l2, model.use_l3, model.use_l4 = prev
    return out


def weighted_photometric(pred, gt, lam, w, ssim_map_fn):
    """(1-lam) w*L1 + lam w*(1-SSIM map); w (H,W)."""
    l1 = ((pred - gt).abs().mean(-1) * w).mean()
    smap = ssim_map_fn(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])       # (h', w') valid conv
    ph, pw = (w.shape[0] - smap.shape[0]) // 2, (w.shape[1] - smap.shape[1]) // 2
    ws = w[ph:ph + smap.shape[0], pw:pw + smap.shape[1]]
    return (1 - lam) * l1 + lam * ((1 - smap) * ws).mean()


def ssim_map(X, Y, win=11, sigma=1.5):
    g = torch.arange(win, device=X.device, dtype=X.dtype) - win // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].repeat(X.shape[1], 1, 1, 1)
    f = lambda t: F.conv2d(t, k, groups=X.shape[1])  # noqa: E731
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx, my = f(X), f(Y)
    sxx, syy, sxy = f(X * X) - mx ** 2, f(Y * Y) - my ** 2, f(X * Y) - mx * my
    return (((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))).mean(1)[0]


def weighted_lpips(fn_spatial, pred, gt, w, bbox):
    y0, y1, x0, x1 = bbox
    a = pred[y0:y1, x0:x1].permute(2, 0, 1)[None] * 2 - 1
    b = gt[y0:y1, x0:x1].permute(2, 0, 1)[None] * 2 - 1
    m = fn_spatial(a, b)[0, 0]                                                          # (h, w) spatial map
    ww = w[y0:y1, x0:x1]
    if m.shape != ww.shape:
        ww = F.interpolate(ww[None, None], size=m.shape, mode="bilinear", align_corners=False)[0, 0]
    return (m * ww).sum() / ww.sum().clamp_min(1e-6)
