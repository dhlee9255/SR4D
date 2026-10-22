"""Co-visibility masks for held-out generated views (method-independent, Dycheck-style masked metric).

For every held-out (test) view and frame, a foreground pixel is scored only if the surface point behind it was seen by
the training cameras: depth comes from a visual hull carved from the GENERATED masks of all views of that frame (no
method's geometry is used); the 3D point is co-visible if it projects inside the image AND inside the mask of at least
`--min_views` training cameras. Pixels in the mask whose ray misses the hull are not co-visible (reported as `miss`).

Carving is tolerant because generated views are not perfectly consistent: a voxel is removed if it projects inside the
image but outside the (dilated) mask in more than `--carve_tol` views; out-of-frame projections never carve.
Output: <root>/covis_<setting>/<cam>/<frame:06d>.png (0/255) and covis_<setting>/stats.json.
usage: python tools/covis_masks.py --root <data> --setting r15h35 [--voxel 0.008] [--min_views 2]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors


def cam_tensors(cams, ids, dev):
    K = torch.tensor(np.stack([np.array(cams[c]["K"]) for c in ids]), dtype=torch.float32, device=dev)
    R = torch.tensor(np.stack([np.array(cams[c]["R"]) for c in ids]), dtype=torch.float32, device=dev)
    t = torch.tensor(np.stack([np.array(cams[c]["t"]).reshape(3) for c in ids]), dtype=torch.float32, device=dev)
    return K, R, t


def project(K, R, t, X):  # X (N,3) -> uv (C,N,2), z (C,N)
    Xc = torch.einsum("cij,nj->cni", R, X) + t[:, None]
    z = Xc[..., 2]
    uv = torch.einsum("cij,cnj->cni", K, Xc)[..., :2] / z.clamp_min(1e-6)[..., None]
    return uv, z


def lookup(masks, uv, z, W, H):  # masks (C,H,W) bool -> inside-image (C,N), inside-mask (C,N)
    u, v = uv[..., 0].round().long(), uv[..., 1].round().long()
    inside = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
    idx = torch.arange(masks.shape[0], device=masks.device)[:, None].expand_as(uc)
    return inside, inside & masks[idx, vc, uc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--voxel", type=float, default=0.008)
    ap.add_argument("--pad", type=float, default=0.35, help="bbox padding around the SMPL-X mesh (skirt, hair) [m]")
    ap.add_argument("--carve_tol", type=int, default=2)
    ap.add_argument("--dilate", type=int, default=5, help="mask dilation (px) before carving")
    ap.add_argument("--min_views", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.root)
    dev = "cuda"
    split = io.load_split(root, a.setting)
    cams = io.load_cameras(root)
    train, test = split["train_cams"], split["test_extrap_cams"] + split.get("test_interp_cams", [])
    allc = sorted(cams)
    W, H = cams[allc[0]]["W"], cams[allc[0]]["H"]
    meta = io.load_meta(root)
    bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
    Ka, Ra, ta = cam_tensors(cams, allc, dev)
    Kt, Rt, tt = cam_tensors(cams, train, dev)
    Ke, Re, te = cam_tensors(cams, test, dev)
    out = root / f"covis_{a.setting}"
    ker = np.ones((2 * a.dilate + 1,) * 2, np.uint8)
    stats = {"mask_px": 0, "covis_px": 0, "miss_px": 0, "per_cam": {c: [0, 0] for c in test}}
    for f in split["frames"]:
        fn = io.frame_name(f)
        mk = {c: io.read_mask(root / f"masks/{c}/{fn}.png") > 127 for c in allc}
        Ma = torch.tensor(np.stack([cv2.dilate(mk[c].astype(np.uint8), ker) > 0 for c in allc]), device=dev)
        Mt = torch.tensor(np.stack([mk[c] for c in train]), device=dev)
        betas, pose, transl, _ = params_to_tensors([io.load_body(root, f)])
        with torch.no_grad():
            v, _, _ = bm(betas, pose, transl)
        v = v[0].numpy()
        lo, hi = v.min(0) - a.pad, v.max(0) + a.pad
        n = np.ceil((hi - lo) / a.voxel).astype(int)
        axes = [torch.linspace(float(lo[i]), float(lo[i] + (n[i] - 1) * a.voxel), int(n[i]), device=dev) for i in range(3)]
        X = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        carve = torch.zeros(len(X), dtype=torch.int32, device=dev)
        for s in range(0, len(X), 1 << 20):
            uv, z = project(Ka, Ra, ta, X[s:s + (1 << 20)])
            ins, inm = lookup(Ma, uv, z, W, H)
            carve[s:s + (1 << 20)] = (ins & ~inm).sum(0).int()
        occ = (carve <= a.carve_tol).float().reshape(*n)  # (nx,ny,nz)
        occ_g = occ.permute(2, 1, 0)[None, None]  # grid_sample wants (D=z,H=y,W=x)
        lo_t, ext = torch.tensor(lo, device=dev, dtype=torch.float32), torch.tensor((n - 1) * a.voxel, device=dev, dtype=torch.float32)
        for ci, c in enumerate(test):
            m = mk[c]
            ys, xs = np.nonzero(m)
            pix = torch.tensor(np.stack([xs, ys, np.ones_like(xs)], 1), dtype=torch.float32, device=dev)
            Kinv = torch.linalg.inv(Ke[ci])
            d_cam = pix @ Kinv.T
            d_w = d_cam @ Re[ci]  # R^T d
            d_w = d_w / d_w.norm(dim=1, keepdim=True)
            C = -Re[ci].T @ te[ci]
            # march from the near bbox face; step = half voxel
            dist = (torch.tensor(np.linalg.norm((lo + hi) / 2 - C.cpu().numpy()), device=dev))
            span = float(np.linalg.norm(hi - lo))
            ts = torch.arange(float(dist) - span / 2, float(dist) + span / 2, a.voxel / 2, device=dev)
            hit_t = torch.full((len(pix),), float("nan"), device=dev)
            for s in range(0, len(ts), 64):
                tt_ = ts[s:s + 64]
                P = C[None, None] + d_w[:, None] * tt_[None, :, None]  # (N,S,3)
                g = (P - lo_t) / ext * 2 - 1
                o = F.grid_sample(occ_g, g[None, :, :, None, :], mode="nearest", align_corners=True, padding_mode="zeros")[0, 0, :, :, 0]
                first = torch.where(o > 0.5, torch.arange(len(tt_), device=dev)[None].expand_as(o), len(tt_))
                fi = first.min(1).values
                newly = torch.isnan(hit_t) & (fi < len(tt_))
                hit_t[newly] = tt_[fi[newly].clamp_max(len(tt_) - 1)]
                if not torch.isnan(hit_t).any():
                    break
            hit = ~torch.isnan(hit_t)
            P = C[None] + d_w * torch.nan_to_num(hit_t)[:, None]
            uv, z = project(Kt, Rt, tt, P)
            _, inm = lookup(Mt, uv, z, W, H)
            cov = hit & (inm.sum(0) >= a.min_views)
            cm = np.zeros_like(m, dtype=np.uint8)
            cm[ys, xs] = cov.cpu().numpy().astype(np.uint8) * 255
            p = out / c / f"{fn}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), cm)
            stats["mask_px"] += int(len(xs)); stats["covis_px"] += int(cov.sum()); stats["miss_px"] += int((~hit).sum())
            stats["per_cam"][c][0] += int(len(xs)); stats["per_cam"][c][1] += int(cov.sum())
        print(f"frame {f}: grid {tuple(n)} occ {int(occ.sum())}  covis so far {stats['covis_px'] / max(stats['mask_px'], 1):.3f}", flush=True)
    stats["covis_frac"] = stats["covis_px"] / max(stats["mask_px"], 1)
    stats["miss_frac"] = stats["miss_px"] / max(stats["mask_px"], 1)
    stats["per_cam"] = {c: round(v[1] / max(v[0], 1), 4) for c, v in stats["per_cam"].items()}
    stats["args"] = vars(a)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print("covis fraction %.3f, ray-miss fraction %.3f" % (stats["covis_frac"], stats["miss_frac"]))


if __name__ == "__main__":
    main()
