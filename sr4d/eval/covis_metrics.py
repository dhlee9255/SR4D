"""Co-visible masked metrics (Dycheck-style) for held-out generated views; complements sr4d.eval.metrics.

Scores only the pixels in <data_root>/covis_<setting>/<cam>/<frame>.png (from tools/covis_masks.py: foreground pixels
whose visual-hull surface point is inside the image and mask of >= 2 training cameras). Same GT composite (white bg)
and renders as the main metrics. Reports per split:
  cv_psnr   PSNR over co-visible pixels
  cv_ipsnr  PSNR over co-visible pixels eroded by 3 px (like fgi_psnr)
  cv_ssim   SSIM map averaged over co-visible pixels (fg bbox crop)
  cv_lpips  spatial LPIPS(VGG) map averaged over co-visible pixels (fg bbox crop)
  cv_tof    flow-warped temporal error inside co-visible pixels (stride 5, as tOF)
  cv_frac   co-visible / mask pixels
Writes <renders>/../metrics_covis_<renders name>.json.
usage: python -m sr4d.eval.covis_metrics --data_root <data> --setting r15h35 --renders <run>/renders
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch

from sr4d.data import io
from sr4d.eval.metrics import ROOT, bbox, to_t
from sr4d.eval.temporal import FlowWarp


def masked_mean(x, m):
    return float((x * m).sum() / m.sum().clamp_min(1))


@torch.no_grad()
def evaluate(data_root: Path, setting: str, renders: Path, splits=("test_interp", "test_extrap"), tof_stride: int = 5):
    split = io.load_split(data_root, setting)
    frames = split["frames"]
    cdir = data_root / f"covis_{setting}"
    fn = lpips.LPIPS(net="vgg", spatial=True, verbose=False).cuda().eval()
    fw = FlowWarp()
    per = {}
    for sname in splits:
        cams = split.get(f"{sname}_cams", [])
        if not cams:
            continue
        acc = {k: [] for k in ("cv_psnr", "cv_ipsnr", "cv_ssim", "cv_lpips", "cv_tof", "cv_frac")}
        for cid in cams:
            hist = []
            for f in frames:
                rp = renders / cid / f"{f:06d}.png"
                img = io.read_image(data_root / "images" / cid / f"{f:06d}.jpg")
                m = io.read_mask(data_root / "masks" / cid / f"{f:06d}.png") > 127
                cv = io.read_mask(cdir / cid / f"{f:06d}.png") > 127
                gt = np.where(m[..., None], img, 255).astype(np.uint8)
                G, R = to_t(gt), to_t(io.read_image(rp))
                ct = torch.as_tensor(cv).cuda().float()
                se = ((G - R) ** 2).mean(-1)
                acc["cv_psnr"].append(-10 * np.log10(max(masked_mean(se, ct), 1e-10)))
                ci = torch.as_tensor(cv2.erode(cv.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0).cuda().float()
                acc["cv_ipsnr"].append(-10 * np.log10(max(masked_mean(se, ci), 1e-10)))
                acc["cv_frac"].append(cv.sum() / max(m.sum(), 1))
                y0, y1, x0, x1 = bbox(m)
                Gb, Rb = G.permute(2, 0, 1)[None, :, y0:y1, x0:x1], R.permute(2, 0, 1)[None, :, y0:y1, x0:x1]
                cb = ct[y0:y1, x0:x1]
                lp = fn(Rb * 2 - 1, Gb * 2 - 1)[0, 0]
                acc["cv_lpips"].append(masked_mean(lp, cb))
                if min(y1 - y0, x1 - x0) > 11:
                    _, smap = ssim_map(Rb, Gb)
                    pad = (cb.shape[0] - smap.shape[0]) // 2, (cb.shape[1] - smap.shape[1]) // 2
                    acc["cv_ssim"].append(masked_mean(smap, cb[pad[0]:pad[0] + smap.shape[0], pad[1]:pad[1] + smap.shape[1]]))
                cur = (R, G, ct > 0)
                hist.append(cur)
                if len(hist) > tof_stride:
                    h = hist.pop(0)
                    acc["cv_tof"].append(fw.tof(R, h[0], G, h[1], cur[2] | h[2]))
        per[sname] = {k: float(np.nanmean(v)) if v else float("nan") for k, v in acc.items()}
    return per


def ssim_map(X, Y, win=11, sigma=1.5):
    """Per-pixel SSIM (valid conv, gaussian window), channel-averaged; matches pytorch_msssim defaults."""
    import torch.nn.functional as F
    g = torch.arange(win, device=X.device, dtype=X.dtype) - win // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(X.shape[1], 1, 1, 1)
    f = lambda t: F.conv2d(t, w, groups=X.shape[1])
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx, my = f(X), f(Y)
    sxx, syy, sxy = f(X * X) - mx ** 2, f(Y * Y) - my ** 2, f(X * Y) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return None, s.mean(1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--renders", required=True)
    a = ap.parse_args()
    data_root = Path(a.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    renders = Path(a.renders)
    per = evaluate(data_root, a.setting, renders)
    (renders.parent / f"metrics_covis_{renders.name}.json").write_text(json.dumps(per, indent=1))
    for s, r in per.items():
        print(f"{s:12s} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
