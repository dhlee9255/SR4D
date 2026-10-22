"""Crown-of-head hair close-up (EXP-005): rows = methods, columns = frames, one camera. The crop is a square centred
above the SMPL-X head joint (head + 0.08 m along neck->head), side = 3x the projected neck-head length; each crop shows
its fg PSNR inside the crop (GT mask). usage:
  PYTHONPATH=. python tools/crown_crop.py --root <data> --cam g28 --frames 10 30 50 --out x.jpg --methods "SR4D=<dir>" ...
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors
from tools.cmp_figure import label_bar, put

S = 220


def box(R, cam, bm, f):
    b, p, t, _ = params_to_tensors([io.load_body(R, f)])
    with torch.no_grad():
        J = bm(b, p, t)[1][0].numpy()
    Rm, tt, K = np.array(cam["R"]), np.array(cam["t"]).reshape(3), np.array(cam["K"])
    uv = lambda x: (lambda y: y[:2] / y[2])(K @ (Rm @ x + tt))  # noqa: E731
    up = J[15] - J[12]; up /= np.linalg.norm(up)
    c = uv(J[15] + 0.08 * up)
    side = int(max(3.0 * np.linalg.norm(uv(J[15]) - uv(J[12])), 48))
    return int(c[0]), int(c[1]), side


def crop(img, cx, cy, s):
    p = s
    im = np.pad(img, ((p, p), (p, p), (0, 0)), constant_values=255) if img.ndim == 3 else np.pad(img, p)
    return im[cy - s // 2 + p: cy + s // 2 + p, cx - s // 2 + p: cx + s // 2 + p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--cam", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[10, 30, 50]); ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", required=True); ap.add_argument("--title", default="")
    a = ap.parse_args()
    R = Path(a.root); cams = io.load_cameras(R); meta = io.load_meta(R)
    bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
    rows = [("생성 GT", None)] + [tuple(m.split("=", 1)) for m in a.methods]
    grid = []
    for name, d in rows:
        cells = []
        for f in a.frames:
            cx, cy, s = box(R, cams[a.cam], bm, f)
            gt = io.read_image(R / f"images/{a.cam}/{f:06d}.jpg"); m = io.read_mask(R / f"masks/{a.cam}/{f:06d}.png") > 127
            gtw = np.where(m[..., None], gt, 255).astype(np.uint8)
            if d is None:
                im, txt = gtw, ""
            else:
                p = Path(d) / a.cam / f"{f:06d}.png"
                im = io.read_image(p) if p.exists() else np.full_like(gtw, 160)
                mc = crop(m.astype(np.uint8), cx, cy, s) > 0
                e = ((crop(im, cx, cy, s).astype(float) - crop(gtw, cx, cy, s).astype(float)) / 255) ** 2
                txt = f"{-10 * np.log10(e[mc].mean() + 1e-10):.2f} dB" if mc.any() else ""
            c = cv2.resize(crop(im, cx, cy, s), (S, S), interpolation=cv2.INTER_AREA if s > S else cv2.INTER_CUBIC)
            cells.append(put(c, txt, 15) if txt else c)
        row = np.concatenate([put(np.full((S, 150, 3), 255, np.uint8), name, 16)] + cells, 1)
        grid.append(np.pad(row, ((0, 4), (0, 0), (0, 0)), constant_values=255))
    head = label_bar([""] + [f"frame {f}" for f in a.frames], [150] + [S] * len(a.frames))
    out = np.concatenate(([put(np.full((30, head.shape[1], 3), 255, np.uint8), a.title, 18)] if a.title else []) + [head] + grid, 0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("crown", a.out)


if __name__ == "__main__":
    main()
