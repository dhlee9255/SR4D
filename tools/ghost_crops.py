"""Ghosting diagnosis: generated GT vs render on TRAINING views, same frame, native-resolution crops enlarged with
nearest-neighbour (no interpolation) around the face, both hands and the skirt / hem edge (between hip and knee).

Crop centres come from the input SMPL-X joints projected into each view. One PNG per (view, frame):
rows = GT, then each render dir; cols = face, L hand, R hand, L hem, R hem, plus a full-frame thumbnail with boxes.
usage: python tools/ghost_crops.py --root <data> --cams g00 g06 --frames 20 40 --render V0=<dir> 4DGS=<dir> --out <dir>
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors

REGIONS = [("face", (15, 15, 0.0), 88), ("L hand", (20, 20, 0.0), 44), ("R hand", (21, 21, 0.0), 44),
           ("L hem", (1, 4, 0.55), 88), ("R hem", (2, 5, 0.55), 88)]  # (joint a, joint b, t): crop at a + t(b-a); sizes divide CELL
CELL = 176


def project(cam, X):
    R, t, K = np.array(cam["R"]), np.array(cam["t"]).reshape(3), np.array(cam["K"])
    x = (K @ (R @ X.T + t[:, None])).T
    return x[:, :2] / x[:, 2:3]


def pt(uv, spec):
    a, b, t = spec
    return uv[a] + t * (uv[b] - uv[a])


def crop(img, c, s):
    H, W = img.shape[:2]
    x0, y0 = int(round(c[0] - s / 2)), int(round(c[1] - s / 2))
    out = np.full((s, s, 3), 128, np.uint8)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    k = max(1, CELL // s)
    return np.repeat(np.repeat(out, k, 0), k, 1)  # nearest neighbour


def label(img, txt):
    img = img.copy()
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cams", nargs="+", required=True)
    ap.add_argument("--frames", nargs="+", type=int, required=True)
    ap.add_argument("--render", nargs="*", default=[], help="LABEL=<renders dir with <cam>/<frame:06d>.png>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = io.load_cameras(root)
    meta = io.load_meta(root)
    bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
    rnd = [r.split("=", 1) for r in a.render]
    for f in a.frames:
        betas, pose, transl, expr = params_to_tensors([io.load_body(root, f)])
        with torch.no_grad():
            _, joints, _ = bm(betas, pose, transl)
        Xj = joints[0].numpy()
        for c in a.cams:
            gt = cv2.cvtColor(cv2.imread(str(root / f"images/{c}/{io.frame_name(f)}.jpg")), cv2.COLOR_BGR2RGB)
            m = cv2.imread(str(root / f"masks/{c}/{io.frame_name(f)}.png"), 0)
            gt = np.where(m[..., None] > 127, gt, 255).astype(np.uint8)  # white-bg composite like the training target
            uv = project(cams[c], Xj)
            imgs = [("GT", gt)]
            for lab, d in rnd:
                p = Path(d) / c / f"{io.frame_name(f)}.png"
                if p.exists():
                    imgs.append((lab, cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)))
            rows = []
            for lab, im in imgs:
                cells = [label(crop(im, pt(uv, j), s), f"{lab} {name}") for name, j, s in REGIONS]
                th = im.copy()
                if lab == "GT":
                    for name, j, s in REGIONS:
                        x, y = pt(uv, j)
                        cv2.rectangle(th, (int(x - s / 2), int(y - s / 2)), (int(x + s / 2), int(y + s / 2)), (255, 0, 0), 1)
                h = CELL
                th = cv2.resize(th, (int(th.shape[1] * h / th.shape[0]), h), interpolation=cv2.INTER_AREA)
                rows.append(np.concatenate(cells + [th], 1))
            W = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
            cv2.imwrite(str(out / f"{c}_{f:03d}.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
            print("wrote", out / f"{c}_{f:03d}.png")


if __name__ == "__main__":
    main()
