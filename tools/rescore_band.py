"""D-032: add boundary-band / interior split metrics to existing results without re-rendering.
band = GT silhouette +-3 px, interior = mask eroded 3 px. Writes <run>/metrics_band_<renders>.json:
{split: {band_psnr, band_mae8, interior_psnr, interior_mae8, mask_iou}} (mask_iou from render alpha > 0.5 if present).
usage: python tools/rescore_band.py --data_root <data> --setting <s> --renders <run>/renders [--frames_every 1]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sr4d.data import io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True); ap.add_argument("--setting", required=True); ap.add_argument("--renders", required=True)
    ap.add_argument("--frames_every", type=int, default=1)
    a = ap.parse_args()
    root, rd = Path(a.data_root), Path(a.renders)
    sp = io.load_split(root, a.setting); k = np.ones((7, 7), np.uint8)
    out = {}
    for sname in ("test_interp", "test_extrap"):
        cams = sp.get(f"{sname}_cams", [])
        if not cams:
            continue
        acc = {"bp": [], "ip": [], "ba": [], "ia": [], "iou": []}  # per-image values, averaged (same as metrics.py)
        for c in cams:
            for f in sp["frames"][::a.frames_every]:
                img = io.read_image(root / f"images/{c}/{f:06d}.jpg").astype(np.float32) / 255
                m = io.read_mask(root / f"masks/{c}/{f:06d}.png") > 127
                gt = np.where(m[..., None], img, 1.0)
                r = io.read_image(rd / c / f"{f:06d}.png").astype(np.float32) / 255
                e2 = ((r - gt) ** 2).mean(-1); e1 = np.abs(r - gt).mean(-1)
                band = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~(cv2.erode(m.astype(np.uint8), k) > 0)
                inter = cv2.erode(m.astype(np.uint8), k) > 0
                acc["bp"].append(-10 * np.log10(max(e2[band].mean(), 1e-10))); acc["ba"].append(e1[band].mean() * 255)
                acc["ip"].append(-10 * np.log10(max(e2[inter].mean(), 1e-10))); acc["ia"].append(e1[inter].mean() * 255)
                ap_ = rd / c / f"{f:06d}.alpha.png"
                if ap_.exists():
                    al = cv2.imread(str(ap_), 0) > 127
                    acc["iou"].append((al & m).sum() / max((al | m).sum(), 1))
        out[sname] = dict(band_psnr=float(np.mean(acc["bp"])), band_mae8=float(np.mean(acc["ba"])),
                          interior_psnr=float(np.mean(acc["ip"])), interior_mae8=float(np.mean(acc["ia"])),
                          mask_iou=float(np.mean(acc["iou"])) if acc["iou"] else None)
    (rd.parent / f"metrics_band_{rd.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
