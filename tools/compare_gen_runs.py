"""Item 4 check: does co-generating a second ring change the original ring?

Compares the original 24-view run with the 2-ring run view by view: camera match (same pose?) and per-view PSNR between
the generated videos (whole frame, all frames). If PSNR is high (~>40 dB) the 15-deg ring is unchanged and the original
training data stays valid; otherwise the new 15-deg ring must be used as training data together with its 35-deg ring.
usage: python tools/compare_gen_runs.py <orig_clip_dir> <new_clip_dir> [--json out.json]
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def cams(d: Path) -> list:
    c = json.load(open(d / "cameras.json"))
    return c["cameras"] if isinstance(c, dict) and "cameras" in c else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig")
    ap.add_argument("new")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    o, n = Path(a.orig), Path(a.new)
    co, cn = cams(o), cams(n)
    print(f"orig {len(co)} cams, new {len(cn)} cams")
    rows = []
    for i in range(len(co)):
        vo = iio.imread(o / f"videos/dense/{i:02d}.mp4").astype(np.float32) / 255
        vn = iio.imread(n / f"videos/dense/{i:02d}.mp4").astype(np.float32) / 255
        T = min(len(vo), len(vn))
        mse = ((vo[:T] - vn[:T]) ** 2).mean(axis=(1, 2, 3))
        psnr = float(np.mean(10 * np.log10(1 / np.maximum(mse, 1e-10))))
        same_cam = bool(np.allclose(co[i]["camera_to_world"], cn[i]["camera_to_world"], atol=1e-6) and np.allclose(co[i]["K"], cn[i]["K"], atol=1e-3))
        rows.append(dict(view=i, psnr=psnr, same_cam=same_cam, frames=T))
        print(f"view {i:02d}  same_cam {same_cam}  PSNR(orig vs new) {psnr:6.2f} dB")
    print("mean PSNR %.2f dB, min %.2f" % (np.mean([r["psnr"] for r in rows]), min(r["psnr"] for r in rows)))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
