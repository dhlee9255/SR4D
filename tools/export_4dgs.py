"""SR4D unified -> HUST 4DGaussians 'PanopticSports' layout (per-camera K / w2c, no COLMAP).

<out>/data/train_meta.json : all frames x TRAINING cams   {w,h,fn,k,w2c,cam_id}; time = index / T
<out>/data/test_meta.json  : all frames x ONE training cam (only used so per-frame export index == frame index)
<out>/data/ims/<cam>/<rel>.png composited on WHITE; Panoptic cameras hard-code a black background, so training
runs through scripts/baselines/fourdgs_train_whitebg.py (white background, D-011).
<out>/data/init_pt_cld.npz : frame-0 visual hull (train masks) + SMPL(-X) vertices, rgb in [0,1].
K is written in the OpenCV pixel-centre convention, as 4DGaussians' setup_camera expects (cx used as given).

  python tools/export_4dgs.py --root data_dev/zju_mocap/387 --setting real12 --out outputs/4dgs/387/real12
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from export_common import check_centered, composite, io
from export_dualgs import init_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    split = io.load_split(root, a.setting)
    cams = io.load_cameras(root)
    frames = split["frames"]
    train = sorted(split["train_cams"])
    check_centered(cams, train)
    data = out / "data"

    def job(arg):
        cid, rel, f = arg
        img = io.read_image(root / "images" / cid / f"{f:06d}.jpg")
        m = io.read_mask(root / "masks" / cid / f"{f:06d}.png")
        p = data / "ims" / cid / f"{rel}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), cv2.cvtColor(composite(img, m, 255), cv2.COLOR_RGB2BGR))

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(job, [(c, r, f) for c in train for r, f in enumerate(frames)]))
    W, H = cams[train[0]]["W"], cams[train[0]]["H"]
    assert all(cams[c]["W"] == W and cams[c]["H"] == H for c in train)

    def meta(cids):
        return {"w": W, "h": H,
                "fn": [[f"{c}/{r}.png" for c in cids] for r in range(len(frames))],
                "k": [[cams[c]["K"].tolist() for c in cids] for _ in frames],
                "w2c": [[io.w2c_matrix(cams[c]).tolist() for c in cids] for _ in frames],
                "cam_id": [[int(c) if c.isdigit() else i for i, c in enumerate(cids)] for _ in frames]}

    (data / "train_meta.json").write_text(json.dumps(meta(train)))
    (data / "test_meta.json").write_text(json.dumps(meta(train[:1])))
    pts = init_points(root, a.setting, frames[0])
    rgb = np.full_like(pts, 0.5)
    np.savez(data / "init_pt_cld.npz", data=np.concatenate([pts, rgb], 1).astype(np.float32))
    (out / "export_meta.json").write_text(json.dumps({"frames": frames, "train": train, "n_init": len(pts)}, indent=1))
    print(f"4DGS export: {len(train)} train cams x {len(frames)} frames, init {len(pts)} pts -> {out}")


if __name__ == "__main__":
    main()
