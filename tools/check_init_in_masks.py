"""Fraction of a baseline's initial points that project inside the foreground mask of EVERY training camera
(coordinate-frame check for inits, e.g. Z-up vs Y-up mix-ups). Points are in the unified dataset world.

  python tools/check_init_in_masks.py --root data_dev/zju_mocap/387 --setting real12 --frame 0 --points <ply|npz>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402


def load_points(p: Path) -> np.ndarray:
    if p.suffix == ".npz":
        d = np.load(p)
        return (d["data"] if "data" in d else d[list(d.keys())[0]])[:, :3]
    from plyfile import PlyData
    v = PlyData.read(str(p))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--points", nargs="+", required=True)
    ap.add_argument("--dilate", type=int, default=0, help="mask dilation in px (tolerance)")
    a = ap.parse_args()
    root = Path(a.root)
    cams = io.load_cameras(root)
    train = io.load_split(root, a.setting)["train_cams"]
    import cv2
    for pp in a.points:
        X = load_points(Path(pp)).astype(np.float64)
        inside_all = np.ones(len(X), bool)
        per = []
        for cid in train:
            c = cams[cid]
            m = io.read_mask(root / "masks" / cid / f"{a.frame:06d}.png") > 127
            if a.dilate:
                m = cv2.dilate(m.astype(np.uint8), np.ones((2 * a.dilate + 1,) * 2, np.uint8)) > 0
            w2c = io.w2c_matrix(c)
            Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
            uv = Xc @ np.asarray(c["K"]).T
            u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            ok = (Xc[:, 2] > 0) & (ui >= 0) & (ui < c["W"]) & (vi >= 0) & (vi < c["H"])
            ins = np.zeros(len(X), bool)
            ins[ok] = m[vi[ok], ui[ok]]
            inside_all &= ins
            per.append(ins.mean())
        print(f"{pp}: n={len(X)}  inside ALL {len(train)} train masks: {inside_all.mean():.3f}  "
              f"per-camera min/mean {min(per):.3f}/{np.mean(per):.3f}  bbox {np.round(X.min(0), 2)}..{np.round(X.max(0), 2)}")


if __name__ == "__main__":
    main()
