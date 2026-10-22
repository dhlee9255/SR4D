"""SR4D unified -> DualGS layout (docs/ENV_REPORT.md §5).

<out>/data/<frame_rel>/<view>.png   white-composited (DualGS default white_background=True)
<out>/data/colmap/sparse/0/{cameras,images,points3D}.txt  (PINHOLE, COLMAP pixel-corner convention)
<out>/data/colmap/sparse/0/points3d.ply  first-frame init = visual-hull surface (train masks) + SMPL(-X) vertices
view id = index into the sorted camera list; eval views are listed in export_meta.json (passed as --eval_views).

  python tools/export_dualgs.py --root data_dev/zju_mocap/387 --setting real12 --out outputs/dualgs/387/real12
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from export_common import check_centered, composite, io


def init_points(root: Path, setting: str, frame: int, device="cuda"):
    from sr4d.data.dataset import MultiViewSequence
    from sr4d.model.body_model import BodyModel, params_to_tensors
    from sr4d.model.build import carve_frame

    data = MultiViewSequence(root, setting, [frame], device=device)
    bm = BodyModel(data.meta["body_model"], data.meta.get("gender", "neutral")).to(device)
    b, p, t, e = params_to_tensors(data.body_params, device)
    with torch.no_grad():
        v, _, _ = bm(b, p, t, e)
    v = v[0]
    P = carve_frame(data, 0, v.min(0).values - 0.25, v.max(0).values + 0.25, 0.01)
    return torch.cat([P, v], 0).cpu().numpy()


def write_ply(path: Path, xyz: np.ndarray):
    rgb = np.full_like(xyz, 128, dtype=np.uint8)
    el = np.empty(xyz.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    el["x"], el["y"], el["z"] = xyz.T
    el["nx"] = el["ny"] = el["nz"] = 0
    el["red"], el["green"], el["blue"] = rgb.T
    PlyData([PlyElement.describe(el, "vertex")]).write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg", type=int, default=255)
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    split = io.load_split(root, a.setting)
    cams = io.load_cameras(root)
    frames = split["frames"]
    cids = sorted(set(split["train_cams"]) | set(split["test_interp_cams"]) | set(split["test_extrap_cams"]))
    check_centered(cams, cids)
    data = out / "data"

    def job(arg):
        v, cid, rel, f = arg
        img = io.read_image(root / "images" / cid / f"{f:06d}.jpg")
        m = io.read_mask(root / "masks" / cid / f"{f:06d}.png")
        p = data / str(rel) / f"{v}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), cv2.cvtColor(composite(img, m, a.bg), cv2.COLOR_RGB2BGR))

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(job, [(v, c, r, f) for v, c in enumerate(cids) for r, f in enumerate(frames)]))

    sp = data / "colmap" / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    with open(sp / "cameras.txt", "w") as fc, open(sp / "images.txt", "w") as fi:
        for v, cid in enumerate(cids):
            c = cams[cid]
            K = c["K"]
            fc.write(f"{v + 1} PINHOLE {c['W']} {c['H']} {K[0, 0]:.8f} {K[1, 1]:.8f} {K[0, 2] + 0.5:.8f} {K[1, 2] + 0.5:.8f}\n")
            q = Rotation.from_matrix(c["R"]).as_quat()  # xyzw
            t = c["t"]
            fi.write(f"{v + 1} {q[3]:.10f} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {v + 1} {v}.png\n\n")
    (sp / "points3D.txt").write_text("")
    write_ply(sp / "points3d.ply", init_points(root, a.setting, frames[0]))
    idx = {c: i for i, c in enumerate(cids)}
    # eval views are excluded from DualGS training; a test cam that is also a train cam (self-fit settings such as
    # gen48_all, D-030) must stay in training, so only held-out cams are listed
    ev = [idx[c] for c in split["test_interp_cams"] + split["test_extrap_cams"] if c not in set(split["train_cams"])]
    (out / "export_meta.json").write_text(json.dumps({"cam_index": idx, "frames": frames, "eval_views": ev}, indent=1))
    print(f"DualGS export: {len(cids)} views, eval {ev}, {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
