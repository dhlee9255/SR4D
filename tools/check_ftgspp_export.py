"""Diagnostic (FTGS++ venv): compare our export_ftgspp poses_bounds with legacy build_ftgspp_dynerf.py on the same
cameras, round-trip both through ftgspp.data.sfm.parse_poses_bounds_npy, and measure the per-camera pass rate of
ftgspp.init.points' cheirality test (w2c[:3,2]·(X − w2c[:3,3])) vs the true depth test on body vertices.

  PYTHONPATH=<ftgspp>:<sr4d> <ftgspp venv python> tools/check_ftgspp_export.py --root data_dev/zju_mocap/387 --cams train:real12
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))
from sr4d.data import io  # noqa: E402
from build_ftgspp_dynerf import build_poses_bounds  # noqa: E402
from ftgspp.data.sfm import parse_poses_bounds_npy  # noqa: E402


def ours(cams, cids):
    rows = []
    for cid in cids:
        c = cams[cid]
        c2w = np.linalg.inv(io.w2c_matrix(c))
        pose = np.stack([c2w[:3, 1], c2w[:3, 0], -c2w[:3, 2], c2w[:3, 3]], 1)
        hwf = np.array([[c["H"]], [c["W"]], [c["K"][0, 0]]])
        rows.append(np.concatenate([np.concatenate([pose, hwf], 1).reshape(-1), [0.1, np.linalg.norm(c2w[:3, 3]) * 2]]))
    return np.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cams", default="all", help="'all', 'train:<setting>' or 'prefix:<p>'")
    a = ap.parse_args()
    root = Path(a.root)
    cams = io.load_cameras(root)
    if a.cams == "all":
        cids = sorted(cams)
    elif a.cams.startswith("train:"):
        cids = sorted(io.load_split(root, a.cams[6:])["train_cams"])
    else:
        cids = sorted(k for k in cams if k.startswith(a.cams[7:]))
    c0 = cams[cids[0]]
    manifest = dict(image_width=c0["W"], image_height=c0["H"], cameras={str(i): dict(K=cams[c]["K"].tolist(), world_to_camera=io.w2c_matrix(cams[c]).tolist()) for i, c in enumerate(cids)})
    pb_legacy = build_poses_bounds(manifest, list(range(len(cids))))
    pb_ours = ours(cams, cids)
    d = np.abs(pb_legacy - pb_ours)
    print(f"poses_bounds ours vs legacy: max |diff| pose cols {d[:, :15].max():.2e}; near ours {pb_ours[0, 15]} legacy {pb_legacy[0, 15]}; far max diff {d[:, 16].max():.2e}")
    names = [f"{i}/x.png" for i in range(len(cids))]
    res = {}
    for tag, pb in (("ours", pb_ours), ("legacy", pb_legacy)):
        with tempfile.TemporaryDirectory() as td:
            np.save(Path(td) / "pb.npy", pb)
            rec = parse_poses_bounds_npy(Path(td) / "pb.npy", names)
        errs = []
        for i, c in enumerate(cids):
            im = rec.image(i + 1)
            w2c = np.eye(4)
            w2c[:3, :3] = im.cam_from_world().rotation.matrix()
            w2c[:3, 3] = im.cam_from_world().translation
            errs.append(np.abs(w2c - io.w2c_matrix(cams[c])).max())
            res.setdefault(c, {})[tag] = w2c
        print(f"[{tag}] round-trip through ftgspp parse_poses_bounds_npy: max |w2c diff| {max(errs):.2e}")

    # cheirality on body points (frame 0 body vertices approximated by the SMPL transl region if no verts available)
    meta = io.load_meta(root)
    fr = meta.get("frames", [0])[0] if isinstance(meta.get("frames"), list) else 0
    b = io.load_body(root, fr)
    centre = np.asarray(b["transl"], dtype=np.float64)
    rng = np.random.default_rng(0)
    X = centre + rng.normal(scale=[0.25, 0.25, 0.25], size=(5000, 3))
    print(f"cheirality on 5000 points around the body centre {np.round(centre, 3)}:")
    rows = []
    for c in cids:
        w2c = res[c]["ours"]
        bug = (w2c[:3, 2] @ (X - w2c[:3, 3]).T) > 0
        true = (X @ w2c[2, :3] + w2c[2, 3]) > 0
        C = -w2c[:3, :3].T @ w2c[:3, 3]
        az = np.degrees(np.arctan2(*(C - centre)[[1, 0]]))
        rows.append((c, round(float(az), 1), float(true.mean()), float(bug.mean())))
        print(f"  cam {c:>4s} az {az:7.1f}  true-depth pass {true.mean():.2f}   ftgspp-test pass {bug.mean():.2f}")
    nb = sum(r[3] < 0.05 for r in rows)
    print(f"cameras with <5% ftgspp-test pass: {nb}/{len(rows)}")


if __name__ == "__main__":
    main()
