"""Render per-frame Inria-format 3DGS PLYs (DualGS SkinGS / 4DGS per-frame export) with gsplat at the SR4D
evaluation cameras, white background. Run in the sr4d env.

  python scripts/baselines/render_ply.py --ply_pattern "<dir>/point_cloud_{rel}.ply" --root <unified> --setting real12 --out <dir>/renders
"""
import argparse
import sys
from pathlib import Path

import cv2
import gsplat
import numpy as np
import torch
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sr4d.data import io  # noqa: E402


def load_ply(path):
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], -1)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], -1)[:, None, :]
    rest_n = sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    rest = np.stack([v[n] for n in rest_n], -1).reshape(len(xyz), 3, -1).transpose(0, 2, 1) if rest_n else np.zeros((len(xyz), 0, 3))
    sh = np.concatenate([dc, rest], 1)
    deg = int(round(np.sqrt(sh.shape[1]))) - 1
    op = 1 / (1 + np.exp(-v["opacity"]))
    sc = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1))
    rot = np.stack([v[f"rot_{i}"] for i in range(4)], -1)
    t = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device="cuda")  # noqa: E731
    return t(xyz), t(rot), t(sc), t(op), t(sh), deg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply_pattern", required=True, help="format string with {rel} (frame index from 0) and/or {abs}")
    ap.add_argument("--root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orbit", default=None, help="orbit dir from tools/make_orbit.py -> <out>/<i:04d>.png")
    a = ap.parse_args()
    root = Path(a.root)
    split = io.load_split(root, a.setting)
    cams = io.load_cameras(root)
    test = split["test_interp_cams"] + split["test_extrap_cams"]
    bg = torch.ones(1, 3, device="cuda")
    n = 0
    if a.orbit:
        import json
        sched = json.loads((Path(a.orbit) / "orbit.json").read_text())
        oc = io.load_cameras(Path(a.orbit))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        for i, (cid, rel) in enumerate(zip(sched["cams"], sched["frame_rel"])):
            means, quats, scales, op, sh, deg = load_ply(a.ply_pattern.format(rel=rel, abs=split["frames"][rel]))
            c = oc[cid]
            K = c["K"].copy()
            K[:2, 2] += 0.5
            img, al, _ = gsplat.rasterization(means, quats, scales, op, sh, torch.tensor(io.w2c_matrix(c), dtype=torch.float32, device="cuda")[None],
                                              torch.tensor(K, dtype=torch.float32, device="cuda")[None], c["W"], c["H"], sh_degree=deg, backgrounds=bg, packed=False)
            cv2.imwrite(str(Path(a.out) / f"{i:04d}.png"), cv2.cvtColor((img[0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(Path(a.out) / f"{i:04d}.alpha.png"), (al[0, ..., 0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
        print("orbit ->", a.out)
        return
    for rel, f in enumerate(split["frames"]):
        means, quats, scales, op, sh, deg = load_ply(a.ply_pattern.format(rel=rel, abs=f))
        n = len(means)
        for cid in test:
            c = cams[cid]
            K = c["K"].copy()
            K[:2, 2] += 0.5
            img, al, _ = gsplat.rasterization(means, quats, scales, op, sh, torch.tensor(io.w2c_matrix(c), dtype=torch.float32, device="cuda")[None],
                                              torch.tensor(K, dtype=torch.float32, device="cuda")[None], c["W"], c["H"], sh_degree=deg, backgrounds=bg, packed=False)
            d = Path(a.out) / cid
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{f:06d}.png"), cv2.cvtColor((img[0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(d / f"{f:06d}.alpha.png"), (al[0, ..., 0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
    print("rendered ->", a.out, "last-frame n_gauss", n)


if __name__ == "__main__":
    main()
