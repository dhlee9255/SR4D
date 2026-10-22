"""Render a trained FreeTimeGS++ model at SR4D evaluation cameras with each camera's OWN intrinsics
(works around render.py's camera-0 FOV issue) and composite onto white with the rendered alpha.
Run with the FTGS++ venv python and FTGS++ on PYTHONPATH (imports only; external code untouched).

  python ftgspp_render.py --run <run_path> --root <unified root> --setting real12 --out <dir>/renders
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from ftgspp.models.gaussians import Gaussians

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sr4d.data import io  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--export_meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--composite", choices=["white", "none"], default="white",
                    help="white = rendered alpha onto white (protocol); none = raw rasteriser output (raw-bg models, fg-only eval)")
    ap.add_argument("--orbit", default=None, help="orbit dir from tools/make_orbit.py -> <out>/<i:04d>.png")
    a = ap.parse_args()
    root = Path(a.root)
    split = io.load_split(root, a.setting)
    cams = io.load_cameras(root)
    em = json.loads(Path(a.export_meta).read_text())
    gs = Gaussians.load(Path(a.run) / "gaussians.pt").cuda()
    if a.orbit:
        sched = json.loads((Path(a.orbit) / "orbit.json").read_text())
        oc = io.load_cameras(Path(a.orbit))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        for i, (cid, rel) in enumerate(zip(sched["cams"], sched["frame_rel"])):
            c = oc[cid]
            K = c["K"].copy()
            K[:2, 2] += 0.5
            img, alpha, _ = gs(t=rel / em["fps"], w2c=torch.tensor(io.w2c_matrix(c), dtype=torch.float32, device="cuda")[None],
                               intrinsic=torch.tensor(K, dtype=torch.float32, device="cuda")[None], shape=(c["H"], c["W"]), clamp=False)
            rgb = (img[0] + (1 - alpha[0])).clamp(0, 1) if a.composite == "white" else img[0].clamp(0, 1)
            cv2.imwrite(str(Path(a.out) / f"{i:04d}.png"), cv2.cvtColor((rgb.cpu().numpy() * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(Path(a.out) / f"{i:04d}.alpha.png"), (alpha[0].reshape(c["H"], c["W"]).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
        print("orbit ->", a.out)
        return
    for cid in split["test_interp_cams"] + split["test_extrap_cams"]:
        c = cams[cid]
        K = c["K"].copy()
        K[:2, 2] += 0.5  # gsplat corner convention (== pycolmap SIMPLE_PINHOLE cx=W/2 used in training)
        Kt = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
        w2c = torch.tensor(io.w2c_matrix(c), dtype=torch.float32, device="cuda")[None]
        d = Path(a.out) / cid
        d.mkdir(parents=True, exist_ok=True)
        for rel, f in enumerate(em["frames"]):
            img, alpha, _ = gs(t=rel / em["fps"], w2c=w2c, intrinsic=Kt, shape=(c["H"], c["W"]), clamp=False)
            rgb = (img[0] + (1 - alpha[0])).clamp(0, 1) if a.composite == "white" else img[0].clamp(0, 1)
            cv2.imwrite(str(d / f"{f:06d}.png"), cv2.cvtColor((rgb.cpu().numpy() * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(d / f"{f:06d}.alpha.png"), (alpha[0].reshape(c["H"], c["W"]).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
    print("rendered ->", a.out, "n_gauss", len(gs))


if __name__ == "__main__":
    main()
