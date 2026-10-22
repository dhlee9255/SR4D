"""Render a trained 4DGaussians model with ITS OWN renderer (gaussian_renderer.render, fine stage) at the SR4D evaluation
cameras, white background, time = rel / T (PanopticSports grid). Used for presets whose per-frame PLY export does not
reproduce the native render (D-038: dynerf preset with SH/opacity deformation looked purple through the PLY path).
Run with cwd = third_party/4DGaussians in the hust4dgs env:
  python fourdgs_render_native.py -m <model> --configs <preset> --root <data> --setting <s> --out <renders dir>
"""
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel  # noqa: E402
from scene.dataset_readers import setup_camera  # noqa: E402
from utils.params_utils import merge_hparams  # noqa: E402

if __name__ == "__main__":
    p = ArgumentParser()
    mp, pp, hp = ModelParams(p, sentinel=True), PipelineParams(p), ModelHiddenParams(p)
    p.add_argument("--configs", default=""); p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--root", required=True); p.add_argument("--setting", required=True); p.add_argument("--out", required=True)
    p.add_argument("--orbit", default=None, help="orbit dir (tools/make_orbit.py / make_camlist.py) -> <out>/<i:04d>.png")
    a = get_combined_args(p)
    if a.configs:
        g_ = {}; exec(compile(open(a.configs).read(), a.configs, "exec"), g_)
        a = merge_hparams(a, {k: v for k, v in g_.items() if k.endswith("Params")})
    root = Path(a.root)
    split = json.load(open(root / f"splits/{a.setting}.json")); cams = json.load(open(root / "cameras.json"))
    frames = split["frames"]; T = len(frames)
    g = GaussianModel(mp.extract(a).sh_degree, hp.extract(a))
    md = Path(a.model_path) / "point_cloud"
    it = a.iteration if a.iteration > 0 else max(int(d.name.split("_")[1]) for d in md.iterdir())
    g.load_ply(str(md / f"iteration_{it}" / "point_cloud.ply"))
    g.load_model(str(md / f"iteration_{it}"))
    pipe = pp.extract(a)
    white = torch.ones(3, device="cuda")
    def cam_of(c):
        K = np.array(c["K"], dtype=np.float64)
        w2c = np.eye(4); w2c[:3, :3] = np.array(c["R"]); w2c[:3, 3] = np.array(c["t"]).reshape(3)
        return setup_camera(c["W"], c["H"], K, w2c)._replace(bg=white)

    def shot(cam, rel):
        with torch.no_grad():
            out = render({"camera": cam, "time": rel / T}, g, pipe, white, stage="fine", cam_type="PanopticSports")
        return (out["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)[..., ::-1]

    if getattr(a, "orbit", None):
        sched = json.load(open(Path(a.orbit) / "orbit.json")); oc = json.load(open(Path(a.orbit) / "cameras.json"))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        for i, (cid, rel) in enumerate(zip(sched["cams"], sched["frame_rel"])):
            cv2.imwrite(str(Path(a.out) / f"{i:04d}.png"), shot(cam_of(oc[cid]), rel))
        print("orbit ->", a.out)
        sys.exit(0)
    for cid in split["test_interp_cams"] + split["test_extrap_cams"]:
        c = cams[cid]
        K = np.array(c["K"], dtype=np.float64)
        R, t = np.array(c["R"]), np.array(c["t"]).reshape(3)
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
        cam = setup_camera(c["W"], c["H"], K, w2c)._replace(bg=white)
        d = Path(a.out) / cid; d.mkdir(parents=True, exist_ok=True)
        for rel, f in enumerate(frames):
            with torch.no_grad():
                out = render({"camera": cam, "time": rel / T}, g, pipe, white, stage="fine", cam_type="PanopticSports")
            img = (out["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(str(d / f"{f:06d}.png"), img[..., ::-1])
    print("rendered", a.out)
