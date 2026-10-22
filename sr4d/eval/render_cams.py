"""Render an SR4D checkpoint at the evaluation cameras (white background, absorber discarded).

  python -m sr4d.eval.render_cams --ckpt outputs/sr4d/387/real12/ckpt_final.pt --out outputs/sr4d/387/real12 [--mode replay|animate]
Writes <out>/renders[_<mode>]/<cam>/<frame:06d>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from sr4d.data.dataset import MultiViewSequence
from sr4d.model.renderer import composite, render
from sr4d.train.trainer import load_model


def save_alpha(path: Path, alpha: torch.Tensor):
    """accumulated opacity (H,W[,1]) -> 8-bit png (floater metric, sr4d.eval.metrics)"""
    a = alpha.reshape(alpha.shape[0], alpha.shape[1]).clamp(0, 1).cpu().numpy()
    cv2.imwrite(str(path), (a * 255 + 0.5).astype(np.uint8))


@torch.no_grad()
def render_model(model, data, cams, frames_idx, out_dir: Path, replay=True, l4="mean"):
    """l4: mean = absorber discarded (protocol); view = per-view colour code; view_cam = + per-view camera refinement
    (diagnostic only, training views of generated data)."""
    model.eval()
    model.use_l2 = model.l2 is not None and not getattr(model, "_no_l2", False)
    model.use_l3 = model.l3 is not None
    model.use_l4 = model.l4 is not None
    model.body_adhere = False
    for cid in cams:
        cam = data.cameras[cid]
        vi = data.cam_index[cid]
        w2c = cam.w2c
        if l4 == "view_cam" and model.l4 is not None:
            w2c = model.l4.refine_w2c(vi, w2c)
        d = out_dir / cid
        d.mkdir(parents=True, exist_ok=True)
        for t in frames_idx:
            g = model.deform(t, replay=replay, need_aux=False)
            r = render(g, cam.K, w2c, cam.W, cam.H, model.sh_degree, model.filter3d, with_normal=False)
            aff = []
            if g["l3"] is not None:
                aff.append((g["l3"]["color_A"], g["l3"]["color_b"]))
            if model.l4 is not None:
                aff.append(model.l4.affine(None if l4 == "mean" else vi))
            rgb = composite(r["rgb_p"], r["alpha"], data.background, aff).clamp(0, 1)
            img = (rgb.cpu().numpy() * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(str(d / f"{data.frames[t]:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            save_alpha(d / f"{data.frames[t]:06d}.alpha.png", r["alpha"])


@torch.no_grad()
def render_orbit(model, data, orbit_dir: Path, out_dir: Path, replay=True):
    """Novel-view orbit, absorber discarded (mean L4 code, no camera refinement)."""
    import json
    from sr4d.data import io
    from sr4d.data.dataset import Camera
    model.eval()
    model.use_l2, model.use_l3, model.use_l4 = model.l2 is not None and not getattr(model, "_no_l2", False), model.l3 is not None, model.l4 is not None
    model.body_adhere = False
    sched = json.loads((orbit_dir / "orbit.json").read_text())
    cams = io.load_cameras(orbit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (cid, t) in enumerate(zip(sched["cams"], sched["frame_rel"])):
        cam = Camera(cid, cams[cid], "cuda")
        g = model.deform(t, replay=replay, need_aux=False)
        r = render(g, cam.K, cam.w2c, cam.W, cam.H, model.sh_degree, model.filter3d, with_normal=False)
        aff = []
        if g["l3"] is not None:
            aff.append((g["l3"]["color_A"], g["l3"]["color_b"]))
        if model.l4 is not None:
            aff.append(model.l4.affine(None))
        rgb = composite(r["rgb_p"], r["alpha"], data.background, aff).clamp(0, 1)
        cv2.imwrite(str(out_dir / f"{i:04d}.png"), cv2.cvtColor((rgb.cpu().numpy() * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        save_alpha(out_dir / f"{i:04d}.alpha.png", r["alpha"])
    print(f"orbit {len(sched['cams'])} views -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="replay", choices=["replay", "animate"])
    ap.add_argument("--splits", nargs="+", default=["test_interp", "test_extrap"])
    ap.add_argument("--l4", default="mean", choices=["mean", "view", "view_cam"])
    ap.add_argument("--sub", default=None, help="output sub-directory name (default renders[_animate])")
    ap.add_argument("--orbit", default=None, help="orbit dir from tools/make_orbit.py -> <out>/<sub or orbit>/<i:04d>.png")
    ap.add_argument("--no_l2", action="store_true", help="diagnostic: also switch off L2 (pose-dependent non-rigid)")
    ap.add_argument("--no_l2dx", action="store_true", help="diagnostic: drop only L2's position offset dx (keep dq, dlog_s, dcolor)")
    ap.add_argument("--data_root", default=None, help="override the dataset root (e.g. re-registered evaluation cameras)")
    a = ap.parse_args()
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(st["cfg"])
    if a.data_root:
        cfg.data.root = str(Path(a.data_root).resolve())
    data = MultiViewSequence(cfg.data.root, cfg.data.setting, list(range(*cfg.data.frames)), load_images=False)
    model = load_model(Path(a.ckpt), data, cfg)
    model._no_l2 = a.no_l2
    if a.no_l2dx:
        model.l2_no_dx = True
    if a.orbit:
        render_orbit(model, data, Path(a.orbit), Path(a.out) / (a.sub or "orbit"), replay=(a.mode == "replay"))
        return
    cams = []
    for s in a.splits:
        cams += data.split[f"{s}_cams"]
    sub = a.sub or ("renders" if a.mode == "replay" else "renders_animate")
    render_model(model, data, cams, range(data.num_frames), Path(a.out) / sub, replay=(a.mode == "replay"), l4=a.l4)
    print(f"rendered {len(cams)} cams x {data.num_frames} frames -> {Path(a.out) / sub}")


if __name__ == "__main__":
    main()
