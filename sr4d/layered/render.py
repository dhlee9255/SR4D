"""Render the 2-layer model (docs/DECISIONS.md D-040).

  python -m sr4d.layered.render --run <train out dir> [--ckpt outer_final.pt] --mode both|outer|inner --out <dir>
      [--orbit <orbit dir>] [--cams g03 g15 ... --frames 0 30 59] [--fade TH1 TH2 [--vis vis_outer_final.pt --vis_min 1]]
--fade: EXP-004 New-fade (D-041), outer opacity x view-coverage fade (sr4d.layered.fade).
Default: every held-out camera x every frame -> <out>/<cam>/<frame:06d>.png, .alpha.png (accumulated alpha) and
.inner.png (inner layer's share of the accumulated alpha, = inner alpha / total alpha; only for --mode both).
Orbit: <out>/<i:04d>.png (+ .alpha / .inner).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from sr4d.data import io
from sr4d.data.dataset import Camera, MultiViewSequence
from sr4d.layered.core import InnerCache, render2, save_png

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "ftgs_outer"))
from outer import TimeGaussians  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="outer_final.pt")
    ap.add_argument("--mode", default="both", choices=["both", "outer", "inner"]); ap.add_argument("--out", required=True)
    ap.add_argument("--orbit", default=None); ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--frames", nargs="*", type=int, default=None)
    ap.add_argument("--fade", nargs=2, type=float, default=None, metavar=("TH1", "TH2"))
    ap.add_argument("--vis", default="vis_outer_final.pt"); ap.add_argument("--vis_min", type=float, default=1.0)
    a = ap.parse_args()
    run = Path(a.run); args = json.loads((run / "args.json").read_text())
    data = MultiViewSequence(args["root"], args["setting"], load_images=False)
    inner = InnerCache(Path(args["inner"]), data) if a.mode in ("both", "inner") else None
    outer = None
    if a.mode in ("both", "outer"):
        outer = TimeGaussians.from_state(torch.load(run / a.ckpt, map_location="cpu", weights_only=False)["outer"]).cuda()
    sh = outer.sh_degree if outer is not None else 0
    vis = None
    if a.fade is not None:
        from sr4d.layered.fade import fade_factor
        vis = {k: v.cuda() for k, v in torch.load(run / a.vis, map_location="cpu").items() if torch.is_tensor(v)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def shot(cam, t, path_stem):
        mul = None if vis is None else fade_factor(outer, vis, t, cam.center, *a.fade, vis_min=a.vis_min)
        rgb, alpha, ain, _, _ = render2(inner, outer, t, cam.K, cam.w2c, cam.W, cam.H, sh, outer_opac_mul=mul)
        save_png(f"{path_stem}.png", rgb)
        save_png(f"{path_stem}.alpha.png", alpha)
        if a.mode == "both":
            save_png(f"{path_stem}.inner.png", ain / alpha.clamp_min(1e-4) * (alpha > 0.02))

    if a.orbit:
        sched = json.loads((Path(a.orbit) / "orbit.json").read_text()); oc = io.load_cameras(Path(a.orbit))
        for i, (cid, t) in enumerate(zip(sched["cams"], sched["frame_rel"])):
            shot(Camera(cid, oc[cid], "cuda"), t, out / f"{i:04d}")
        print("orbit ->", out)
        return
    cams = a.cams or (data.interp_ids + data.extrap_ids)
    frames = a.frames if a.frames is not None else list(range(data.num_frames))
    for cid in cams:
        if cid not in data.cameras:
            data.cameras[cid] = Camera(cid, io.load_cameras(data.root)[cid], "cuda")
        d = out / cid; d.mkdir(parents=True, exist_ok=True)
        for t in frames:
            shot(data.cameras[cid], t, d / f"{data.frames[t]:06d}")
    print(f"rendered {len(cams)} cams x {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
