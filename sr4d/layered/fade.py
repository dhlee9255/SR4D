"""EXP-004 New-fade (docs/DECISIONS.md D-041): view-coverage fade of the outer layer, render-time only (no retraining).

1) record: replay every training (camera, frame) with the trained outer layer alone (as in the inner-dropout steps) and
   accumulate, per outer Gaussian and training camera, its rendered contribution sum_pixels(T*alpha) and the
   contribution-weighted direction Gaussian -> camera centre.
     python -m sr4d.layered.fade --run <train out dir> [--ckpt outer_final.pt]   -> <run>/vis_<ckpt stem>.pt
2) render: a camera counts as "seen" for a Gaussian when its total contribution >= vis_min (pixels). The outer opacity is
   multiplied by fade(theta), theta = angle between the current view direction and the nearest seen direction:
   1 for theta <= th1, 0 for theta >= th2, smoothstep in between; never-seen Gaussians -> 0.  (fade_factor)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gsplat
import torch

from sr4d.data.dataset import MultiViewSequence
from sr4d.layered.core import FPS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "ftgs_outer"))
from outer import TimeGaussians  # noqa: E402


def record(outer: TimeGaussians, data) -> dict:
    N, C = len(outer), len(data.train_ids)
    contrib = torch.zeros(N, C, device="cuda")
    dsum = torch.zeros(N, C, 3, device="cuda")
    cov = outer.covars().detach()
    for ci, cid in enumerate(data.train_ids):
        cam = data.cameras[cid]
        for t in range(data.num_frames):
            ts = t / FPS
            with torch.no_grad():
                mo = outer.means_t(ts).detach(); op = outer.opacities_t(ts)[:, 0].detach()
            f = torch.zeros(N, 1, device="cuda", requires_grad=True)
            out, _, _ = gsplat.rasterization(mo, None, None, op, f, cam.w2c[None], cam.K[None], cam.W, cam.H,
                                             covars=cov, packed=False, rasterize_mode="antialiased")
            out.sum().backward()  # d(sum_px sum_i w_i f_i)/d f_i = sum_px w_i
            w = f.grad[:, 0]
            d = torch.nn.functional.normalize(cam.center[None] - mo, dim=-1)
            contrib[:, ci] += w; dsum[:, ci] += w[:, None] * d
    return {"contrib": contrib.cpu(), "dirs": torch.nn.functional.normalize(dsum, dim=-1).half().cpu(), "cams": list(data.train_ids)}


def smoothstep(x):
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


@torch.no_grad()
def fade_factor(outer: TimeGaussians, vis: dict, t: int, campos, th1: float, th2: float, vis_min: float = 1.0):
    """(N,) multiplier for the outer opacities. vis tensors must be on cuda; th in degrees."""
    mo = outer.means_t(t / FPS)
    v = torch.nn.functional.normalize(campos[None] - mo, dim=-1)
    cos = (vis["dirs"].float() * v[:, None]).sum(-1)
    cos = torch.where(vis["contrib"] >= vis_min, cos, torch.full_like(cos, -2.0))
    theta = torch.rad2deg(torch.arccos(cos.max(1).values.clamp(-1, 1)))
    return 1 - smoothstep((theta - th1) / max(th2 - th1, 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="outer_final.pt")
    a = ap.parse_args()
    run = Path(a.run); args = json.loads((run / "args.json").read_text())
    data = MultiViewSequence(args["root"], args["setting"], load_images=False)
    outer = TimeGaussians.from_state(torch.load(run / a.ckpt, map_location="cpu", weights_only=False)["outer"]).cuda()
    vis = record(outer, data)
    out = run / f"vis_{Path(a.ckpt).stem}.pt"
    torch.save(vis, out)
    c = vis["contrib"]
    n_seen = (c >= 1.0).sum(1)
    print(f"{out}: {len(outer)} Gaussians x {c.shape[1]} train cams; seen by >=1 cam {float((n_seen > 0).float().mean()) * 100:.1f}%, "
          f"median #cams {float(n_seen.float().median()):.0f}")


if __name__ == "__main__":
    main()
