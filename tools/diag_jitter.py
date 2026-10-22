"""Temporal jitter diagnosis for an SR4D checkpoint: RMS 2nd difference (acceleration, mm/frame^2) of
(a) SMPL joints from the input poses, (b) joints from the optimised poses, and (c) deformed Gaussian centres under
different layer settings. Large (c) vs (a) => jitter added by the model; large (a) => input pose jitter.
  python tools/diag_jitter.py --ckpt outputs/sr4d/387_gen/gen24/ckpt_final.pt
"""
import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data.dataset import MultiViewSequence  # noqa: E402
from sr4d.train.trainer import load_model  # noqa: E402


def acc(x):  # (T, N, 3) -> RMS of 2nd difference in mm, and RMS velocity in mm
    a = x[2:] - 2 * x[1:-1] + x[:-2]
    v = x[1:] - x[:-1]
    return float(a.norm(dim=-1).pow(2).mean().sqrt() * 1000), float(v.norm(dim=-1).pow(2).mean().sqrt() * 1000)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", default=None, help="write accel RMS + body/free Gaussian stats here")
    a = ap.parse_args()
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(st["cfg"])
    data = MultiViewSequence(cfg.data.root, cfg.data.setting, list(range(*cfg.data.frames)), load_images=False)
    m = load_model(Path(a.ckpt), data, cfg).eval()
    sc = m.scaffold
    T = data.num_frames


    def run(label, **kw):
        xs = torch.stack([m.deform(t, need_aux=False, **kw)["means"] for t in range(T)])
        ac, ve = acc(xs)
        print(f"{label:38s} accel RMS {ac:6.2f} mm   vel RMS {ve:6.2f} mm   ratio {ac / ve:.3f}")
        return xs

    m.body_adhere = False
    import json
    stats = {}
    for name in ("body", "free"):
        gs = getattr(m, name)
        if gs is not None:
            op = torch.sigmoid(gs["opacity"]).reshape(-1)
            stats[name] = dict(n=int(op.numel()), opac_sum=float(op.sum()), n_visible=int((op > 0.05).sum()))
    print("gaussians:", json.dumps(stats))
    m.use_l2, m.use_l3, m.use_l4 = m.l2 is not None, m.l3 is not None, m.l4 is not None
    xs = run("full (replay: L2+bones+L3)", replay=True)
    stats["accel_rms_mm"], stats["vel_rms_mm"] = acc(xs)
    run("L3 off (animate)", replay=False)
    if m.l2 is not None:
        m.use_l2 = False
        run("L3 off, L2 off", replay=False)
        m.use_l2 = True
    # bones off
    if getattr(m, "bones", None) is not None:
        # zeroing all bone motion params gives identity for every mode (free / MLP last layer / spline ctrl)
        saved = [q.data.clone() for q in m.bones.motion_parameters()]
        for q in m.bones.motion_parameters():
            q.data.zero_()
        run("L3 off, free bones identity", replay=False)
        for q, v in zip(m.bones.motion_parameters(), saved):
            q.data.copy_(v)
    # input poses instead of optimised ones
    p, tl = sc.pose.detach().clone(), sc.transl.detach().clone()
    snap = sc.reset_to_input()
    run("L3 off, INPUT SMPL poses", replay=False)
    sc.restore(snap)
    if a.json:
        Path(a.json).write_text(json.dumps(stats, indent=1))
    print("scaffold pose change vs input: mean |dpose| %.4f rad, |dtransl| %.2f mm" % (
        (p - sc.pose_init).norm(dim=-1).mean(), (tl - sc.transl_init).norm(dim=-1).mean() * 1000))
    d = p - sc.pose_init
    print("   frame-to-frame variation of the pose correction: %.4f rad (2nd diff %.4f)" % (
        (d[1:] - d[:-1]).norm(dim=-1).mean(), (d[2:] - 2 * d[1:-1] + d[:-2]).norm(dim=-1).mean()))


if __name__ == "__main__":
    main()
