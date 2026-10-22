"""Count background-coloured streak floaters in an FTGS-style time-Gaussian state (layered outer layer or FTGS++ dump).
Fixed definition (user item 1/2, 2026-10-18; decided before EXP-008):
  floater  = at its own time centre (clamped to the sequence) the centre is outside the training-view visual hull
             (every in-image training mask dilated by 15 px), peak opacity sigmoid(o) >= 0.1, max/min axis >= 10
  colour   = DC colour luminance > 0.85 -> white, < 0.15 -> black
  python tools/count_bg_streaks.py --root <data> --setting <split> --state <ckpt with "outer"> [--state ...]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from sr4d.data.dataset import MultiViewSequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "ftgs_outer"))
from outer import TimeGaussians  # noqa: E402

C0 = 0.28209479177387814


@torch.no_grad()
def count(g, data, dil, fps=30.0):
    T = data.num_frames
    tc = (g.times[:, 0] * fps).round().clamp(0, T - 1).long()
    mo = g.means + (tc[:, None].float() / fps - g.times) * g.velocities
    outside = torch.zeros(len(g), dtype=torch.bool, device="cuda")
    for c in data.train_ids:
        cm = data.cameras[c]; pc = mo @ cm.w2c[:3, :3].T + cm.w2c[:3, 3]; z = pc[:, 2].clamp_min(1e-6)
        u = (pc[:, 0] / z * cm.K[0, 0] + cm.K[0, 2]).long(); v = (pc[:, 1] / z * cm.K[1, 1] + cm.K[1, 2]).long()
        inimg = (pc[:, 2] > 0.05) & (u >= 0) & (u < cm.W) & (v >= 0) & (v < cm.H)
        outside |= inimg & ~(dil[c][tc, v.clamp(0, cm.H - 1), u.clamp(0, cm.W - 1)] > 127)
    s = g.scales.exp(); ratio = s.max(1).values / s.min(1).values
    op = g.opacities.sigmoid()[:, 0]
    lum = ((g.sh_0[:, 0] * C0 + 0.5).clamp(0, 1) * torch.tensor([0.299, 0.587, 0.114], device="cuda")).sum(-1)
    fl = outside & (op >= 0.1) & (ratio >= 10)
    return {"n": len(g), "floater": int(fl.sum()), "white": int((fl & (lum > 0.85)).sum()), "black": int((fl & (lum < 0.15)).sum()),
            "outside_vh_op>=0.1": int((outside & (op >= 0.1)).sum()), "ratio>=10_frac": float((ratio >= 10).float().mean()),
            "ratio_median": float(ratio.median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--setting", required=True); ap.add_argument("--state", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    data = MultiViewSequence(a.root, a.setting)
    ker = np.ones((31, 31), np.uint8)
    dil = {c: torch.as_tensor(np.stack([cv2.dilate(m.cpu().numpy(), ker) for m in data.masks[c]])).cuda() for c in data.train_ids}
    res = {}
    for p in a.state:
        g = TimeGaussians.from_state(torch.load(p, map_location="cpu", weights_only=False)["outer"]).cuda()
        res[p] = count(g, data, dil); print(p, json.dumps(res[p]))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
