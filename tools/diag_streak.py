"""Diagnosis of the white streaks above the head (user 2026-10-18 item 1A) for one layered run (New-fade).
1) pick the (eval cam, frame) pairs with the most "streak pixels" in the crown box: GT mask inside and GT dark (hair,
   luminance < 0.35) but the New-fade render bright (> 0.75);
2) render GT / New-fade / inner only / outer only (fade, on black) / outer only (no fade, on black) for the top pairs;
3) streak Gaussians = outer Gaussians carrying 80% of the contribution sum_px T*alpha inside the streak pixels (joint
   render with the inner layer and the fade, exact via a zero probe channel); report colour at this view, opacity,
   max/min axis ratio, speed, duration (FWHM frames), outside the dilated training-view visual hull at that time,
   #training cams that saw it (vis_outer_final.pt, contribution >= 1 px), against the contribution-weighted
   reference of all outer Gaussians visible in the foreground of the same view.
  PYTHONPATH=. python tools/diag_streak.py --run outputs/layered/exp003b_dance_r15h35_both_drop05 --out <dir> [--top 2]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from sr4d.data import io
from sr4d.data.dataset import Camera, MultiViewSequence
from sr4d.layered.core import FPS, InnerCache, render2, save_png
from sr4d.layered.fade import fade_factor
from sr4d.model.body_model import BodyModel
from tools.crown_crop import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "ftgs_outer"))
from outer import TimeGaussians  # noqa: E402

lum = lambda x: x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114  # noqa: E731


def wq(x, w, qs=(0.5,)):  # weighted quantiles
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return [float(x[o][np.searchsorted(c, q)]) for q in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--frames", type=int, nargs="*", default=[0, 10, 20, 30, 40, 50, 59]); ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--fade", type=float, nargs=2, default=[10, 25])
    a = ap.parse_args()
    run = Path(a.run); args = json.loads((run / "args.json").read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = MultiViewSequence(args["root"], args["setting"])
    R = Path(args["root"]); cams = io.load_cameras(R); meta = io.load_meta(R); bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
    inner = InnerCache(Path(args["inner"]), data)
    g = TimeGaussians.from_state(torch.load(run / "outer_final.pt", map_location="cpu", weights_only=False)["outer"]).cuda()
    vis = {k: v.cuda() for k, v in torch.load(run / "vis_outer_final.pt").items() if torch.is_tensor(v)}
    nseen = (vis["contrib"] >= 1).sum(1)
    sh = g.sh_degree
    ker = np.ones((2 * a.dilate + 1,) * 2, np.uint8)
    dil = {c: torch.as_tensor(np.stack([cv2.dilate(m.cpu().numpy(), ker) for m in data.masks[c]])).cuda() for c in data.train_ids}
    eval_ids = data.interp_ids + data.extrap_ids
    cand = []
    with torch.no_grad():
        for cid in eval_ids:
            cam = data.cameras[cid]
            for t in a.frames:
                mul = fade_factor(g, vis, t, cam.center, *a.fade)
                rgb, *_ = render2(inner, g, t, cam.K, cam.w2c, cam.W, cam.H, sh, outer_opac_mul=mul)
                gt, m, _ = data.get(cid, t)
                cx, cy, s = box(R, cams[cid], bm, data.frames[t])
                B = torch.zeros_like(m, dtype=torch.bool); B[max(cy - s // 2, 0):cy + s // 2, max(cx - s // 2, 0):cx + s // 2] = True
                S = B & (m > 0.5) & (lum(gt) < 0.35) & (lum(rgb) > 0.75)
                cand.append((int(S.sum()), cid, t))
    cand.sort(reverse=True)
    rep = {"run": str(run), "candidates_top10": cand[:10], "views": []}
    for npx, cid, t in cand[: a.top]:
        cam = data.cameras[cid]; ts = t / FPS; stem = out / f"{cid}_f{t:02d}"
        gt, m, _ = data.get(cid, t)
        mul = fade_factor(g, vis, t, cam.center, *a.fade)
        with torch.no_grad():
            rgb, al, ain, _, _ = render2(inner, g, t, cam.K, cam.w2c, cam.W, cam.H, sh, outer_opac_mul=mul)
            ri, *_ = render2(inner, g, t, cam.K, cam.w2c, cam.W, cam.H, sh, use_outer=False)
            ro, alo, *_ = render2(inner, g, t, cam.K, cam.w2c, cam.W, cam.H, sh, use_inner=False, outer_opac_mul=mul, bg=0.0)
            ron, *_ = render2(inner, g, t, cam.K, cam.w2c, cam.W, cam.H, sh, use_inner=False, bg=0.0)
        cx, cy, s = box(R, cams[cid], bm, data.frames[t])
        B = torch.zeros_like(m, dtype=torch.bool); B[max(cy - s // 2, 0):cy + s // 2, max(cx - s // 2, 0):cx + s // 2] = True
        S = B & (m > 0.5) & (lum(gt) < 0.35) & (lum(rgb) > 0.75)
        gtw = gt * m[..., None] + (1 - m[..., None])
        for nm, im in (("gt", gtw), ("newfade", rgb), ("inner_only", ri), ("outer_only_fade_black", ro), ("outer_only_nofade_black", ron)):
            save_png(f"{stem}_{nm}.png", im)
        save_png(f"{stem}_streakmask.png", S.float())
        # contributions: inside streak pixels, and over the whole foreground (reference)
        res = {}
        for key, W in (("streak", S.float()), ("fg", (m > 0.5).float())):
            p = torch.zeros(len(g), device="cuda", requires_grad=True)
            _, _, _, mt, _ = render2(inner, g, t, cam.K, cam.w2c, cam.W, cam.H, sh, outer_opac_mul=mul, outer_probe=p)
            (mt["probe"] * W).sum().backward()
            res[key] = p.grad.detach().clone()
        w = res["streak"]; o = torch.argsort(w, descending=True); c = torch.cumsum(w[o], 0) / w.sum().clamp_min(1e-9)
        sel = o[: int((c < 0.8).sum()) + 1]
        with torch.no_grad():
            mo = g.means_t(ts)
            col = torch.clamp_min(__import__("gsplat").spherical_harmonics(sh, mo - cam.center, g.shs()) + 0.5, 0).clamp(0, 1)
            op_t = g.opacities_t(ts)[:, 0]; ratio = (g.scales.exp().max(1).values / g.scales.exp().min(1).values)
            speed = g.velocities.norm(dim=-1); fwhm = g.sigma_t()[:, 0] * FPS * 2.355
            outside = torch.zeros(len(g), dtype=torch.bool, device="cuda")
            for c2 in data.train_ids:
                cm = data.cameras[c2]; pc = mo @ cm.w2c[:3, :3].T + cm.w2c[:3, 3]; z = pc[:, 2].clamp_min(1e-6)
                u = (pc[:, 0] / z * cm.K[0, 0] + cm.K[0, 2]).long(); v = (pc[:, 1] / z * cm.K[1, 1] + cm.K[1, 2]).long()
                inimg = (pc[:, 2] > 0.05) & (u >= 0) & (u < cm.W) & (v >= 0) & (v < cm.H)
                mv = dil[c2][t][v.clamp(0, cm.H - 1), u.clamp(0, cm.W - 1)] > 127
                outside |= inimg & ~mv
        def stats(idx, wt):
            wt = wt[idx].cpu().numpy().astype(np.float64); f = lambda x: x[idx].cpu().numpy().astype(np.float64)  # noqa: E731
            return {"n": int(len(idx)), "colour_lum_med": wq(f(lum(col)), wt)[0], "frac_lum>0.8": float((wt * (f(lum(col)) > 0.8)).sum() / wt.sum()),
                    "opacity_t_med": wq(f(op_t), wt)[0], "fade_mul_med": wq(f(mul), wt)[0],
                    "axis_ratio_med_p90": wq(f(ratio), wt, (0.5, 0.9)), "frac_ratio>10": float((wt * (f(ratio) > 10)).sum() / wt.sum()),
                    "speed_mps_med": wq(f(speed), wt)[0], "fwhm_frames_med": wq(f(fwhm), wt)[0],
                    "frac_outside_vh": float((wt * f(outside.float())).sum() / wt.sum()),
                    "frac_seen_ge12": float((wt * (f(nseen.float()) >= 12)).sum() / wt.sum()), "nseen_med": wq(f(nseen.float()), wt)[0]}
        ref_idx = torch.nonzero(res["fg"] > 0)[:, 0]
        v = {"cam": cid, "frame": data.frames[t], "streak_px": npx, "streak": stats(sel, w), "reference_fg": stats(ref_idx, res["fg"]),
             "streak_share_of_crown_sqerr": float((((rgb - gt) ** 2).sum(-1) * S).sum() / (((rgb - gt) ** 2).sum(-1) * B * (m > 0.5)).sum()),
             "outer_alpha_in_streak_px_mean": float(alo[S].mean()), "inner_alpha_in_streak_px_mean": float((ain)[S].mean())}
        rep["views"].append(v)
        torch.save({"sel": sel.cpu(), "w": w[sel].cpu()}, f"{stem}_streak_ids.pt")
        print(json.dumps(v, indent=1))
    (out / "report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
