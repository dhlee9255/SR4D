"""Train the outer layer of the 2-layer model (docs/DECISIONS.md D-040).

  python -m sr4d.layered.train --root <data> --setting gen48_ho8 --inner <sr4d ckpt_final.pt> --out <dir> \
      [--no_render_inner] [--iters 30000] [--seed 0]
--no_render_inner: outer layer only (EXP-001); the inner checkpoint is still used for the keyframe initialisation.
Writes <out>/outer_<it>.pt, <out>/outer_final.pt, <out>/train.log, <out>/log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import torch

from sr4d.data.dataset import MultiViewSequence
from sr4d.layered.core import FPS, InnerCache, init_outer, render2
from sr4d.train import losses

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "ftgs_outer"))
from outer import (DEFAULT_LRS, DEFAULT_SCHED, TrainState, make_optimizers, relocate_mcmc,  # noqa: E402
                   relocation_binoms, scene_extent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--setting", required=True)
    ap.add_argument("--inner", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no_render_inner", action="store_true")
    ap.add_argument("--inner_drop", type=float, default=0.0, help="prob. of rendering a training step WITHOUT the inner layer")
    ap.add_argument("--iters", type=int, default=30000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_outer", type=int, default=500_000)
    ap.add_argument("--lpips_w", type=float, default=0.05); ap.add_argument("--lpips_start", type=int, default=1000)
    ap.add_argument("--opac_w", type=float, default=0.01); ap.add_argument("--scale_w", type=float, default=0.01)
    ap.add_argument("--gate_w", type=float, default=0.001)
    ap.add_argument("--dur_w", type=float, default=0.01); ap.add_argument("--dur_min_frames", type=float, default=3.0)
    ap.add_argument("--vel_w", type=float, default=0.01); ap.add_argument("--mask_w", type=float, default=0.0)
    ap.add_argument("--reloc_start", type=int, default=500); ap.add_argument("--reloc_stop", type=int, default=25000)
    ap.add_argument("--reloc_every", type=int, default=100); ap.add_argument("--dead", type=float, default=0.005)
    ap.add_argument("--cov_min_views", type=int, default=0,
                    help="EXP-005 (D-042): outer Gaussians seen by fewer training cameras are not cloned and are pruned; 0 = off")
    ap.add_argument("--cov_vis_min", type=float, default=1.0, help="camera counts as seen if est. contribution per replay >= this (px)")
    ap.add_argument("--cov_decay", type=float, default=0.97, help="decay of the view statistics every reloc_every iterations")
    ap.add_argument("--cov_start", type=int, default=3000); ap.add_argument("--cov_prune_every", type=int, default=1000)
    ap.add_argument("--rand_bg", action="store_true", help="EXP-008 (D-043): random background colour every iteration (GT composited with the mask)")
    ap.add_argument("--max_ratio", type=float, default=0.0, help="EXP-008 (b): cap max/min axis of outer Gaussians (projection after each step); 0 = off")
    ap.add_argument("--vh_prune", action="store_true", help="EXP-008 (b): at every relocation mark outer Gaussians outside the dilated training-view visual hull (at their time centre) dead")
    ap.add_argument("--vh_dilate", type=int, default=15)
    ap.add_argument("--save_at", type=int, nargs="*", default=[2000]); ap.add_argument("--eval_every", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a"); jl = open(out / "log.jsonl", "a")

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    data = MultiViewSequence(a.root, a.setting)
    eval_ids = data.interp_ids + data.extrap_ids
    log(f"data {a.root} {a.setting}: {len(data.train_ids)} train cams, {len(eval_ids)} eval cams, {data.num_frames} frames")
    t0 = time.time()
    inner = InnerCache(Path(a.inner), data)
    log(f"inner cache: {inner.n} Gaussians x {data.num_frames} frames, sh {inner.sh_degree} ({time.time() - t0:.0f}s)")
    gs = init_outer(inner, data, n_total=a.n_outer, seed=a.seed, log=log).cuda()
    log(f"outer init: {len(gs)} Gaussians ({time.time() - t0:.0f}s)")
    use_inner = not a.no_render_inner
    w2cs = torch.stack([data.cameras[c].w2c for c in data.train_ids])
    ext = scene_extent(w2cs)
    opts, scheds = make_optimizers(gs, DEFAULT_LRS, DEFAULT_SCHED, a.iters, ext)
    log(f"scene extent {ext:.3f}; render inner: {use_inner}")
    state = TrainState(len(gs)); binoms = relocation_binoms().cuda()
    lp_fn = lpips.LPIPS(net="vgg", verbose=False).cuda().eval()
    for p in lp_fn.parameters():
        p.requires_grad_(False)
    sig_min = a.dur_min_frames / FPS / 2.355
    gen = torch.Generator().manual_seed(a.seed)
    pairs = [(t, c) for t in range(data.num_frames) for c in data.train_ids]
    cam_idx = {c: i for i, c in enumerate(data.train_ids)}
    if a.vh_prune:
        import cv2
        ker = np.ones((2 * a.vh_dilate + 1,) * 2, np.uint8)
        vh_masks = {c: torch.as_tensor(np.stack([cv2.dilate(m.cpu().numpy(), ker) for m in data.masks[c]])).cuda() for c in data.train_ids}

    @torch.no_grad()
    def outside_vh():  # same criterion as tools/count_bg_streaks.py (without the opacity / axis-ratio conditions)
        tc = (gs.times[:, 0] * FPS).round().clamp(0, data.num_frames - 1).long()
        mo = gs.means + (tc[:, None].float() / FPS - gs.times) * gs.velocities
        o = torch.zeros(len(gs), dtype=torch.bool, device="cuda")
        for c in data.train_ids:
            cm = data.cameras[c]; pc = mo @ cm.w2c[:3, :3].T + cm.w2c[:3, 3]; z = pc[:, 2].clamp_min(1e-6)
            u = (pc[:, 0] / z * cm.K[0, 0] + cm.K[0, 2]).long(); v = (pc[:, 1] / z * cm.K[1, 1] + cm.K[1, 2]).long()
            inimg = (pc[:, 2] > 0.05) & (u >= 0) & (u < cm.W) & (v >= 0) & (v < cm.H)
            o |= inimg & ~(vh_masks[c][tc, v.clamp(0, cm.H - 1), u.clamp(0, cm.W - 1)] > 127)
        return o
    cov = a.cov_min_views > 0
    if cov:  # per outer Gaussian x training camera: decayed contribution sum_px T*alpha, and decayed #steps per camera
        cov_c = torch.zeros(len(gs), len(data.train_ids), device="cuda"); cov_n = torch.zeros(len(data.train_ids), device="cuda")

    def n_views():
        est = cov_c / cov_n.clamp_min(1)[None] * data.num_frames  # ~ contribution over one replay of all frames
        return (est >= a.cov_vis_min).sum(1)
    tic = time.time(); acc = {}
    for it in range(a.iters + 1):
        t, cid = pairs[int(torch.randint(len(pairs), (1,), generator=gen))]
        cam = data.cameras[cid]
        sh = min(it // 1000, gs.sh_degree)
        ui = use_inner and not (a.inner_drop > 0 and torch.rand(1, generator=gen).item() < a.inner_drop)
        probe = torch.zeros(len(gs), device="cuda", requires_grad=True) if cov else None
        bgc = torch.rand(3, generator=gen).cuda() if a.rand_bg else 1.0
        rgb, alpha, ain, meta, n_in = render2(inner, gs, t, cam.K, cam.w2c, cam.W, cam.H, sh, use_inner=ui, outer_probe=probe, bg=bgc)
        meta["means2d"].retain_grad()
        gt, mask, raw = data.get(cid, t)
        if a.rand_bg:
            gt = raw * mask[..., None] + bgc * (1 - mask[..., None])
        loss_ph = losses.photometric(rgb, gt, 0.2)
        loss = loss_ph
        lp = torch.zeros((), device="cuda")
        if a.lpips_w > 0 and it >= a.lpips_start:
            lp = losses.lpips_loss(lp_fn, rgb.clamp(0, 1), gt, mask)
            loss = loss + a.lpips_w * lp
        ts = t / FPS
        tem = gs.temporal_opacity(ts)
        reg_op = (gs.opacities.sigmoid() * tem.detach()).mean()
        reg_sc = gs.scales.exp().mean()
        reg_gate = (1 - gs.gate()).mean()
        reg_dur = (torch.relu(sig_min - gs.sigma_t()) / sig_min).mean()
        reg_vel = gs.velocities.norm(dim=-1).mean()
        loss = loss + a.opac_w * reg_op + a.scale_w * reg_sc + a.gate_w * reg_gate + a.dur_w * reg_dur + a.vel_w * reg_vel
        if a.mask_w > 0:
            loss = loss + a.mask_w * losses.mask_loss(alpha, mask)
        if cov:
            loss = loss + meta["probe"].sum()  # == 0, leaves every other gradient unchanged; probe.grad = contribution
        loss.backward()
        if cov:
            with torch.no_grad():
                cov_c[:, cam_idx[cid]] += probe.grad; cov_n[cam_idx[cid]] += 1
        for o in opts.values():
            o.step()
        for s in scheds:
            s.step()
        if a.max_ratio > 0:
            with torch.no_grad():
                lo = gs.scales.max(1, keepdim=True).values - float(np.log(a.max_ratio))
                gs.scales.data = torch.maximum(gs.scales.data, lo)
        g2 = meta["means2d"].grad[0, n_in:]
        radii = meta["radii"][0, n_in:]
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        state.update_(g2, radii, cam.W, cam.H)
        nrel = None
        if a.vh_prune and it % a.reloc_every == 0 and it > a.reloc_start:  # also after reloc_stop (then just stays dead)
            with torch.no_grad():
                kill_vh = outside_vh() & (gs.opacities.sigmoid()[:, 0] > a.dead)
                gs.opacities.data[kill_vh] = float(np.log(a.dead / 2 / (1 - a.dead / 2)))
                acc["vh_pruned"] = acc.get("vh_pruned", 0) + int(kill_vh.sum())
        if it % a.reloc_every == 0 and a.reloc_start < it < a.reloc_stop:
            src_ok, rows = None, ()
            if a.vh_prune:
                src_ok = ~kill_vh
            if cov:
                with torch.no_grad():
                    if it >= a.cov_start:
                        nv = n_views(); src_ok = nv >= a.cov_min_views
                        if it % a.cov_prune_every == 0:  # prune = mark dead -> relocated onto well-seen Gaussians below
                            kill = (~src_ok) & (gs.opacities.sigmoid()[:, 0] > a.dead)
                            gs.opacities.data[kill] = float(np.log(a.dead / 2 / (1 - a.dead / 2)))
                            acc["pruned"] = acc.get("pruned", 0) + int(kill.sum())
                            acc["low_view_frac"] = float((~src_ok).float().mean())
                    rows = (cov_c,)
            nrel = relocate_mcmc(gs, opts, state, binoms, a.dead, src_ok=src_ok, rows=rows)
            state.zero_()
            if cov:
                cov_c.mul_(a.cov_decay); cov_n.mul_(a.cov_decay)
        for k, v in (("loss", loss), ("photo", loss_ph), ("lpips", lp), ("reg_op", reg_op), ("reg_dur", reg_dur), ("reg_vel", reg_vel)):
            acc[k] = acc.get(k, 0.0) + float(v)
        if nrel is not None:
            acc["relocated"] = acc.get("relocated", 0) + nrel
        if it % 100 == 0 and it > 0:
            with torch.no_grad():
                frac_in = float((ain.sum() / alpha.sum().clamp_min(1e-6))) if ui else -1.0
                sig = gs.sigma_t()[:, 0] * FPS
                special = ("relocated", "pruned", "low_view_frac", "vh_pruned")
                rec = {"it": it, **{k: v / 100 for k, v in acc.items() if k not in special}, "relocated": acc.get("relocated", 0),
                       **{k: acc[k] for k in ("pruned", "low_view_frac", "vh_pruned") if k in acc},
                       "inner_share_train": frac_in, "sigma_frames_med": float(sig.median()), "gate_on": float((gs.gate() > 0.5).float().mean()),
                       "vel_med": float(gs.velocities.norm(dim=-1).median()), "opac_alive": float((gs.opacities.sigmoid() > a.dead).float().mean()),
                       "s_per_it": (time.time() - tic) / 100, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            jl.write(json.dumps(rec) + "\n"); jl.flush()
            log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            acc = {}; tic = time.time()
        if (it > 0 and it % a.eval_every == 0) or it == a.iters:
            with torch.no_grad():
                ps = []
                for c in eval_ids[:8]:
                    for tt in (0, data.num_frames // 2, data.num_frames - 1):
                        cm = data.cameras[c]
                        r, _, _, _, _ = render2(inner, gs, tt, cm.K, cm.w2c, cm.W, cm.H, gs.sh_degree, use_inner=use_inner)
                        g, _, _ = data.get(c, tt)
                        ps.append(float(-10 * torch.log10(((r.clamp(0, 1) - g) ** 2).mean())))
                log(f"EVAL it {it}: held-out PSNR (8 cams x 3 frames) {np.mean(ps):.2f}")
                jl.write(json.dumps({"it": it, "eval_psnr": float(np.mean(ps))}) + "\n"); jl.flush()
        if it in a.save_at or it == a.iters:
            if cov and it == a.iters:
                torch.save({"n_views": n_views().cpu(), "cov_c": cov_c.half().cpu(), "cov_n": cov_n.cpu()}, out / "cov_final.pt")
            torch.save({"outer": gs.state(), "args": vars(a), "it": it}, out / (f"outer_{it:06d}.pt" if it != a.iters else "outer_final.pt"))
            log(f"saved it {it}")
    log("DONE")


if __name__ == "__main__":
    main()
