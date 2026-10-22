"""White-rim diagnosis on held-out views (user step 2, 2026-10-13), no training.
 a) rim error by outward-normal direction (8 bins) and per-view error maps of the 3 px boundary band
 b) L4 camera refinement of the SR4D checkpoint: per-view rotation (deg) and camera-centre shift in WORLD frame (mm);
    a non-zero mean = the whole scene sits shifted w.r.t. the original cameras (held-out views use unrefined cameras)
 c) global image shift render vs GT on held-out views (phase correlation on masked luminance), every method
 d) render alpha inside the GT mask boundary band (inner 3 px) vs interior, every method
usage: PYTHONPATH=. python tools/diag_edge.py --root <data> --setting gen48_ho8 --run LABEL=<dir> ... --ckpt <sr4d ckpt> --out <dir>
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sr4d.data import io
from sr4d.utils.geometry import rotvec_to_rotmat


def rings(m):
    k = np.ones((7, 7), np.uint8)
    inner = m & ~(cv2.erode(m.astype(np.uint8), k) > 0)            # 3 px inside the GT boundary
    outer = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~m            # 3 px outside
    return inner, outer


def normal_bins(m, band):
    d_in = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    sd = d_out - d_in                                                 # signed distance, + outside
    gy, gx = np.gradient(cv2.GaussianBlur(sd, (0, 0), 2))
    ang = (np.degrees(np.arctan2(-gy, gx)) + 360) % 360               # outward normal, 0 = right, 90 = up
    return ((ang + 22.5) // 45).astype(int) % 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--setting", required=True)
    ap.add_argument("--run", nargs="+", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=list(range(0, 60, 5)))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sp = io.load_split(root, a.setting); cams = io.load_cameras(root)
    test = sp["test_interp_cams"] + sp["test_extrap_cams"]
    runs = [r.split("=", 1) for r in a.run]
    res = {"rim_by_direction": {}, "global_shift_px": {}, "alpha": {}, "band_err": {}}
    DIRS = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    vis = {lab: [] for lab, _ in runs}
    for lab, d in runs:
        bins = np.zeros(8); cnt = np.zeros(8); shifts = []; a_in = []; a_int = []; e_in = []; e_out = []; e_int = []
        for c in test:
            emap = None
            for f in a.frames:
                img = io.read_image(root / f"images/{c}/{f:06d}.jpg").astype(np.float32)
                m = io.read_mask(root / f"masks/{c}/{f:06d}.png") > 127
                gt = np.where(m[..., None], img, 255.0)
                r = io.read_image(Path(d) / f"renders/{c}/{f:06d}.png").astype(np.float32)
                al = cv2.imread(str(Path(d) / f"renders/{c}/{f:06d}.alpha.png"), 0).astype(np.float32) / 255
                e = np.abs(r - gt).mean(-1)
                inner, outer = rings(m)
                interior = m & ~inner
                nb = normal_bins(m, inner | outer)
                band = inner | outer
                np.add.at(bins, nb[band], e[band]); np.add.at(cnt, nb[band], 1)
                e_in.append(e[inner].mean()); e_out.append(e[outer].mean()); e_int.append(e[interior].mean())
                a_in.append(al[inner].mean()); a_int.append(al[interior].mean())
                # luminance inside a dilated mask (background white for both), phase correlation (subpixel)
                win = cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(np.float32)
                g1 = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) * win
                g2 = cv2.cvtColor(r.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) * win
                (dx, dy), _ = cv2.phaseCorrelate(g1, g2, cv2.createHanningWindow(g1.shape[::-1], cv2.CV_32F))
                shifts.append((dx, dy))
                if f == 30:
                    emap = np.where(band, e, 0)
            if emap is not None:
                vis[lab].append(cv2.applyColorMap(np.clip(emap * 2, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
        s = np.array(shifts)
        res["rim_by_direction"][lab] = {DIRS[i]: round(float(bins[i] / max(cnt[i], 1)), 2) for i in range(8)}
        res["global_shift_px"][lab] = dict(mean_dx=round(float(s[:, 0].mean()), 3), mean_dy=round(float(s[:, 1].mean()), 3),
                                           mean_abs=round(float(np.linalg.norm(s, axis=1).mean()), 3))
        res["alpha"][lab] = dict(inner_band=round(float(np.mean(a_in)), 4), interior=round(float(np.mean(a_int)), 4))
        res["band_err"][lab] = dict(inner=round(float(np.mean(e_in)), 2), outer=round(float(np.mean(e_out)), 2), interior=round(float(np.mean(e_int)), 2))
    # b) L4 camera refinement
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = st["cfg"]; l4 = st["l4"]; dl = l4["cam_delta"].float()
    from omegaconf import OmegaConf
    c4 = OmegaConf.create(cfg).model.l4
    import math
    max_rot, max_trans = math.radians(float(c4.max_cam_rot_deg)), float(c4.max_cam_trans_m)
    allc = sorted(cams)  # MultiViewSequence.cam_index = enumerate(sorted(all cameras.json ids))
    rot_deg, dC = [], []
    for i, c in enumerate(allc):
        if c not in sp["train_cams"]:
            continue
        R = rotvec_to_rotmat(torch.tanh(dl[i, :3]) * max_rot); t = torch.tanh(dl[i, 3:]) * max_trans
        Rw, tw = torch.tensor(cams[c]["R"]).float(), torch.tensor(cams[c]["t"]).float().reshape(3)
        C0 = -Rw.T @ tw; R1, t1 = R @ Rw, R @ tw + t; C1 = -R1.T @ t1
        rot_deg.append(float(torch.rad2deg(torch.linalg.norm(torch.tanh(dl[i, :3]) * max_rot))))
        dC.append((C1 - C0).numpy() * 1000)
    dC = np.array(dC)
    res["l4_camera_refine"] = dict(note="rows = sorted camera ids (dataset cam_index); bounds from cfg.model.l4",
                                   max_rot_rad=max_rot, max_trans_m=max_trans, rot_deg_mean=round(float(np.mean(rot_deg)), 4),
                                   rot_deg_max=round(float(np.max(rot_deg)), 4), centre_shift_mm_mean=dC.mean(0).round(3).tolist(),
                                   centre_shift_mm_abs_mean=round(float(np.linalg.norm(dC, axis=1).mean()), 3),
                                   centre_shift_mm_norm_of_mean=round(float(np.linalg.norm(dC.mean(0))), 3))
    (out / "edge_diag.json").write_text(json.dumps(res, indent=1))
    for lab in vis:
        if vis[lab]:
            cv2.imwrite(str(out / f"rim_{lab}.jpg"), cv2.resize(np.concatenate(vis[lab], 1), None, fx=0.5, fy=0.5))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
