"""Diagnostic 1 (user 2026-10-12): drop only L2's position offset dx at render time (keep dq, dlog_s, dcolor) and report
the RELATIVE change in head/hair and skirt regions vs the torso, on training views against generated GT.
Regions from projected input SMPL-X joints; eroded mask (3 px). Reads <run>/diag_train_full and diag_train_nol2dx.
Rule (user): removal that changes nothing may be concluded; a drop is only reported relative to the torso, not as a cause.
usage: PYTHONPATH=. python tools/diag_l2dx_regions.py --root <data> --run <run_dir> [--json out]
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors


def regions(uv, H, W):
    yy, xx = np.mgrid[:H, :W]
    hl = np.linalg.norm(uv[15] - uv[12])
    hipy = (uv[1, 1] + uv[2, 1]) / 2
    kny = min(uv[4, 1], uv[5, 1], H - 1)
    sw = abs(uv[16, 0] - uv[17, 0]) * 0.35 + 1
    cx = (uv[3, 0] + uv[9, 0]) / 2
    return {"head": (np.abs(xx - uv[15, 0]) < 2.2 * hl) & (yy < uv[15, 1] + 1.0 * hl) & (yy > uv[15, 1] - 3.0 * hl),
            "skirt": (yy > hipy) & (yy < hipy + 0.8 * (kny - hipy)),
            "torso": (np.abs(xx - cx) < sw) & (yy > uv[9, 1]) & (yy < uv[3, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--run", required=True); ap.add_argument("--json", default=None)
    ap.add_argument("--fig", default=None, help="save an example: GT / full / no-dx / |diff| x4 for one view+frame")
    a = ap.parse_args()
    R, D = Path(a.root), Path(a.run)
    sp = io.load_split(R, "r15h35"); cams = io.load_cameras(R); meta = io.load_meta(R)
    bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
    acc = {r: dict(se_full=0., se_nodx=0., diff=0., n=0) for r in ("head", "skirt", "torso")}
    for f in sp["frames"][::3]:
        b, p, t, _ = params_to_tensors([io.load_body(R, f)])
        with torch.no_grad():
            _, J, _ = bm(b, p, t)
        J = J[0].numpy()
        for c in sp["train_cams"][::2]:
            Rm, tv, K = np.array(cams[c]["R"]), np.array(cams[c]["t"]).reshape(3), np.array(cams[c]["K"])
            x = (K @ (Rm @ J.T + tv[:, None])).T; uv = x[:, :2] / x[:, 2:]
            img = io.read_image(R / f"images/{c}/{f:06d}.jpg").astype(np.float32) / 255
            m = io.read_mask(R / f"masks/{c}/{f:06d}.png") > 127
            gt = np.where(m[..., None], img, 1.0); me = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            A = io.read_image(D / f"diag_train_full/{c}/{f:06d}.png").astype(np.float32) / 255
            B = io.read_image(D / f"diag_train_nol2dx/{c}/{f:06d}.png").astype(np.float32) / 255
            for r, M in regions(uv, *m.shape).items():
                M = M & me
                if M.sum() < 50:
                    continue
                q = acc[r]
                q["se_full"] += ((A - gt) ** 2).mean(-1)[M].sum(); q["se_nodx"] += ((B - gt) ** 2).mean(-1)[M].sum()
                q["diff"] += np.abs(A - B).mean(-1)[M].sum(); q["n"] += int(M.sum())
            if a.fig and f == 30 and c == sp["train_cams"][0]:
                d = np.clip(np.abs(A - B) * 4, 0, 1)
                cv2.imwrite(a.fig, cv2.cvtColor((np.concatenate([gt, A, B, d], 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    res = {}
    for r, q in acc.items():
        pf, pn = -10 * np.log10(q["se_full"] / q["n"]), -10 * np.log10(q["se_nodx"] / q["n"])
        res[r] = dict(psnr_full=round(pf, 2), psnr_nol2dx=round(pn, 2), drop=round(pf - pn, 2),
                      mean_abs_change_8bit=round(q["diff"] / q["n"] * 255, 2), px=q["n"])
    for r in ("head", "skirt"):
        res[r]["drop_minus_torso"] = round(res[r]["drop"] - res["torso"]["drop"], 2)
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
