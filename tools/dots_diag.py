"""Polka-dot blur diagnosis, step 1 (user 2026-10-13), CPU only, existing results.
Fixed crops: torso/skirt patch (centre = SMPL-X spine1, 72 px) at 5 frames x (2 views seen in training + 2 held-out).
  seen views     : g01, g11 from the gen48_all runs (all 48 views trained; ho8 runs only rendered the held-out views)
  held-out views : g00, g12 from the gen48_ho8 runs
Per crop: masked PSNR vs generated GT, LPIPS(VGG, crop resized to 128) and sharpness = variance of the Laplacian (grey, inside the mask; GT given too).
Also: Gaussians on torso+hips per method, SR4D train-loss slope over the last 10 %, L4 confidence maps.
usage: PYTHONPATH=. python tools/dots_diag.py --out <dir> [--extra LABEL=<renders dir> ...]
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors

R = Path("data_dev/fouranyone_gen/dance5885633_gen48_2ring")
FRAMES = [0, 15, 30, 45, 59]
SEEN, HELD = ["g01", "g11"], ["g00", "g12"]
RUNS = {"SR4D": "sr4d_final", "DualGS": "dualgs", "FTGS++": "ftgspp_black_official"}
TORSO = [0, 1, 2, 3, 6, 9, 13, 14]  # pelvis, hips, spine1-3, collars (dress body + skirt top)
S = 72


def crop_centres(bm, cams):
    out = {}
    for f in FRAMES:
        b, p, t, _ = params_to_tensors([io.load_body(R, f)])
        with torch.no_grad():
            _, J, _ = bm(b, p, t)
        J = J[0].numpy()
        for c in SEEN + HELD:
            Rm, tt, K = np.array(cams[c]["R"]), np.array(cams[c]["t"]).reshape(3), np.array(cams[c]["K"])
            x = K @ (Rm @ J[3] + tt)
            out[(c, f)] = (int(x[0] / x[2] - S / 2), int(x[1] / x[2] - S / 2))
    return out


def cut(img, x0, y0):
    H, W = img.shape[:2]
    pad = np.full((S, S) + img.shape[2:], 255, img.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + S, W), min(y0 + S, H)
    pad[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return pad


def sharp(img, m):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    me = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return float(lap[me].var()) if me.sum() > 20 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="LABEL=<renders dir with held-out g00/g12> (variants)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    global LP
    import lpips
    LP = lpips.LPIPS(net="vgg", verbose=False).eval()
    cams = io.load_cameras(R); meta = io.load_meta(R)
    bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
    C = crop_centres(bm, cams)
    S_ = io.load_split(R, "gen48_ho8")
    assert not set(HELD) & set(S_["train_cams"]) and set(SEEN) <= set(S_["train_cams"])
    src = {lab: {"seen": Path(f"outputs/{d}/dance_gen48_all/renders"), "held": Path(f"outputs/{d}/dance_gen48_ho8/renders")} for lab, d in RUNS.items()}
    for e in a.extra:
        lab, d = e.split("=", 1); src[lab] = {"held": Path(d)}
    res = {"crop": {}, "gaussians_torso": {}, "loss_slope": {}}
    rows = {"seen": [], "held": []}
    for grp, views in (("seen", SEEN), ("held", HELD)):
        acc = {lab: {"psnr": [], "lpips": [], "sharp": []} for lab in src if grp in src[lab]}; acc["GT"] = {"sharp": []}
        for c in views:
            for f in FRAMES:
                x0, y0 = C[(c, f)]
                img = io.read_image(R / f"images/{c}/{f:06d}.jpg"); m = io.read_mask(R / f"masks/{c}/{f:06d}.png") > 127
                gt = cut(np.where(m[..., None], img, 255).astype(np.uint8), x0, y0); mc = cut(m.astype(np.uint8), x0, y0) > 0
                acc["GT"]["sharp"].append(sharp(gt, mc)); cells = [gt]
                for lab in acc:
                    if lab == "GT":
                        continue
                    r = cut(io.read_image(src[lab][grp] / c / f"{f:06d}.png"), x0, y0)
                    e2 = ((r.astype(np.float32) - gt) / 255) ** 2
                    acc[lab]["psnr"].append(float(-10 * np.log10(max(e2.mean(-1)[mc].mean(), 1e-10))))
                    acc[lab]["sharp"].append(sharp(r, mc)); cells.append(r)
                    tt = lambda z: torch.tensor(cv2.resize(z, (128, 128), interpolation=cv2.INTER_CUBIC)).float().permute(2, 0, 1)[None] / 127.5 - 1  # noqa: E731
                    with torch.no_grad():
                        acc[lab]["lpips"].append(float(LP(tt(r), tt(gt))))
                if f in (0, 30):
                    rows[grp].append(np.concatenate([np.repeat(np.repeat(z, 3, 0), 3, 1) for z in cells], 1))
        res["crop"][grp] = {lab: {k: round(float(np.nanmean(v)), 3) for k, v in d.items()} for lab, d in acc.items()}
        cv2.imwrite(str(out / f"dots_{grp}.jpg"), cv2.cvtColor(np.concatenate(rows[grp], 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    # Gaussians on torso + hips (frame 30 for per-frame methods; SR4D by skinning joint in canonical space)
    b, p, t, _ = params_to_tensors([io.load_body(R, 30)])
    with torch.no_grad():
        V, _, _ = bm(b, p, t)
    V = V[0].numpy(); lbs = bm.lbs_weights.argmax(-1).cpu().numpy()
    st = torch.load("outputs/sr4d_final/dance_gen48_ho8/ckpt_final.pt", map_location="cpu", weights_only=False)
    tv = np.isin(lbs, TORSO)
    from scipy.spatial import cKDTree
    tree = cKDTree(V[tv])
    def near(P, op):
        d, _ = tree.query(P, distance_upper_bound=0.06)
        return int(((d < 0.06) & (op > 0.1)).sum())
    body, free = st["body"], st["free"]
    faces = bm.faces.cpu().numpy() if torch.is_tensor(bm.faces) else np.asarray(bm.faces)
    tri = body["buffers"]["tri"].numpy()
    bop = torch.sigmoid(body["params"]["opacity"][:, 0]).numpy(); fop = torch.sigmoid(free["params"]["opacity"][:, 0]).numpy()
    fv = free["buffers"]["lbs_vidx"].numpy()
    res["gaussians_torso"]["SR4D body surfels"] = int((np.isin(lbs[faces[tri][:, 0]], TORSO) & (bop > 0.1)).sum())
    res["gaussians_torso"]["SR4D free"] = int((np.isin(lbs[fv], TORSO) & (fop > 0.1)).sum())
    res["gaussians_torso"]["SR4D total (all regions)"] = int(len(bop) + len(fop))
    from plyfile import PlyData
    v = PlyData.read("outputs/dualgs/dance_gen48_ho8/model/ckt/point_cloud_30.ply")["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], -1); op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    res["gaussians_torso"]["DualGS (frame 30)"] = near(xyz, op)
    res["gaussians_torso"]["DualGS total"] = int(len(xyz))
    fz = np.load("outputs/ftgspp_black_official/dance_gen48_ho8/frame30_pts.npz")  # means_t / temporal opacity at frame 30 (FTGS++ env)
    res["gaussians_torso"]["FTGS++ (frame 30, temporal opacity)"] = near(fz["xyz"], fz["op"])
    res["gaussians_torso"]["FTGS++ total"] = int(len(fz["xyz"]))
    res["note_counts"] = "SR4D counted by skinning joint (canonical); DualGS/FTGS++ by distance < 6 cm to torso/hip SMPL-X vertices at frame 30; opacity > 0.1"
    # SR4D train loss over the last 10 %
    txt = Path("logs/g48_dance_gen48_ho8_sr4d.log").read_text()
    pts = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in re.finditer(r"^it (\d+) \[\w+\] loss ([\d.]+) psnr ([\d.]+)", txt, re.M)]
    last = [q for q in pts if q[0] >= 0.9 * pts[-1][0]]
    it, lo, ps = map(np.array, zip(*last))
    res["loss_slope"] = dict(range=[int(it[0]), int(it[-1])], loss_per_1k=float(np.polyfit(it / 1000, lo, 1)[0]), psnr_per_1k=float(np.polyfit(it / 1000, ps, 1)[0]),
                             loss_first_last=[float(lo[:3].mean()), float(lo[-3:].mean())], train_psnr_first_last=[float(ps[:3].mean()), float(ps[-3:].mean())])
    # L4 confidence maps (training views), crop box drawn
    cl = st["l4"].get("conf_logits")
    if cl is not None:
        allc = sorted(cams); tiles = []
        for c in SEEN:
            l = torch.nn.functional.interpolate(cl[allc.index(c):allc.index(c) + 1].float(), size=(640, 352), mode="bilinear", align_corners=False)[0, 0]
            w = torch.exp(l.clamp(-3, 3)).numpy()
            img = io.read_image(R / f"images/{c}/000030.jpg")
            hm = cv2.applyColorMap(np.clip(w / 3 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]
            ov = (0.5 * img + 0.5 * hm).astype(np.uint8); x0, y0 = C[(c, 30)]
            cv2.rectangle(ov, (x0, y0), (x0 + S, y0 + S), (255, 0, 0), 2); tiles.append(ov)
            res.setdefault("confidence_at_crop", {})[c] = round(float(w[max(y0, 0):y0 + S, max(x0, 0):x0 + S].mean()), 3)
            res.setdefault("confidence_mean", {})[c] = round(float(w.mean()), 3)
        cv2.imwrite(str(out / "confidence.jpg"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
    (out / "dots_diag.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
