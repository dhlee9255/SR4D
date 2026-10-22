"""Crop comparison (dance): face+hair and polka-dot crops, 5 frames x (seen views g01, g11 + held-out g00, g12),
for any list of runs. Per region/group: masked PSNR, LPIPS(VGG, crop resized 128) and sharpness (Laplacian variance;
counts noise as sharp). Seen views read <run>/renders_train (tools/render_some.py), held-out views <run>/renders.
usage: PYTHONPATH=. python tools/crop_compare.py --out DIR --run "LABEL=<run dir>" ... [--json out.json]
"""
import argparse
import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch

from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors

R = Path("data_dev/fouranyone_gen/dance5885633_gen48_2ring")
FRAMES = [0, 15, 30, 45, 59]
GROUPS = {"seen": ["g01", "g11"], "held": ["g00", "g12"]}
CELL = 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--run", nargs="+", required=True); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = [r.split("=", 1) for r in a.run]
    cams = io.load_cameras(R); meta = io.load_meta(R); bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
    LP = lpips.LPIPS(net="vgg", verbose=False).eval()
    boxes = {}
    for f in FRAMES:
        b, p, t, _ = params_to_tensors([io.load_body(R, f)])
        with torch.no_grad():
            J = bm(b, p, t)[1][0].numpy()
        for c in sum(GROUPS.values(), []):
            Rm, tt, K = np.array(cams[c]["R"]), np.array(cams[c]["t"]).reshape(3), np.array(cams[c]["K"])
            uv = lambda j: (lambda x: x[:2] / x[2])(K @ (Rm @ J[j] + tt))  # noqa: E731
            hl = np.linalg.norm(uv(15) - uv(12)); h = uv(15)
            boxes[(c, f)] = {"face+hair": (h[0], h[1] - 0.3 * hl, int(2.8 * hl)), "dots": (*uv(3), 72)}

    def cut(img, cx, cy, s):
        H, W = img.shape[:2]; x0, y0 = int(cx - s / 2), int(cy - s / 2)
        pad = np.full((s, s) + img.shape[2:], 255 if img.ndim == 3 else 0, img.dtype)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
        pad[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
        return pad

    def sharp(img, m):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32); lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        me = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return float(lap[me].var()) if me.sum() > 20 else float("nan")

    res = {}; tiles = {g: [] for g in GROUPS}
    for grp, views in GROUPS.items():
        for c in views:
            for f in FRAMES:
                img = io.read_image(R / f"images/{c}/{f:06d}.jpg"); m = io.read_mask(R / f"masks/{c}/{f:06d}.png") > 127
                gtf = np.where(m[..., None], img, 255).astype(np.uint8)
                rr = []
                for lab, d in runs:
                    sub = "renders_train" if grp == "seen" else "renders"
                    p = Path(d) / sub / c / f"{f:06d}.png"
                    rr.append(io.read_image(p) if p.exists() else None)
                for reg, (cx, cy, s) in boxes[(c, f)].items():
                    gt = cut(gtf, cx, cy, s); mc = cut(m.astype(np.uint8), cx, cy, s) > 0
                    res.setdefault(grp, {}).setdefault(reg, {}).setdefault("GT", {"sharp": []})["sharp"].append(sharp(gt, mc))
                    row = [cv2.resize(gt, (CELL, CELL), interpolation=cv2.INTER_NEAREST)]
                    for (lab, _), r in zip(runs, rr):
                        if r is None:
                            row.append(np.full((CELL, CELL, 3), 128, np.uint8)); continue
                        rc = cut(r, cx, cy, s); e2 = ((rc.astype(np.float32) - gt) / 255) ** 2
                        tt = lambda z: torch.tensor(cv2.resize(z, (128, 128), interpolation=cv2.INTER_CUBIC)).float().permute(2, 0, 1)[None] / 127.5 - 1  # noqa: E731
                        with torch.no_grad():
                            lp = float(LP(tt(rc), tt(gt)))
                        d = res[grp][reg].setdefault(lab, {"psnr": [], "lpips": [], "sharp": []})
                        d["psnr"].append(float(-10 * np.log10(max(e2.mean(-1)[mc].mean(), 1e-10)))); d["lpips"].append(lp); d["sharp"].append(sharp(rc, mc))
                        row.append(cv2.resize(rc, (CELL, CELL), interpolation=cv2.INTER_NEAREST))
                    if f in (0, 30):
                        tiles[grp].append(np.concatenate(row, 1))
    summ = {g: {reg: {lab: {k: round(float(np.nanmean(v)), 3) for k, v in d.items()} for lab, d in R_.items()} for reg, R_ in G_.items()} for g, G_ in res.items()}
    for g in GROUPS:
        cv2.imwrite(str(out / f"crops_{g}.jpg"), cv2.cvtColor(np.concatenate(tiles[g], 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if a.json:
        Path(a.json).write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
