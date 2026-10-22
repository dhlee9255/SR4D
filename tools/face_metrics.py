"""Face-region metrics (user 2026-10-13): crop a square around the SMPL-X head (joint 15, size = 2.6 x head-neck
distance in pixels, clipped to the image), composite GT on white with its mask, and compare renders per frame:
PSNR (mask-weighted inside the crop) and LPIPS(VGG) of the crop resized to 128x128.
Writes <run>/metrics_face_<renders>.json: {split: {face_psnr, face_lpips, per_frame: {frame: [psnr, lpips]}}}.
usage: PYTHONPATH=. python tools/face_metrics.py --data_root <data> --setting <s> --renders <run>/renders [--cams ...]
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

_BM = {}


def head_boxes(root: Path, frames, cams_needed):
    meta = io.load_meta(root)
    key = (meta["body_model"], meta.get("gender", "neutral"))
    if key not in _BM:
        _BM[key] = BodyModel(*key)
    bm = _BM[key]; cams = io.load_cameras(root); out = {}
    for f in frames:
        b, p, t, _ = params_to_tensors([io.load_body(root, f)])
        with torch.no_grad():
            _, J, _ = bm(b, p, t)
        J = J[0].numpy()
        for c in cams_needed:
            R, tt, K = np.array(cams[c]["R"]), np.array(cams[c]["t"]).reshape(3), np.array(cams[c]["K"])
            x = (K @ (R @ J[[12, 15]].T + tt[:, None])).T; uv = x[:, :2] / x[:, 2:]
            s = max(16, int(2.6 * np.linalg.norm(uv[1] - uv[0])))
            cx, cy = uv[1]
            out[(c, f)] = (int(cx - s / 2), int(cy - s * 0.6), s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True); ap.add_argument("--setting", required=True); ap.add_argument("--renders", required=True)
    ap.add_argument("--cams", nargs="*", default=None)
    a = ap.parse_args()
    root, rd = Path(a.data_root), Path(a.renders)
    sp = io.load_split(root, a.setting)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    fn = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval()
    res = {}
    for sname in ("test_interp", "test_extrap"):
        cams = a.cams or sp.get(f"{sname}_cams", [])
        if not cams:
            continue
        boxes = head_boxes(root, sp["frames"], cams)
        per = {}
        for f in sp["frames"]:
            ps, ls = [], []
            for c in cams:
                x0, y0, s = boxes[(c, f)]
                img = io.read_image(root / f"images/{c}/{f:06d}.jpg").astype(np.float32) / 255
                m = io.read_mask(root / f"masks/{c}/{f:06d}.png") > 127
                H, W = m.shape
                xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
                if xb - xa < 8 or yb - ya < 8:
                    continue
                gt = np.where(m[..., None], img, 1.0)[ya:yb, xa:xb]
                r = io.read_image(rd / c / f"{f:06d}.png").astype(np.float32)[ya:yb, xa:xb] / 255
                mc = m[ya:yb, xa:xb]
                if mc.sum() < 20:
                    continue
                ps.append(float(-10 * np.log10(max(((gt - r) ** 2).mean(-1)[mc].mean(), 1e-10))))
                t = lambda z: torch.tensor(cv2.resize(z, (128, 128), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].to(DEV) * 2 - 1  # noqa: E731
                with torch.no_grad():
                    ls.append(float(fn(t(r), t(gt))))
            if ps:
                per[f] = [float(np.mean(ps)), float(np.mean(ls))]
        res[sname] = dict(face_psnr=float(np.mean([v[0] for v in per.values()])), face_lpips=float(np.mean([v[1] for v in per.values()])),
                          cams=cams, per_frame=per)
    tag = "" if a.cams is None else "_cams"
    (rd.parent / f"metrics_face{tag}_{rd.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("face_psnr", "face_lpips")} for k, v in res.items()}))


if __name__ == "__main__":
    main()
