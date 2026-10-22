"""Experiment-card figures for the "SR4D 실험 기록" artifact (D-040 experiments).

One JPEG per (camera): rows = frames, columns = GT / each setting. Each cell = full image (half res) + hair / skirt crops,
|error|x3 row below, fg PSNR in the corner. --share adds, for the given settings, a column with the inner layer's share
of the accumulated alpha (<frame>.inner.png from sr4d.layered.render; orange = outer, blue = inner, white = empty).
  PYTHONPATH=. python tools/exp_figure.py --root <data> --cams g12 g15 --frames 0 30 59 --out <dir> --prefix held \
      --methods "label=<renders dir>" ... [--share "label=<renders dir>" ...]
renders dir layout: <dir>/<cam>/<frame:06d>.png
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors
from tools.cmp_figure import CS, FH, FW, cut, errmap, label_bar, put

CROPS = 2


def cell(img, gt, boxes, m):
    crops = [cv2.resize(cut(img, *b), (CS, CS), interpolation=cv2.INTER_AREA) for b in boxes]
    pad = np.full((FH - CROPS * CS, CS, 3), 255, np.uint8)
    top = np.concatenate([cv2.resize(img, (FW, FH), interpolation=cv2.INTER_AREA), np.concatenate(crops + [pad], 0)], 1)
    if gt is None:
        return np.concatenate([top, np.full_like(top, 255)], 0)
    e2 = ((img.astype(np.float32) - gt) / 255) ** 2
    top = put(top, f"{-10 * np.log10(max(e2.mean(-1)[m].mean(), 1e-10)):.2f} dB")
    ec = [cv2.resize(errmap(cut(img, *b), cut(gt, *b)), (CS, CS), interpolation=cv2.INTER_AREA) for b in boxes]
    bot = np.concatenate([cv2.resize(errmap(img, gt), (FW, FH), interpolation=cv2.INTER_AREA), np.concatenate(ec + [pad], 0)], 1)
    return np.concatenate([top, bot], 0)


def share_cell(sh, alpha, boxes):
    """sh, alpha in [0,1] (H,W). colour: blend(outer orange, inner blue) by share, faded by alpha onto white."""
    ora, blue = np.array([240, 130, 20], np.float32), np.array([40, 90, 230], np.float32)
    c = (1 - sh[..., None]) * ora + sh[..., None] * blue
    img = (alpha[..., None] * c + (1 - alpha[..., None]) * 255).astype(np.uint8)
    frac = float((sh * alpha).sum() / max(alpha.sum(), 1e-6))
    return cell(img, None, boxes, None)[:FH].copy(), frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--cams", nargs="+", required=True)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 30, 59]); ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="held"); ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--share", nargs="*", default=[])
    a = ap.parse_args()
    R = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meths = [m.split("=", 1) for m in a.methods]; shares = [m.split("=", 1) for m in a.share]
    cams = io.load_cameras(R); meta = io.load_meta(R)
    bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))

    def boxes(c, f):
        b, p, t, _ = params_to_tensors([io.load_body(R, f)])
        with torch.no_grad():
            J = bm(b, p, t)[1][0].numpy()
        Rm, tt, K = np.array(cams[c]["R"]), np.array(cams[c]["t"]).reshape(3), np.array(cams[c]["K"])
        uv = lambda j: (lambda x: x[:2] / x[2])(K @ (Rm @ J[j] + tt))  # noqa: E731
        hl = max(np.linalg.norm(uv(15) - uv(12)), 8); h = uv(15)
        hip = uv(0); kn = (uv(4) + uv(5)) / 2; sk = (hip + kn) / 2
        return [(h[0], h[1] - 0.3 * hl, int(2.8 * hl)), (sk[0], sk[1], int(max(1.6 * np.linalg.norm(kn - hip), 40)))]

    names = ["생성 GT"] + [n for n, _ in meths] + [f"층 기여: {n}" for n, _ in shares]
    widths = [FW + CS] * len(names)
    stats = {}
    files = []
    for c in a.cams:
        rows = [label_bar(names, widths)]
        for f in a.frames:
            img = io.read_image(R / f"images/{c}/{f:06d}.jpg"); m = io.read_mask(R / f"masks/{c}/{f:06d}.png") > 127
            gt = np.where(m[..., None], img, 255).astype(np.uint8); bx = boxes(c, f)
            row = [cell(gt, None, bx, m)]
            for _, d in meths:
                p = Path(d) / c / f"{f:06d}.png"
                row.append(cell(io.read_image(p), gt, bx, m) if p.exists() else np.full((2 * FH, FW + CS, 3), 160, np.uint8))
            for n, d in shares:
                p = Path(d) / c / f"{f:06d}.inner.png"; pa = Path(d) / c / f"{f:06d}.alpha.png"
                if not p.exists():
                    row.append(np.full((2 * FH, FW + CS, 3), 160, np.uint8)); continue
                sh = cv2.imread(str(p), 0) / 255.0; al = cv2.imread(str(pa), 0) / 255.0
                top, frac = share_cell(sh, al, bx)
                stats.setdefault(n, []).append(frac)
                # inner share inside / outside the GT mask
                lab = np.full((FH, FW + CS, 3), 255, np.uint8)
                lab = put(lab, f"안쪽층 비율 {frac * 100:.0f}%  (주황=바깥, 파랑=안쪽)", 15)
                row.append(np.concatenate([put(top, f"{frac * 100:.0f}%"), lab], 0))
            rows.append(put(np.concatenate(row, 1), f"{c} · frame {f}", 18))
        name = f"{a.prefix}_{c}.jpg"
        cv2.imwrite(str(out / name), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 78])
        files.append(name)
    idx = out / "index.json"
    I = json.loads(idx.read_text()) if idx.exists() else {}
    I[a.prefix] = files
    I.setdefault("share_stats", {})[a.prefix] = {n: float(np.mean(v)) for n, v in stats.items()}
    idx.write_text(json.dumps(I, ensure_ascii=False, indent=1))
    print(json.dumps(I, ensure_ascii=False))


if __name__ == "__main__":
    main()
