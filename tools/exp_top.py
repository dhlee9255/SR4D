"""Top figure of the "SR4D 실험 기록" artifact: rows = view groups (near / far), columns = GT + methods, one frame.
Each cell = image + hair / skirt crops, |error|x3 below (tools/exp_figure.cell). Under each row: layer-share strip for the
methods given with ":share" (orange = outer, blue = inner), blank for the rest.
  PYTHONPATH=. python tools/exp_top.py --out top.jpg --frame 30 \
     --row "가까운 시점 · gen48_ho8 홀드아웃 g12|<data root>|g12|SR4D=<dir>|New=<dir>:share|..." --row ...
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors
from tools.cmp_figure import CS, FH, FW, label_bar, put
from tools.exp_figure import cell, share_cell


def boxes(R, cams, bm, c, f):
    b, p, t, _ = params_to_tensors([io.load_body(R, f)])
    with torch.no_grad():
        J = bm(b, p, t)[1][0].numpy()
    Rm, tt, K = np.array(cams[c]["R"]), np.array(cams[c]["t"]).reshape(3), np.array(cams[c]["K"])
    uv = lambda j: (lambda x: x[:2] / x[2])(K @ (Rm @ J[j] + tt))  # noqa: E731
    hl = max(np.linalg.norm(uv(15) - uv(12)), 8); h = uv(15)
    hip = uv(0); kn = (uv(4) + uv(5)) / 2; sk = (hip + kn) / 2
    return [(h[0], h[1] - 0.3 * hl, int(2.8 * hl)), (sk[0], sk[1], int(max(1.6 * np.linalg.norm(kn - hip), 40)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--frame", type=int, default=30)
    ap.add_argument("--row", action="append", required=True)
    a = ap.parse_args()
    blocks = []
    for spec in a.row:
        title, root, c, *ms = spec.split("|")
        R = Path(root); cams = io.load_cameras(R); meta = io.load_meta(R)
        bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
        f = a.frame; bx = boxes(R, cams, bm, c, f)
        img = io.read_image(R / f"images/{c}/{f:06d}.jpg"); m = io.read_mask(R / f"masks/{c}/{f:06d}.png") > 127
        gt = np.where(m[..., None], img, 255).astype(np.uint8)
        names, ims, shares = ["생성 GT"], [cell(gt, None, bx, m)], [np.full((FH, FW + CS, 3), 255, np.uint8)]
        for spec_m in ms:
            name, d = spec_m.split("=", 1)
            want_share = d.endswith(":share"); d = Path(d[:-6] if want_share else d)
            p = d / c / f"{f:06d}.png"
            names.append(name)
            ims.append(cell(io.read_image(p), gt, bx, m) if p.exists() else np.full((2 * FH, FW + CS, 3), 160, np.uint8))
            s = np.full((FH, FW + CS, 3), 255, np.uint8)
            if want_share and (d / c / f"{f:06d}.inner.png").exists():
                sh = cv2.imread(str(d / c / f"{f:06d}.inner.png"), 0) / 255.0; al = cv2.imread(str(d / c / f"{f:06d}.alpha.png"), 0) / 255.0
                top, frac = share_cell(sh, al, bx)
                s = put(top, f"안쪽층 {frac * 100:.0f}%")
            shares.append(s)
        w = [FW + CS] * len(names)
        head = put(np.full((30, sum(w), 3), 255, np.uint8), f"{title} · frame {f}", 20)
        sbar = put(np.full((26, sum(w), 3), 245, np.uint8), "층 기여 지도 (주황 = 바깥층, 파랑 = 안쪽층, 흐릴수록 알파 낮음)", 17)
        blocks.append(np.concatenate([head, label_bar(names, w), np.concatenate(ims, 1), sbar, np.concatenate(shares, 1)], 0))
    W = max(b.shape[1] for b in blocks)
    blocks = [np.pad(b, ((0, 12), (0, W - b.shape[1]), (0, 0)), constant_values=255) for b in blocks]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(blocks, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("top", out)


if __name__ == "__main__":
    main()
