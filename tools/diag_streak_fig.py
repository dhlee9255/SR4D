"""Montage for tools/diag_streak.py output: per view, crown crop of GT / New-fade / inner only / outer only (fade, black bg)
/ outer only (no fade, black bg) / streak pixels / FTGS++ official black & white bg (composited on white) [+ extra dirs].
  PYTHONPATH=. python tools/diag_streak_fig.py --diag <dir> --root <data> --out x.jpg --extra "Name=<render dir>" ...
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from sr4d.data import io
from sr4d.model.body_model import BodyModel
from tools.cmp_figure import label_bar, put
from tools.crown_crop import box, crop

S = 230
ap = argparse.ArgumentParser()
ap.add_argument("--diag", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--scale", type=float, default=1.6)
a = ap.parse_args()
D = Path(a.diag); R = Path(a.root); cams = io.load_cameras(R); meta = io.load_meta(R); bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
rows = []
names = ["GT", "New-fade", "안쪽층만", "바깥층만 (fade, 검정 배경)", "바깥층만 (fade 없음, 검정)", "흰 줄 픽셀"] + [e.split("=", 1)[0] for e in a.extra]
for gtp in sorted(D.glob("*_gt.png")):
    cid, ft = gtp.name.split("_")[:2]; f = int(ft[1:])
    cx, cy, s = box(R, cams[cid], bm, f); s = int(s * a.scale)
    ims = [io.read_image(D / f"{cid}_{ft}_{k}.png") for k in ("gt", "newfade", "inner_only", "outer_only_fade_black", "outer_only_nofade_black")]
    sm = cv2.imread(str(D / f"{cid}_{ft}_streakmask.png"), 0); ov = ims[1].copy(); ov[sm > 127] = (255, 0, 255); ims.append(ov)
    for e in a.extra:
        p = Path(e.split("=", 1)[1]) / cid / f"{f:06d}.png"
        ims.append(io.read_image(p) if p.exists() else np.full_like(ims[0], 160))
    cells = [cv2.resize(crop(im, cx, cy, s), (S, S), interpolation=cv2.INTER_AREA) for im in ims]
    rows.append(put(np.full((26, S * len(cells), 3), 255, np.uint8), f"{cid} frame {f} · 정수리 확대 (자홍 = 흰 줄 픽셀: GT는 어두운 머리카락인데 렌더가 밝음)", 16))
    rows.append(np.concatenate(cells, 1))
out = np.concatenate([label_bar(names, [S] * len(names))] + rows, 0)
cv2.imwrite(a.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(a.out)
