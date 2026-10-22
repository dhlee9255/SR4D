"""Settings of the user's 2026-10-12 protocol change (r15h35 frozen as an appendix), on the existing 2-ring 48-view data:
  gen48_all : all 48 generated views train (= the old 'gen48' split; test = the same views, self-fit only)
  gen48_ho8 : hold out 4 views per ring at 6-view spacing (15-deg ring g00,g06,g12,g18; 35-deg ring staggered by 3:
              g27,g33,g39,g45 so no two held-out views share an azimuth), train on the other 40, score the 8 (test_interp).
usage: python tools/make_g48_splits.py <data_root> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

HO = ["g00", "g06", "g12", "g18", "g27", "g33", "g39", "g45"]
for root in map(Path, sys.argv[1:]):
    base = json.load(open(root / "splits/gen48.json"))
    cams = sorted(base["train_cams"]); assert len(cams) == 48
    cj = json.load(open(root / "cameras.json"))
    ctr = lambda c: -np.array(cj[c]["R"]).T @ np.array(cj[c]["t"]).reshape(3)
    cen = np.mean([ctr(c) for c in cams], 0)
    def ang(a, b):
        u, v = ctr(a) - cen, ctr(b) - cen
        return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
    allp = dict(base, setting="gen48_all", note="all 48 generated views train; test = same views (self-fit only, no held-out)")
    json.dump(allp, open(root / "splits/gen48_all.json", "w"), indent=1)
    tr = [c for c in cams if c not in HO]
    near = {h: dict(cam=min(tr, key=lambda c: ang(h, c)), angle_deg=round(min(ang(h, c) for c in tr), 2)) for h in HO}
    ho = dict(base, setting="gen48_ho8", train_cams=tr, test_interp_cams=HO, test_extrap_cams=[], nearest_train=near,
              note="40 train / 8 held-out generated views (4 per ring, 6-view spacing, 35-deg ring staggered by 3)")
    json.dump(ho, open(root / "splits/gen48_ho8.json", "w"), indent=1)
    print(root.name, "ho8 nearest-train angles", sorted({v["angle_deg"] for v in near.values()}))
