#!/usr/bin/env bash
# Item 4: convert a 2-ring (15/35 deg) 4DAnyone run into the SR4D format, generated-only, and add split
# "r15h35": train = 15-deg ring (g00-g23), test_extrap = 35-deg ring (g24-g47, held out). Also gen48 (all, self-eval).
# usage: prep_2ring.sh <run_data_dir> <clip> <out_root> <fps> <gpu>
set -euo pipefail
cd "$(dirname "$0")/.."
D=$(realpath "$1"); C=$2; OUT=$3; FPS=$4; G=$5
source ~/miniconda3/etc/profile.d/conda.sh; conda activate 4danyone
CUDA_VISIBLE_DEVICES=$G python tools/fouranyone_prep.py --fdanyone_dir $D/fdanyone/$C --motion_dir $D/gvhmr/results/$C \
  --out $OUT --frames 0 60 --downscale 2
conda activate sr4d
python tools/convert_4danyone.py --gen_root $OUT --fps $FPS
python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); sp = json.load(open(next((root / "splits").glob("gen*.json"))))
cams = sorted(sp["train_cams"]); assert len(cams) == 48, len(cams)
tr, ho = cams[:24], cams[24:]
import numpy as np
cj = json.load(open(root / "cameras.json"))
def ctr(c):  # camera centre from world->cam R, t
    R, t = np.array(cj[c]["R"]), np.array(cj[c]["t"]).reshape(3)
    return -R.T @ t
cen = np.mean([ctr(c) for c in cams], 0)
def ang(a, b):
    u, v = ctr(a) - cen, ctr(b) - cen
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
near = {}
for h in ho:
    best = min(tr, key=lambda c: ang(h, c)); near[h] = dict(cam=best, angle_deg=round(ang(h, best), 2))
out = dict(sp, setting="r15h35", train_cams=tr, test_interp_cams=[], test_extrap_cams=ho, nearest_train=near,
           note="train 15-deg ring, held-out 35-deg ring (generated GT, item 4)")
print("nearest-train angles:", sorted({v["angle_deg"] for v in near.values()}))
json.dump(out, open(root / "splits/r15h35.json", "w"), indent=1)
print("split r15h35:", len(tr), "train /", len(ho), "held-out")
EOF
