"""Projected SMPL-X face landmark vertices (nose, leye, reye, lear, rear) per frame/camera -> JSON (input of face_kp_check.py).
usage: PYTHONPATH=. python tools/face_landmarks.py --root <data> --setting <s> --frames ... --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors

FACE_V = [9120, 9448, 9929, 6, 616]
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--setting", required=True)
ap.add_argument("--frames", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
root = Path(a.root); sp = io.load_split(root, a.setting); cams = io.load_cameras(root); meta = io.load_meta(root)
bm = BodyModel(meta["body_model"], meta.get("gender", "neutral")); out = {}
for f in a.frames:
    b, p, t, _ = params_to_tensors([io.load_body(root, f)])
    with torch.no_grad():
        V = bm(b, p, t)[0][0].numpy()
    for c in sp["train_cams"]:
        R, tt, K = np.array(cams[c]["R"]), np.array(cams[c]["t"]).reshape(3), np.array(cams[c]["K"])
        x = (K @ (R @ V[FACE_V].T + tt[:, None])).T
        out[f"{f}/{c}"] = (x[:, :2] / x[:, 2:]).tolist()
Path(a.out).write_text(json.dumps(out))
print("wrote", len(out))
