"""2D whole-body keypoints (RTMW-x, rtmlib 'performance' mode; COCO-WholeBody 133, first 17 = COCO body) for the
cameras of a unified dataset. Run with the pose_wb env python (rtmlib + onnxruntime).

  <pose_wb python> tools/detect_keypoints_rtmw.py --root data_dev/zju_mocap/387 --cams 01 05 06 07 08 09 16
writes <root>/keypoints_rtmw/<cam>.npz : kpts (T,133,2) px, scores (T,133), frames (T,)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Wholebody


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cams", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    root = Path(a.root)
    frames = json.loads((root / "meta.json").read_text())["frames"]
    model = Wholebody(mode="performance", backend="onnxruntime", device=a.device)
    out = root / "keypoints_rtmw"
    out.mkdir(exist_ok=True)
    for cid in a.cams:
        K, S = [], []
        for f in frames:
            img = cv2.imread(str(root / "images" / cid / f"{f:06d}.jpg"))
            m = cv2.imread(str(root / "masks" / cid / f"{f:06d}.png"), 0)
            kp, sc = model(img)
            if len(kp) > 1 and m is not None:  # pick the person whose keypoints fall most inside the mask
                ins = []
                for p in kp:
                    u = np.clip(p[:17, 0].astype(int), 0, m.shape[1] - 1)
                    v = np.clip(p[:17, 1].astype(int), 0, m.shape[0] - 1)
                    ins.append((m[v, u] > 127).mean())
                i = int(np.argmax(ins))
            else:
                i = 0
            if len(kp) == 0:
                K.append(np.zeros((133, 2))), S.append(np.zeros(133))
            else:
                K.append(kp[i]), S.append(sc[i])
        np.savez(out / f"{cid}.npz", kpts=np.asarray(K, np.float32), scores=np.asarray(S, np.float32), frames=np.asarray(frames))
        print(cid, "mean body score", float(np.mean(np.asarray(S)[:, :17])), flush=True)


if __name__ == "__main__":
    main()
