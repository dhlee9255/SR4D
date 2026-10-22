"""Verify a unified dataset: project the body mesh into every camera and compute IoU with the masks.

Low IoU => camera / coordinate / unit error (CLAUDE.md 5.2). Exits with code 1 if mean IoU < --min_iou.

Usage: python tools/verify_dataset.py --root data_dev/zju_mocap/387 [--frames 0 30 59]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402
from sr4d.model.body_model import BodyModel, params_to_tensors  # noqa: E402


def mesh_silhouette(verts_w: np.ndarray, faces: np.ndarray, cam: dict) -> np.ndarray:
    pc = verts_w @ cam["R"].T + cam["t"]
    z = pc[:, 2]
    uv = pc[:, :2] / np.clip(z[:, None], 1e-6, None)
    uv = uv @ cam["K"][:2, :2].T + cam["K"][:2, 2]
    sil = np.zeros((cam["H"], cam["W"]), np.uint8)
    ok = (z[faces] > 0.01).all(1)
    tris = np.round(uv[faces[ok]] * 4).astype(np.int32)  # 2 bits of subpixel precision
    for tri in tris:
        cv2.fillConvexPoly(sil, tri, 255, lineType=cv2.LINE_8, shift=2)
    return sil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--min_iou", type=float, default=0.7)
    args = ap.parse_args()
    root = Path(args.root)
    meta = io.load_meta(root)
    cams = io.load_cameras(root)
    frames = args.frames if args.frames else [meta["frames"][0], meta["frames"][len(meta["frames"]) // 2], meta["frames"][-1]]
    bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
    faces = bm.faces.numpy()
    params = [io.load_body(root, f) for f in frames]
    betas, pose, transl, expr = params_to_tensors(params)
    with torch.no_grad():
        verts, _, _ = bm(betas, pose, transl, expr)
    verts = verts.numpy()

    out_dir = root / "verify"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for fi, f in enumerate(frames):
        for cid, cam in cams.items():
            mpath = root / "masks" / cid / f"{f:06d}.png"
            if not mpath.exists():
                continue
            m = io.read_mask(mpath) > 127
            s = mesh_silhouette(verts[fi], faces, cam) > 0
            iou = (m & s).sum() / max((m | s).sum(), 1)
            rows.append((f, cid, float(iou)))
        # overlay for the first camera
        cid0 = sorted(cams)[0]
        img = io.read_image(root / "images" / cid0 / f"{f:06d}.jpg")
        s = mesh_silhouette(verts[fi], faces, cams[cid0])
        ov = img.copy()
        ov[s > 0] = (0.5 * ov[s > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        io.write_image(out_dir / f"overlay_f{f:06d}_c{cid0}.jpg", ov)

    ious = np.array([r[2] for r in rows])
    per_cam = {}
    for f, c, i in rows:
        per_cam.setdefault(c, []).append(i)
    report = dict(mean_iou=float(ious.mean()), min_iou=float(ious.min()),
                  per_cam={c: float(np.mean(v)) for c, v in sorted(per_cam.items())}, frames=frames)
    (out_dir / "iou_report.json").write_text(json.dumps(report, indent=1))
    print(f"mean IoU {report['mean_iou']:.3f} (min {report['min_iou']:.3f}) over {len(rows)} (frame,cam) pairs")
    worst = sorted(report["per_cam"].items(), key=lambda kv: kv[1])[:3]
    print("worst cams:", worst)
    if report["mean_iou"] < args.min_iou:
        print("FAIL: IoU too low -> check camera convention (w2c/c2w, OpenCV/OpenGL), units, BGR/RGB")
        sys.exit(1)


if __name__ == "__main__":
    main()
