"""Convert a 4dgs_benchmark manifest.json into FreeTimeGS++'s expected
"dynerf"-dataset raw layout, so its own extract/colmap/init/train/render
pipeline (ftgspp.data.extract -> ftgspp.data.prep -> ftgspp.init ->
ftgspp.train -> ftgspp.render) can run on our data directly.

Expected layout (see ftgspp/data/extract.py:symlink_image_dataset and
ftgspp/data/sfm.py:parse_poses_bounds_npy):
  <out>/images/<camera_index>/<frame_index>.png   (extract.py's fallback
                                                     path for datasets not
                                                     shipped as raw .mp4s)
  <out>/poses_bounds.npy   (LLFF format, (N_cameras, 17): 3x5 pose+hwf
                             flattened (15) + [near, far] (2), ONE ROW PER
                             CAMERA in the same 0..N-1 order as the
                             images/<camera_index> folders -- poses are
                             assumed static across time, matching every
                             other codebase touched this session)

Pose convention verified by direct round-trip against
ftgspp.data.sfm.parse_poses_bounds_npy's own array logic (reimplemented
inline below and checked to agree to float64 precision, not derived by
guesswork): that function computes
    w2c_recovered = inv(H).mT   where H = [reordered pose columns; 0 0 0 1]
and reordering is NEW=[OLD1,OLD0,-OLD2] (LLFF's classic [down,right,
backward] raw storage -> [right,down,forward] OpenCV convention). Working
backward from our own OpenCV-convention world_to_camera (the convention
used everywhere in this project, confirmed via camera_unit_test.py):
    H = inv(w2c)                      # OpenCV C2W: cols=[right,down,forward]
    raw_pose_cols = [H_down, H_right, -H_forward, H_translation]
round-trips to the exact original w2c (checked: max abs diff ~1e-16).

Camera intrinsics are SIMPLE_PINHOLE (single shared focal length) on the
recovery side, so this only works cleanly because our K already has
fx==fy (confirmed: fx=fy=1664.0 on this dataset) -- if fx!=fy for some
other dataset, this script's single `f` per camera would need revisiting.

Background handling: writes BOTH a masked (black-background composited,
matching every other codebase's established best practice this session)
and a raw/unmasked variant, since the user asked to try both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def build_poses_bounds(manifest: dict, camera_ids: list[int]) -> np.ndarray:
    """One row per camera, in `camera_ids` order. Near/far from camera
    depth (distance from world origin), same heuristic used for STGS/HUST
    this session: near=0.3 (generous close bound), far=2x the farthest
    camera distance."""
    cameras = manifest["cameras"]
    W, H = manifest["image_width"], manifest["image_height"]

    depths = []
    for cid in camera_ids:
        w2c = np.asarray(cameras[str(cid)]["world_to_camera"], dtype=np.float64)
        cam_pos = -w2c[:3, :3].T @ w2c[:3, 3]
        depths.append(np.linalg.norm(cam_pos))
    near, far = 0.3, max(depths) * 2.0

    rows = []
    for cid in camera_ids:
        cam = cameras[str(cid)]
        K = np.asarray(cam["K"], dtype=np.float64)
        fx, fy = K[0, 0], K[1, 1]
        rel_diff = abs(fx - fy) / max(fx, fy)
        assert rel_diff < 1e-2, f"camera {cid}: fx={fx} fy={fy} -- SIMPLE_PINHOLE needs fx==fy (rel diff {rel_diff:.4%} too large)"
        if fx != fy:
            print(f"  [warn] camera {cid}: fx={fx:.4f} fy={fy:.4f} differ by {rel_diff:.4%} "
                  f"-- averaging for SIMPLE_PINHOLE (real-camera calibration, not exactly square pixels)")
        f = (fx + fy) / 2

        w2c = np.asarray(cam["world_to_camera"], dtype=np.float64)
        Hc2w = np.linalg.inv(w2c)
        right, down, forward = Hc2w[:3, 0], Hc2w[:3, 1], Hc2w[:3, 2]
        t = Hc2w[:3, 3]
        raw_pose_3x4 = np.stack([down, right, -forward, t], axis=1)  # (3,4)

        hwf_col = np.array([H, W, f], dtype=np.float64).reshape(3, 1)
        mat_3x5 = np.concatenate([raw_pose_3x4, hwf_col], axis=1)  # (3,5)
        row = np.concatenate([mat_3x5.reshape(-1), [near, far]])  # (17,)
        rows.append(row)

    return np.stack(rows, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", choices=["masked", "raw", "both"], default="both")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.manifest).parent
    camera_ids = sorted(manifest["camera_ids"])
    frame_ids = sorted(set(manifest["frame_ids"]))
    by_cam_frame = {(s["camera_id"], s["frame_id"]): s for s in manifest["samples"]}

    variants = ["masked", "raw"] if args.variant == "both" else [args.variant]
    for variant in variants:
        out = Path(args.out_dir) / variant
        img_root = out / "images"
        img_root.mkdir(parents=True, exist_ok=True)

        for local_idx, cid in enumerate(camera_ids):
            cam_dir = img_root / str(local_idx)
            cam_dir.mkdir(exist_ok=True)
            for fid in frame_ids:
                rec = by_cam_frame[(cid, fid)]
                dst = cam_dir / f"{fid}.png"
                if dst.exists():
                    continue
                if variant == "raw":
                    Image.open(root / rec["image"]).convert("RGB").save(dst)
                else:
                    img = np.asarray(Image.open(root / rec["image"]).convert("RGB"), dtype=np.uint8)
                    mask = np.asarray(Image.open(root / rec["mask"]).convert("L")) >= 128
                    composited = np.where(mask[..., None], img, 0).astype(np.uint8)
                    Image.fromarray(composited).save(dst)
            print(f"  [{variant}] camera {cid} (local {local_idx}): {len(frame_ids)} frames done", flush=True)

        poses_bounds = build_poses_bounds(manifest, camera_ids)
        np.save(out / "poses_bounds.npy", poses_bounds)
        print(f"[{variant}] wrote poses_bounds.npy: {poses_bounds.shape}, "
              f"near={poses_bounds[:, -2].min():.3f} far={poses_bounds[:, -1].max():.3f} -> {out}", flush=True)

    holdout_ids = set(manifest.get("eval_holdout_camera_ids", []))
    train_local = [i for i, cid in enumerate(camera_ids) if cid not in holdout_ids]
    eval_local = [i for i, cid in enumerate(camera_ids) if cid in holdout_ids]
    print(f"camera_ids order -> local index: {dict(zip(camera_ids, range(len(camera_ids))))}", flush=True)
    print(f"train_cameras (local indices) = {train_local}", flush=True)
    print(f"eval_cameras (local indices) = {eval_local}", flush=True)
    print(f"frames: start=0 stop={frame_ids[-1] + 1} (n={len(frame_ids)})", flush=True)


if __name__ == "__main__":
    main()
