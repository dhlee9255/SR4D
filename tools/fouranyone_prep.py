"""B-stage part A (run in the `4danyone` env): 4DAnyone generated views -> unified images/masks + raw GVHMR motion.

- decodes videos/dense/NN.mp4 frames [start, stop), BiRefNet foreground masks (4DAnyone's own predictor),
  undistort-free re-centring remap + downscale (same convention as every other unified dataset, D-005)
- cameras.json of the generated rig (4DAnyone canonical_human_world, OpenCV) -> unified cameras with ids g00..gNN
- GVHMR smpl_params_global + motion_world_to_canonical_world (recomputed with 4DAnyone's own _body_geometry,
  the transform used to render the skeleton conditioning) -> <out>/gvhmr_raw.npz
Part B (tools/convert_4danyone.py, sr4d env) converts the body parameters and registers the real cameras.

  conda activate 4danyone
  python tools/fouranyone_prep.py --fdanyone_dir <run>/fdanyone/<clip> --motion_dir <run>/gvhmr/results/<clip> \
      --out data_dev/zju_mocap_gen/387_gen24 --frames 0 60 --downscale 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

SR4D = Path(__file__).resolve().parents[1]
FOURDA = Path("/data/AI-Graphics/Applications/4Danyone")
sys.path.insert(0, str(SR4D))
sys.path.insert(0, str(FOURDA / "4DAnyone"))
sys.path.insert(0, str(FOURDA / "4DAnyone" / "tools"))
from sr4d.data import io  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdanyone_dir", required=True)
    ap.add_argument("--motion_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 60])
    ap.add_argument("--downscale", type=float, default=2.0)
    ap.add_argument("--gvhmr_root", default=str(FOURDA / "4DAnyone/third_party/GVHMR"))
    ap.add_argument("--model_dir", default=str(FOURDA / "models"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from fdanyone.assets import resolve_foreground_model
    from fdanyone.foreground import predict_foreground_masks
    from fdanyone.motion.result import MotionResult
    from source_camera.geometry import recompute_body_geometry

    fd, out = Path(a.fdanyone_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rig = json.loads((fd / "cameras.json").read_text())
    frames = list(range(*a.frames))
    fg_model = resolve_foreground_model(a.model_dir)
    cams = {}
    for c in rig["cameras"]:
        cid = f"g{int(c['camera_id']):02d}"
        K = np.asarray(c["K"], dtype=np.float64)
        w2c = np.linalg.inv(np.asarray(c["camera_to_world"], dtype=np.float64))
        video = np.stack([iio.imread(fd / c["video"], index=i) for i in frames])  # (T,H,W,3) RGB
        masks = predict_foreground_masks(list(video), fg_model, a.device)
        Kn = None
        for i, f in enumerate(frames):
            img, Kn = io.undistort_and_resize(video[i], K, None, a.downscale)
            m, _ = io.undistort_and_resize(np.asarray(masks[i]), K, None, a.downscale, is_mask=True)
            io.write_image(out / "images" / cid / f"{f:06d}.jpg", img)
            io.write_mask(out / "masks" / cid / f"{f:06d}.png", m)
        H, W = img.shape[:2]
        cams[cid] = dict(K=Kn, R=w2c[:3, :3], t=w2c[:3, 3], W=W, H=H, dist=np.zeros(5), source="generated")
        print(f"{cid}: {len(frames)} frames, {W}x{H}", flush=True)
    io.save_cameras(out, cams)

    motion = MotionResult.load(a.motion_dir)
    geo = recompute_body_geometry(motion, Path(a.gvhmr_root), a.device, model_dir=Path(a.model_dir))
    p = motion.smpl_params_global
    np.savez(out / "gvhmr_raw.npz",
             betas=p["betas"].numpy(), body_pose=p["body_pose"].numpy(), global_orient=p["global_orient"].numpy(),
             transl=p["transl"].numpy(), motion_to_world=np.asarray(geo.motion_world_to_canonical_world),
             source_frame_indices=np.asarray(motion.source_frame_indices), frames=np.asarray(frames))
    (out / "meta_gen.json").write_text(json.dumps(dict(fdanyone_dir=str(fd), motion_dir=str(a.motion_dir), frames=frames,
                                                       downscale=a.downscale, n_views=len(cams),
                                                       world="4DAnyone canonical_human_world"), indent=1))
    print("done ->", out)


if __name__ == "__main__":
    main()
