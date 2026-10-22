"""DNA-Rendering (.smc) -> SR4D unified format.

Format checked against the official reader (third_party/DNA-Rendering/scripts/3DGS/{SMCReader,dataset_readers}.py):
  main  <actor>_<perf>.smc         : Camera_5mp/<id>/color/<frame> (BGR JPEG bytes), ids '0'..'47' (2448x2048)
                                     Camera_12mp/<id>/color/<frame>, ids '48'..'59' (4K)   -> not used (A-stage = 48 x 2K)
                                     attrs: actor_id, performance_id, gender, ...
  annots <actor>_<perf>_annots.smc : Camera_Parameter/<id:02d>/{K, D (k1,k2,p1,p2,k3), RT (= camera-to-world, OpenCV),
                                     Color_Calibration}, Mask/<id>/mask/<frame>, Keypoints_2D/<id:02d>,
                                     SMPLx/{betas, expression, fullpose (N,55,3), transl, scale}
  official loader: cv2.undistort(img, K, D) keeping K; SMPLX(gender, flat_hand_mean=False, use_pca=False,
                   use_face_contour=True), fullpose split = [global, body 1:22, jaw 22, leye 23, reye 24,
                   lhand 25:40, rhand 40:55].
Colour calibration: `Color_Calibration` is applied when --color_calib (default) as a 3x3 matrix on linear RGB
values in [0,1] if it is 3x3, else (3x4) affine. ** UNVERIFIED until a real file is available: the first run
prints its shape and an IoU check (tools/verify_dataset.py) must pass before anything else. **
`SMPLx/scale` is applied to vertices about the origin if != 1 (also unverified -> checked by verify_dataset IoU).

  python tools/convert_dna_rendering.py --main <raw>/main/0012_09.smc --annots <raw>/annotations/0012_09_annots.smc \
      --out <work>/dna_rendering/0012_09 --frames 0 60 --downscale 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402
from sr4d.model.body_model import BodyModel  # noqa: E402


def decode(arr) -> np.ndarray:
    return cv2.imdecode(np.asarray(arr), cv2.IMREAD_COLOR)


def apply_color_calib(img_bgr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
    flat = rgb.reshape(-1, 3)
    if cc.shape == (3, 3):
        out = flat @ cc.T
    elif cc.shape == (3, 4):
        out = flat @ cc[:, :3].T + cc[:, 3]
    else:
        raise ValueError(f"unexpected Color_Calibration shape {cc.shape}")
    out = np.clip(out, 0, 1).reshape(rgb.shape)
    return (out[..., ::-1] * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--annots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 60])
    ap.add_argument("--downscale", type=float, default=2.0)
    ap.add_argument("--cams", nargs="*", default=None, help="default: all Camera_5mp ids (0..47)")
    ap.add_argument("--no_color_calib", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    smc = h5py.File(a.main, "r")
    ann = h5py.File(a.annots, "r")
    gender = smc.attrs["gender"]
    gender = gender.decode() if isinstance(gender, bytes) else str(gender)
    cam_ids = a.cams or sorted(smc["Camera_5mp"].keys(), key=int)
    frames = list(range(*a.frames))
    print("actor", dict(smc.attrs), "| 5mp cams", len(cam_ids), "| keys", list(ann.keys()))

    cams, calib = {}, {}
    for cid in cam_ids:
        p = ann["Camera_Parameter"][f"{int(cid):02d}"]
        K = p["K"][()].astype(np.float64)
        D = p["D"][()].reshape(-1).astype(np.float64)
        RT = p["RT"][()].astype(np.float64)  # camera-to-world
        cc = p["Color_Calibration"][()].astype(np.float64)
        w2c = np.linalg.inv(RT)
        img0 = decode(smc["Camera_5mp"][cid]["color"][str(frames[0])][()])
        H0, W0 = img0.shape[:2]
        _, Kn = io.undistort_and_resize(img0, K, D, a.downscale)
        key = f"{int(cid):02d}"
        cams[key] = dict(K=Kn, R=w2c[:3, :3], t=w2c[:3, 3], W=int(round(W0 / a.downscale)), H=int(round(H0 / a.downscale)),
                         dist=np.zeros(5), source="real")
        calib[key] = (cid, K, D, cc)
    print("Color_Calibration shape:", next(iter(calib.values()))[3].shape)

    def job(arg):
        key, f = arg
        cid, K, D, cc = calib[key]
        img = decode(smc["Camera_5mp"][cid]["color"][str(f)][()])
        if not a.no_color_calib:
            img = apply_color_calib(img, cc)
        m = np.max(decode(ann["Mask"][cid]["mask"][str(f)][()]), 2)
        img, _ = io.undistort_and_resize(img, K, D, a.downscale)
        m, _ = io.undistort_and_resize(m, K, D, a.downscale, is_mask=True)
        io.write_image(out / "images" / key / f"{f:06d}.jpg", img[..., ::-1].copy())
        io.write_mask(out / "masks" / key / f"{f:06d}.png", m)

    for k in cams:  # sequential: a single h5py handle is not thread-safe
        for f in frames:
            job((k, f))
    io.save_cameras(out, cams)

    sx = ann["SMPLx"]
    scale = float(np.asarray(sx["scale"][()]).reshape(-1)[0]) if "scale" in sx else 1.0
    bm = BodyModel("smplx", gender)
    for f in frames:
        fp = sx["fullpose"][()][f].reshape(55, 3).astype(np.float64)
        full = fp.copy()
        full[25:40] += bm.left_hand_mean.numpy()  # stored with flat_hand_mean=False -> absolute
        full[40:55] += bm.right_hand_mean.numpy()
        betas = sx["betas"][()][f if sx["betas"].shape[0] > 1 else 0].reshape(-1)[:10]
        expr = sx["expression"][()][f].reshape(-1)[:10]
        io.save_body(out, f, dict(betas=betas, full_pose=full, transl=sx["transl"][()][f].reshape(3), expression=expr,
                                  global_orient=full[0], body_pose=full[1:22].reshape(-1), scale=np.array([scale])))
    if abs(scale - 1) > 1e-6:
        print(f"WARNING: SMPLx scale={scale} != 1 -- not applied by BodyModel; check verify_dataset IoU")
    seq = Path(a.main).stem
    meta = dict(dataset="dna_rendering", seq=seq, source_main=str(a.main), source_annots=str(a.annots), frames=frames,
                downscale=a.downscale, body_model="smplx", gender=gender, fps=15, fps_verified=False, color_calibration_applied=not a.no_color_calib,
                undistorted=True, principal_point_centered=True, units="m", format_version=io.FORMAT_VERSION, smplx_scale=scale,
                camera_group="Camera_5mp")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(cams)} cams x {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
