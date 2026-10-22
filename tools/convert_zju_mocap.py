"""ZJU-MoCap (refine, EasyMocap layout) -> SR4D unified format.

Development substitute while DNA-Rendering is unavailable (docs/DECISIONS.md D-003).
Source conventions (checked in 4dgs_recon/scripts/build_native_zju_manifest.py and by IoU test):
  intri.yml: K_<cam>, dist_<cam>;  extri.yml: Rot_<cam> (3x3), T_<cam> (meters), x_cam = Rot x_world + T
  mask/<cam>/<frame>.png in {0,1};  new_params/<frame>.npy: Rh, Th, poses(72, poses[:3]=0), shapes(10)
  EasyMocap SMPL: v = R(Rh) @ smpl(poses, shapes) + Th  (rotation about the world origin)
  -> unified (smplx-style, rotation about root joint J0): global = R(Rh) R(poses[:3]), transl = R(Rh) J0 + Th - J0

Usage:
  python tools/convert_zju_mocap.py --src .../zju_mocap_refine/my_387 --out data_dev/zju_mocap/387 \
      --frames 0 60 --downscale 2
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402
from sr4d.model.body_model import BodyModel  # noqa: E402


def read_yml(path: Path, cams: list[str], keys: list[str]) -> dict:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    out = {}
    for c in cams:
        out[c] = {k: fs.getNode(f"{k}_{c}").mat() for k in keys}
    fs.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 60], help="[start, stop)")
    ap.add_argument("--downscale", type=float, default=2.0)
    ap.add_argument("--params_dir", default="new_params")
    ap.add_argument("--mask_dir", default="mask", help="mask (per-frame matting, ragged on dark legs) or mask_cihp (CIHP parsing, used by GauHuman)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam_names = sorted([p.name for p in src.glob("Camera_B*") if p.is_dir()], key=lambda s: int(s.split("B")[-1]))
    intri = read_yml(src / "intri.yml", cam_names, ["K", "dist"])
    extri = read_yml(src / "extri.yml", cam_names, ["Rot", "T"])
    frames = list(range(*args.frames))

    cams = {}
    for cn in cam_names:
        cid = f"{int(cn.split('B')[-1]):02d}"
        K = intri[cn]["K"].astype(np.float64)
        dist = intri[cn]["dist"].reshape(-1).astype(np.float64)
        img0 = cv2.imread(str(src / cn / f"{0:06d}.jpg"))
        H0, W0 = img0.shape[:2]
        _, Kn = io.undistort_and_resize(img0, K, dist, args.downscale)
        Hn, Wn = int(round(H0 / args.downscale)), int(round(W0 / args.downscale))
        cams[cid] = dict(K=Kn, R=extri[cn]["Rot"], t=extri[cn]["T"].reshape(3), W=Wn, H=Hn, dist=np.zeros(5), source="real",
                         _K0=K, _dist0=dist, _name=cn)

    def work(args_):
        cid, f = args_
        c = cams[cid]
        img = cv2.imread(str(src / c["_name"] / f"{f:06d}.jpg"))
        m = cv2.imread(str(src / args.mask_dir / c["_name"] / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
        m = (m > 0).astype(np.uint8) * 255
        img, _ = io.undistort_and_resize(img, c["_K0"], c["_dist0"], args.downscale)
        m, _ = io.undistort_and_resize(m, c["_K0"], c["_dist0"], args.downscale, is_mask=True)
        io.write_image(out / "images" / cid / f"{f:06d}.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        io.write_mask(out / "masks" / cid / f"{f:06d}.png", m)

    jobs = [(cid, f) for cid in cams for f in frames]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(work, jobs))
    io.save_cameras(out, cams)

    # body params
    bm = BodyModel("smpl")
    for f in frames:
        p = np.load(src / args.params_dir / f"{f}.npy", allow_pickle=True).item()
        betas = p["shapes"].reshape(10).astype(np.float32)
        poses = p["poses"].reshape(24, 3).astype(np.float64)
        Rh = Rotation.from_rotvec(p["Rh"].reshape(3))
        Th = p["Th"].reshape(3)
        with torch.no_grad():
            _, J = bm.shaped(torch.as_tensor(betas)[None])
        J0 = J[0, 0].numpy().astype(np.float64)
        g = Rh * Rotation.from_rotvec(poses[0])
        full = poses.copy()
        full[0] = g.as_rotvec()
        transl = Rh.apply(J0) + Th - J0
        io.save_body(out, f, dict(betas=betas, full_pose=full, transl=transl,
                                  global_orient=full[0], body_pose=full[1:].reshape(-1)))

    meta = dict(
        dataset="zju_mocap", source=str(src), frames=frames, downscale=args.downscale,
        body_model="smpl", gender="neutral", fps=50, color_calibration_applied=False,
        undistorted=True, units="m", format_version=io.FORMAT_VERSION,
        params_dir=args.params_dir, mask_dir=args.mask_dir, camera_names={cid: c["_name"] for cid, c in cams.items()},
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(cams)} cams x {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
