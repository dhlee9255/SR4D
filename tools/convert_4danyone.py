"""B-stage part B (sr4d env): finish a 4DAnyone-generated unified dataset and register the real evaluation cameras.

Coordinate frame = 4DAnyone canonical_human_world (the generated rig and GVHMR motion already live there, so the
SMPL-X never needs scaling). The REAL dataset (e.g. data_dev/zju_mocap/387) is mapped INTO it with a similarity
S = (s, R, t) estimated by Umeyama on the 22 shared body joints (SMPL/SMPL-X 0..21) over all frames, real joints
-> GVHMR joints. HMR pose errors are inherited by the generated views (CLAUDE.md 5.4-4), so S is a robust
least-squares fit; the registered real source camera (4DAnyone tools/register_source_camera.py) is reported as an
independent check.

Writes, under <gen_root>:
  smplx/<frame>.npz   GVHMR SMPL-X in canonical_human_world (hands = hand mean, GVHMR flat_hand_mean=False)
  cameras.json        g00..gNN generated (source="generated") + r<id> real test cams (+ source cam, source="source")
  images|masks/r<id>  symlinks to the real dataset's files
  splits/gen<N>.json  train = all generated, test_interp/test_extrap = the real split's evaluation cams
  splits/gen<N>_src.json  same + real source camera in training (B-stage optional x3 weighting)
  meta.json

  python tools/convert_4danyone.py --gen_root data_dev/zju_mocap_gen/387_gen24 --real_root data_dev/zju_mocap/387 \
      --real_setting real12 --source_cam 05 --source_registration <4Danyone>/data/source_camera_registration/zju_387_camB5/source_camera.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402
from sr4d.model.body_model import BodyModel, params_to_tensors  # noqa: E402


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--real_root", default=None, help="omit for generated-only clips (no real rig): self-evaluation split on the generated views")
    ap.add_argument("--real_setting", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--source_cam", default=None)
    ap.add_argument("--source_registration", default=None)
    a = ap.parse_args()
    gen = Path(a.gen_root)
    real = Path(a.real_root) if a.real_root else None
    raw = np.load(gen / "gvhmr_raw.npz")
    frames = [int(f) for f in raw["frames"]]
    T = raw["motion_to_world"]
    RT, tT = T[:3, :3], T[:3, 3]
    assert abs(np.linalg.det(RT) - 1) < 1e-4, "motion_to_world is expected to be rigid"

    # ---- GVHMR SMPL-X -> canonical_human_world (rotation about the root joint, like smplx)
    bmx = BodyModel("smplx", "neutral")
    hand_l, hand_r = bmx.left_hand_mean.numpy(), bmx.right_hand_mean.numpy()
    for i, f in enumerate(frames):
        src_idx = list(raw["source_frame_indices"]).index(f)
        betas = raw["betas"][src_idx][:10]
        with torch.no_grad():
            _, J = bmx.shaped(torch.as_tensor(betas, dtype=torch.float32)[None])
        J0 = J[0, 0].numpy().astype(np.float64)
        go = Rotation.from_matrix(RT) * Rotation.from_rotvec(raw["global_orient"][src_idx])
        full = np.zeros((55, 3))
        full[0] = go.as_rotvec()
        full[1:22] = raw["body_pose"][src_idx].reshape(21, 3)
        full[25:40] = hand_l
        full[40:55] = hand_r
        transl = RT @ (J0 + raw["transl"][src_idx]) + tT - J0
        io.save_body(gen, f, dict(betas=betas, full_pose=full, transl=transl, global_orient=full[0], body_pose=full[1:22].reshape(-1)))

    if real is None:  # generated-only clip: GT = the generated views themselves (self-evaluation, user direction 2026-10-08)
        gtrain = sorted(k for k in io.load_cameras(gen) if k.startswith("g"))
        (gen / "splits").mkdir(exist_ok=True)
        n = len(gtrain)
        sp = dict(setting=f"gen{n}", train_cams=gtrain, test_interp_cams=gtrain, test_extrap_cams=[], excluded_cams=[], frames=frames,
                  nearest_train={c: {"cam": c, "angle_deg": 0.0} for c in gtrain},
                  note="generated-only clip: evaluation = the generated training views (self-evaluation)")
        (gen / "splits" / f"gen{n}.json").write_text(json.dumps(sp, indent=1))
        meta = dict(dataset="fouranyone_gen", seq=gen.name, frames=frames, body_model="smplx", gender="neutral",
                    fps=a.fps, format_version=io.FORMAT_VERSION, world="4DAnyone canonical_human_world")
        (gen / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"generated-only dataset: {n} views, {len(frames)} frames -> {gen}")
        return

    # ---- registration real -> canonical world (Umeyama on body joints over frames)
    real_meta = io.load_meta(real)
    bmr = BodyModel(real_meta["body_model"], real_meta.get("gender", "neutral"))
    common = [f for f in frames if (real / "smplx" / f"{f:06d}.npz").exists()]
    with torch.no_grad():
        jr = bmr(*params_to_tensors([io.load_body(real, f) for f in common]))[1][:, :22].numpy()
        jg = bmx(*params_to_tensors([io.load_body(gen, f) for f in common]))[1][:, :22].numpy()
    s, R, t = umeyama(jr.reshape(-1, 3), jg.reshape(-1, 3))
    resid = np.linalg.norm((s * jr.reshape(-1, 3) @ R.T + t) - jg.reshape(-1, 3), axis=1)
    print(f"registration real->4DA: scale {s:.4f}, joint residual mean {resid.mean() * 100:.1f} cm, p90 {np.percentile(resid, 90) * 100:.1f} cm")

    def map_cam(c):
        c2w = np.linalg.inv(io.w2c_matrix(c))
        Rc = R @ c2w[:3, :3]
        C = s * R @ c2w[:3, 3] + t
        w2c = np.eye(4)
        w2c[:3, :3] = Rc.T
        w2c[:3, 3] = -Rc.T @ C
        return w2c

    check = {}
    real_cams = io.load_cameras(real)
    split = io.load_split(real, a.real_setting)
    if a.source_cam and a.source_registration:
        reg = json.loads(Path(a.source_registration).read_text())
        c2w_reg = np.asarray(reg["camera_to_world"])
        c2w_map = np.linalg.inv(map_cam(real_cams[a.source_cam]))
        ang = np.degrees(np.arccos(np.clip((np.trace(c2w_reg[:3, :3].T @ c2w_map[:3, :3]) - 1) / 2, -1, 1)))
        check = {"source_cam_rot_err_deg": float(ang), "source_cam_center_err_m": float(np.linalg.norm(c2w_reg[:3, 3] - c2w_map[:3, 3]))}
        print("check vs registered source camera:", check)

    cams = io.load_cameras(gen)
    cams = {k: v for k, v in cams.items() if k.startswith("g")}
    eval_ids = split["test_interp_cams"] + split["test_extrap_cams"]
    extra = [a.source_cam] if a.source_cam else []
    for cid in eval_ids + extra:
        c = dict(real_cams[cid])
        w2c = map_cam(c)
        c["R"], c["t"] = w2c[:3, :3], w2c[:3, 3]
        c["source"] = "source" if cid == a.source_cam else "real"
        cams[f"r{cid}"] = c
        for kind, ext in (("images", "jpg"), ("masks", "png")):
            d = gen / kind / f"r{cid}"
            d.mkdir(parents=True, exist_ok=True)
            for f in frames:
                dst = d / f"{f:06d}.{ext}"
                if not dst.exists():
                    os.symlink((real / kind / cid / f"{f:06d}.{ext}").resolve(), dst)
    io.save_cameras(gen, cams)

    # ---- splits (same evaluation cameras as the real setting)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_splits import angles
    allc = io.load_cameras(gen)
    center = np.mean([io.load_body(gen, f)["transl"] for f in frames], 0)
    ids, az, el, dirs = angles(allc, center)
    gtrain = sorted(k for k in allc if k.startswith("g"))
    n = len(gtrain)
    (gen / "splits").mkdir(exist_ok=True)
    for name, train in ((f"gen{n}", gtrain), (f"gen{n}_src", gtrain + ([f"r{a.source_cam}"] if a.source_cam else []))):
        ti = [ids.index(c) for c in train]
        nearest = {}
        for cid in [f"r{c}" for c in eval_ids]:
            k = ids.index(cid)
            angd = np.degrees(np.arccos(np.clip(dirs[ti] @ dirs[k], -1, 1)))
            nearest[cid] = {"cam": train[int(np.argmin(angd))], "angle_deg": float(angd.min())}
        sp = dict(setting=name, train_cams=train, test_interp_cams=[f"r{c}" for c in split["test_interp_cams"]],
                  test_extrap_cams=[f"r{c}" for c in split["test_extrap_cams"]], excluded_cams=[], frames=frames,
                  nearest_train=nearest, real_setting=a.real_setting)
        (gen / "splits" / f"{name}.json").write_text(json.dumps(sp, indent=1))
        print(name, "nearest angles:", {k: round(v["angle_deg"], 1) for k, v in nearest.items()})
    meta = dict(dataset=f"{real_meta['dataset']}_gen", seq=real_meta.get("seq", real.name), frames=frames, body_model="smplx",
                gender="neutral", source_real_root=str(real), registration=dict(scale=float(s), R=R.tolist(), t=t.tolist(),
                joint_residual_mean_m=float(resid.mean()), **check), fps=real_meta.get("fps"), format_version=io.FORMAT_VERSION,
                world="4DAnyone canonical_human_world")
    (gen / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
