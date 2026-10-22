"""B-stage registration v2: place REAL cameras into the 4DAnyone canonical_human_world (GVHMR SMPL-X world).

Each camera pose is R = exp([w]x) R_init, t = t_init + dt (known real intrinsics, fixed), initialised from an existing
B-stage dataset (e.g. the joint-Umeyama or source-silhouette similarity). Loss per frame:
    soft-silhouette Dice (GVHMR SMPL-X surface splats vs real mask)
  + lambda_kp * Huber(projected SMPL-X joints - RTMW-x 2D keypoints, delta 5 px) / image diagonal-normalised
  + tiny prior on |w|, |dt|
MSE silhouette is NOT used (it admits the degenerate "camera escapes off-screen" solution).
Modes:
  similarity   one 7DoF S (log s, w, dt) shared by all real cameras, fitted on --fit_cams only (e.g. the source cam
               -> uses no evaluation data)
  per_camera   independent 6DoF pose per real camera (test-time pose alignment; uses evaluation-camera masks and
               keypoints for POSE only, identical for every method)
Writes a new dataset dir (symlinks to everything else) with the re-posed r* cameras + registration report.

  python tools/register_real_cams.py --gen_root data_dev/zju_mocap_gen/387_gen24 --real_root data_dev/zju_mocap/387 \
      --mode per_camera --out_root data_dev/zju_mocap_gen/387_gen24_pc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gsplat
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402
from sr4d.model.body_model import BodyModel, params_to_tensors  # noqa: E402
from sr4d.model.scaffold import sample_surface  # noqa: E402
from sr4d.utils.geometry import rotvec_to_rotmat  # noqa: E402

# COCO-17 index -> SMPL-X joint index (legacy STABLE_JOINT_NAMES: shoulders, elbows, wrists, hips, knees, ankles)
COCO2SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
KP_W = {11: 0.5, 12: 0.5}  # SMPL hip joints sit inside the pelvis, COCO hips are on the surface


def silhouette(pts, w2c, K, W, H):
    n = pts.shape[0]
    dev = pts.device
    _, a, _ = gsplat.rasterization(pts, torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4), torch.full((n, 3), 0.008, device=dev),
                                   torch.full((n,), 0.9, device=dev), torch.ones(n, 1, device=dev), w2c[None], K[None], W, H, packed=False)
    return a[0, ..., 0]


def dice(a, m):
    return 1 - 2 * (a * m).sum() / (a.sum() + m.sum()).clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", required=True, help="B-stage dataset providing the initial r* poses + GVHMR SMPL-X")
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--mode", choices=["similarity", "per_camera"], required=True)
    ap.add_argument("--fit_cams", nargs="*", default=None, help="similarity mode: real cam ids driving the fit (default: source cam)")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lambda_kp", type=float, default=1.0)
    ap.add_argument("--kp_thresh", type=float, default=0.5)
    a = ap.parse_args()
    gen, real, out = Path(a.gen_root), Path(a.real_root), Path(a.out_root)
    dev = "cuda"
    meta = io.load_meta(gen)
    frames = meta["frames"]
    cams = io.load_cameras(gen)
    rids = sorted(k for k in cams if k.startswith("r"))
    src = next((k for k in rids if cams[k].get("source") == "source"), None)

    bm = BodyModel("smplx", "neutral").to(dev)
    with torch.no_grad():
        verts, joints, _ = bm(*params_to_tensors([io.load_body(gen, f) for f in frames], dev))
    g = torch.Generator(device=dev).manual_seed(0)
    tri, bary = sample_surface(verts[0], bm.faces, 20000, g)
    pts = (verts[:, bm.faces[tri]] * bary[None, :, :, None]).sum(2)  # (T,P,3)
    coco = list(COCO2SMPLX)
    J3 = joints[:, [COCO2SMPLX[c] for c in coco]]  # (T,12,3)
    jw = torch.tensor([KP_W.get(c, 1.0) for c in coco], device=dev)

    data = {}
    for cid in rids:
        rc = cid[1:]
        c = cams[cid]
        K = torch.tensor(c["K"], dtype=torch.float32, device=dev)
        K[:2, 2] += 0.5  # gsplat pixel-centre convention
        m = torch.stack([torch.as_tensor(io.read_mask(real / "masks" / rc / f"{f:06d}.png") > 127) for f in frames]).float().to(dev)
        kf = real / "keypoints_rtmw" / f"{rc}.npz"
        kp = sc = None
        if kf.exists():
            d = np.load(kf)
            idx = [list(d["frames"]).index(f) for f in frames]
            kp = torch.tensor(d["kpts"][idx][:, coco], device=dev)
            sc = torch.tensor(d["scores"][idx][:, coco], device=dev)
        w2c0 = torch.tensor(io.w2c_matrix(c), dtype=torch.float32, device=dev)
        data[cid] = dict(K=K, Kcv=torch.tensor(c["K"], dtype=torch.float32, device=dev), W=c["W"], H=c["H"], m=m, kp=kp, sc=sc, w2c0=w2c0)

    def project(X, w2c, Kcv):
        Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        uv = Xc @ Kcv.T
        return uv[..., :2] / uv[..., 2:].clamp_min(1e-6)

    def metrics(cid, w2c):
        d = data[cid]
        ious, kerr = [], []
        with torch.no_grad():
            for i in range(len(frames)):
                al = silhouette(pts[i], w2c, d["K"], d["W"], d["H"]) > 0.5
                mm = d["m"][i] > 0.5
                ious.append(float((al & mm).sum() / (al | mm).sum().clamp_min(1)))
                if d["kp"] is not None:
                    ok = d["sc"][i] > a.kp_thresh
                    if ok.any():
                        kerr.append(float((project(J3[i], w2c, d["Kcv"]) - d["kp"][i])[ok].norm(dim=-1).mean()))
        return float(np.mean(ious)), (float(np.mean(kerr)) if kerr else float("nan"))

    def frame_loss(cid, w2c, i):
        d = data[cid]
        l = dice(silhouette(pts[i], w2c, d["K"], d["W"], d["H"]), d["m"][i])
        if d["kp"] is not None and a.lambda_kp > 0:
            ok = (d["sc"][i] > a.kp_thresh).float() * jw
            r = (project(J3[i], w2c, d["Kcv"]) - d["kp"][i]).norm(dim=-1)
            diag = float(np.hypot(d["W"], d["H"]))
            l = l + a.lambda_kp * (F.huber_loss(r, torch.zeros_like(r), delta=5.0, reduction="none") * ok).sum() / ok.sum().clamp_min(1) / diag
        return l

    before = {cid: metrics(cid, data[cid]["w2c0"]) for cid in rids}
    new_w2c = {}
    if a.mode == "per_camera":
        for cid in rids:
            d = data[cid]
            w = torch.zeros(3, device=dev, requires_grad=True)
            dt = torch.zeros(3, device=dev, requires_grad=True)
            opt = torch.optim.Adam([{"params": [w], "lr": 2e-3}, {"params": [dt], "lr": 5e-3}])

            def pose():
                M = torch.eye(4, device=dev)
                M[:3, :3] = rotvec_to_rotmat(w) @ d["w2c0"][:3, :3]
                M[:3, 3] = d["w2c0"][:3, 3] + dt
                return M
            for it in range(a.iters):
                idx = torch.randint(0, len(frames), (6,)).tolist()
                M = pose()
                loss = sum(frame_loss(cid, M, i) for i in idx) / len(idx) + 1e-3 * (w * w).sum() + 1e-4 * (dt * dt).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            new_w2c[cid] = pose().detach()
    else:
        fit = [f"r{c}" if not c.startswith("r") else c for c in a.fit_cams] if a.fit_cams else [src]
        log_s = torch.zeros((), device=dev, requires_grad=True)
        w = torch.zeros(3, device=dev, requires_grad=True)
        dt = torch.zeros(3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([{"params": [log_s], "lr": 2e-3}, {"params": [w], "lr": 2e-3}, {"params": [dt], "lr": 5e-3}])

        def pose(cid):
            # extra similarity applied to the world points of the current S: X' = s Rw X + dt  =>  camera c2w' = (s Rw c2w, s Rw C + dt)
            c2w = torch.linalg.inv(data[cid]["w2c0"])
            Rw = rotvec_to_rotmat(w)
            Rc = Rw @ c2w[:3, :3]
            C = log_s.exp() * (Rw @ c2w[:3, 3]) + dt
            M = torch.eye(4, device=dev)
            M[:3, :3] = Rc.T
            M[:3, 3] = -Rc.T @ C
            return M
        for it in range(a.iters):
            idx = torch.randint(0, len(frames), (6,)).tolist()
            loss = sum(frame_loss(cid, pose(cid), i) for cid in fit for i in idx) / (len(idx) * len(fit))
            loss = loss + 1e-3 * (w * w).sum() + 1e-4 * (dt * dt).sum() + 1e-3 * log_s ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        new_w2c = {cid: pose(cid).detach() for cid in rids}
    after = {cid: metrics(cid, new_w2c[cid]) for cid in rids}

    rep = {}
    for cid in rids:
        dR = new_w2c[cid][:3, :3] @ data[cid]["w2c0"][:3, :3].T
        ang = float(torch.rad2deg(torch.arccos(((dR.trace() - 1) / 2).clamp(-1, 1))))
        C0 = -data[cid]["w2c0"][:3, :3].T @ data[cid]["w2c0"][:3, 3]
        C1 = -new_w2c[cid][:3, :3].T @ new_w2c[cid][:3, 3]
        rep[cid] = dict(iou_before=before[cid][0], iou_after=after[cid][0], kp_px_before=before[cid][1], kp_px_after=after[cid][1],
                        rot_change_deg=ang, center_change_m=float((C1 - C0).norm()), role=cams[cid].get("source", "real"))
        print(f"{cid} ({rep[cid]['role']:6s}) IoU {before[cid][0]:.3f} -> {after[cid][0]:.3f} | kp err {before[cid][1]:6.1f} -> {after[cid][1]:6.1f} px "
              f"| dR {ang:5.2f} deg dC {rep[cid]['center_change_m'] * 100:5.1f} cm")
    ev = [k for k in rids if k != src]
    print(f"eval cams mean IoU {np.mean([before[k][0] for k in ev]):.3f} -> {np.mean([after[k][0] for k in ev]):.3f}")

    # ---- new dataset dir: symlink everything, rewrite cameras/meta
    out.mkdir(parents=True, exist_ok=True)
    for p in gen.iterdir():
        if p.name in ("cameras.json", "meta.json", "verify"):
            continue
        q = out / p.name
        if not q.exists():
            os.symlink(p.resolve(), q)
    for cid in rids:
        M = new_w2c[cid].cpu().numpy().astype(np.float64)
        cams[cid]["R"], cams[cid]["t"] = M[:3, :3], M[:3, 3]
    io.save_cameras(out, cams)
    meta["registration_init"] = meta.get("registration")
    meta["registration"] = dict(method=f"register_real_cams {a.mode}: Dice + {a.lambda_kp} x Huber(RTMW-x kp)", init_root=str(gen),
                                fit_cams=(a.fit_cams or [src]) if a.mode == "similarity" else "each camera itself",
                                uses_eval_data=a.mode == "per_camera" or any(c != src for c in (a.fit_cams or [src])), per_cam=rep)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
