"""Refine the real->4DAnyone similarity S of a B-stage dataset using ONLY the real source camera (the input video
camera, e.g. ZJU B5 -- its frames were 4DAnyone's / GVHMR's input, so this uses no evaluation data).

The GVHMR SMPL-X surface (canonical_human_world) is splatted into the source camera placed by S (with the source
camera's KNOWN intrinsics, so scale is observable) and S = (log s, rotvec, t) is optimised so the silhouette matches
the source masks over all frames. All r* cameras in cameras.json are then re-mapped with the refined S.
Evaluation cameras are only used to REPORT diagnostic IoU (never optimised).

  python tools/refine_registration.py --gen_root data_dev/zju_mocap_gen/387_gen24 --real_root data_dev/zju_mocap/387 --source_cam 05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gsplat
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402
from sr4d.model.body_model import BodyModel, params_to_tensors  # noqa: E402
from sr4d.model.scaffold import sample_surface  # noqa: E402
from sr4d.utils.geometry import rotvec_to_rotmat  # noqa: E402


def cam_in_gen(c2w_real: torch.Tensor, s, R, t):
    Rc = R @ c2w_real[:3, :3]
    C = s * (R @ c2w_real[:3, 3]) + t
    w2c = torch.eye(4, device=R.device)
    w2c[:3, :3] = Rc.T
    w2c[:3, 3] = -Rc.T @ C
    return w2c


def silhouette(pts, w2c, K, W, H):
    n = pts.shape[0]
    dev = pts.device
    _, a, _ = gsplat.rasterization(pts, torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4), torch.full((n, 3), 0.008, device=dev),
                                   torch.full((n,), 0.9, device=dev), torch.ones(n, 1, device=dev), w2c[None], K[None], W, H, packed=False)
    return a[0, ..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--source_cam", required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--dry_run", action="store_true", help="only report the IoU change, write nothing")
    a = ap.parse_args()
    gen, real = Path(a.gen_root), Path(a.real_root)
    dev = "cuda"
    meta = io.load_meta(gen)
    reg = meta["registration"]
    frames = meta["frames"]
    bm = BodyModel("smplx", "neutral").to(dev)
    with torch.no_grad():
        verts = bm(*params_to_tensors([io.load_body(gen, f) for f in frames], dev))[0]
    g = torch.Generator(device=dev).manual_seed(0)
    tri, bary = sample_surface(verts[0], bm.faces, 20000, g)
    pts = (verts[:, bm.faces[tri]] * bary[None, :, :, None]).sum(2)  # (T,P,3)

    rc = io.load_cameras(real)
    src = rc[a.source_cam]
    K = torch.tensor(src["K"], dtype=torch.float32, device=dev)
    K[:2, 2] += 0.5
    c2w_src = torch.tensor(np.linalg.inv(io.w2c_matrix(src)), dtype=torch.float32, device=dev)
    masks = torch.stack([torch.as_tensor(io.read_mask(real / "masks" / a.source_cam / f"{f:06d}.png") > 127) for f in frames]).float().to(dev)

    R0 = torch.tensor(reg["R"], dtype=torch.float32, device=dev)
    t0 = torch.tensor(reg["t"], dtype=torch.float32, device=dev)
    log_s = torch.tensor(np.log(reg["scale"]), dtype=torch.float32, device=dev, requires_grad=True)
    drot = torch.zeros(3, device=dev, requires_grad=True)
    dt = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [log_s], "lr": 3e-3}, {"params": [drot], "lr": 3e-3}, {"params": [dt], "lr": 1e-2}])

    def cur():
        return log_s.exp(), rotvec_to_rotmat(drot) @ R0, t0 + dt

    def iou_src():
        with torch.no_grad():
            s, R, t = cur()
            w2c = cam_in_gen(c2w_src, s, R, t)
            v = []
            for i in range(len(frames)):
                al = silhouette(pts[i], w2c, K, src["W"], src["H"]) > 0.5
                m = masks[i] > 0.5
                v.append(float((al & m).sum() / (al | m).sum().clamp_min(1)))
            return float(np.mean(v))

    i0 = iou_src()
    for it in range(a.iters):
        idx = torch.randint(0, len(frames), (6,))
        s, R, t = cur()
        w2c = cam_in_gen(c2w_src, s, R, t)
        loss = sum(((silhouette(pts[i], w2c, K, src["W"], src["H"]) - masks[i]) ** 2).mean() for i in idx.tolist()) / 6
        opt.zero_grad()
        loss.backward()
        opt.step()
    i1 = iou_src()
    s, R, t = (x.detach().cpu().numpy() for x in cur())
    s = float(s)
    print(f"source-camera silhouette IoU {i0:.3f} -> {i1:.3f}; scale {reg['scale']:.4f} -> {s:.4f}, "
          f"rot change {np.degrees(np.linalg.norm(drot.detach().cpu().numpy())):.2f} deg, trans change {np.linalg.norm(dt.detach().cpu().numpy()) * 100:.1f} cm")
    if i1 <= i0:
        print("no improvement -> keeping the joint-based registration")
        return
    if a.dry_run:
        return

    # re-map every real camera with the refined S
    cams = io.load_cameras(gen)
    Rt, tt = torch.tensor(R, device=dev), torch.tensor(t, device=dev)
    for cid in [k for k in cams if k.startswith("r")]:
        c2w = torch.tensor(np.linalg.inv(io.w2c_matrix(rc[cid[1:]])), dtype=torch.float32, device=dev)
        w2c = cam_in_gen(c2w, s, Rt, tt).cpu().numpy().astype(np.float64)
        cams[cid]["R"], cams[cid]["t"] = w2c[:3, :3], w2c[:3, 3]
    io.save_cameras(gen, cams)
    meta["registration_joint"] = reg
    meta["registration"] = dict(scale=s, R=R.tolist(), t=t.tolist(), method="source-camera silhouette refinement",
                                source_iou_before=i0, source_iou_after=i1)
    (gen / "meta.json").write_text(json.dumps(meta, indent=1))
    print("updated cameras.json and meta.json; rerun tools/verify_dataset.py for the diagnostic IoU on all cameras")


if __name__ == "__main__":
    main()
