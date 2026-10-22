"""Diagnostic 2 (user 2026-10-12): magnitude of L2's position offset |dx| per body region, free Gaussians vs body
surfels. Region = argmax LBS joint of the Gaussian's skinning vertex (SMPL-X joints): head (neck, head, jaw, eyes),
hips/thighs (pelvis, hips, knees), torso (spine1-3, collars). mean |dx| and temporal RMS deviation of dx (the pose-driven, moving part). Every 5th frame, opacity > 0.1.
usage: PYTHONPATH=. python tools/diag_l2dx_mag.py --ckpt <run>/ckpt_final.pt [--json out]
"""
import argparse, json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from sr4d.data.dataset import MultiViewSequence
from sr4d.train.trainer import load_model

GROUPS = {"head": [12, 15, 22, 23, 24], "hips/thighs": [0, 1, 2, 4, 5], "torso": [3, 6, 9, 13, 14]}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False); cfg = OmegaConf.create(st["cfg"])
    data = MultiViewSequence(cfg.data.root, cfg.data.setting, list(range(*cfg.data.frames)), load_images=False)
    m = load_model(Path(a.ckpt), data, cfg).eval(); m.use_l2 = True; m.body_adhere = False
    sc = m.scaffold
    jb = sc.lbs_weights[sc.faces[m.body.tri][:, 0]].argmax(-1)
    jf = sc.lbs_weights[m.free.lbs_vidx].argmax(-1)
    nb = m.n_body
    T = data.num_frames
    DX, OP = [], []
    for t in range(0, T, 5):
        g = m.deform(t, need_aux=True)
        DX.append(g["aux"]["l2"]["dx"] * 1000); OP.append(g["opacities"])  # dx in canonical frame, mm
    DX, op = torch.stack(DX), torch.stack(OP).mean(0) > 0.1
    mag = DX.norm(dim=-1).mean(0)                                 # mean |dx| (static shape fix + motion)
    tstd = (DX - DX.mean(0)).norm(dim=-1).pow(2).mean(0).sqrt()   # temporal RMS deviation of dx (pose-driven motion)
    res = {}
    for kind, sl, jj in (("body", slice(0, nb), jb), ("free", slice(nb, None), jf)):
        for name, js in GROUPS.items():
            sel = torch.isin(jj, torch.tensor(js, device=jj.device)) & op[sl]
            res[f"{kind}/{name}"] = dict(mean_dx_mm=round(float(mag[sl][sel].mean()), 2),
                                         temporal_rms_dx_mm=round(float(tstd[sl][sel].mean()), 2), n=int(sel.sum()))
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
