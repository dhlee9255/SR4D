"""SR4D unified -> FreeTimeGS++ 'dynerf' raw layout + TOML config (docs/ENV_REPORT.md §4).

images/<cam_idx>/<frame_rel>.png (TRAINING cameras only) composited on WHITE (--bg white, default) / BLACK (--bg black) / unmasked (--bg raw); white training runs through
scripts/baselines/ftgspp_train_whitebg.py (white rasteriser background, D-011),
poses_bounds.npy (LLFF, one row per camera, order = cam_idx), config.toml (4DAnyone ftgspp preset).

  python tools/export_ftgspp.py --root data_dev/zju_mocap/387 --setting real12 --out outputs/ftgspp/387/real12
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from export_common import check_centered, composite, io

PRESET = """[model]
velocity_model = "explicit"
max_duration = 1
marginal_gating = true

[data]
video_path = "{video}"
calibration_path = "{video}/poses_bounds.npy"
extracted_path = "{work}/_extracted"
memmap_path = "{work}/_memmap"
colmap_path = "{work}/_colmap"
fps = {fps}
scale = 1.0
frames = {{ "start" = 0, "stop" = {nframes} }}
eval_cameras = {eval}
train_cameras = {train}

[init]
points_path = "{work}/_points"
keyframe_stride = 10
num_points_per_frame = 700_000
num_gaussians = {num_gaussians}
num_velocity_nns = 3
sh_degree = 3
opacity = 0.5
scale = 0.1
duration = 0.1

[init.roma]
model = "outdoor"
num_nearest_cameras = 3
upsample_preds = true
symmetric = false
geometric_verification = false
backend = "v1"

[train]
iterations = {iters}
batch_size = 1
lpips_loss = true
color_correction = true
color_correction_start = 3000
color_corrector_weight = 1e-3
hard_separation = false

[train.relocation]
start = 500
stop = {reloc_stop}
every = 100
opacity_threshold = 5e-3
mode = "3d_mcmc"
score_mode = "default"
score_mode_start = 3000

[train.lrs]
means = 1.6e-4
quats = 1e-3
scales = 5e-3
opacities = 5e-2
sh_0 = 2.5e-3
sh_n = 1.25e-4
durations = 5e-3
times = 1.6e-4
velocity_model = 1.6e-3
marginal_gates = 1e-3
color_correctors = 1e-3

[train.lr_schedules]
means = 0.01
velocity_model = 0.01
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--num_gaussians", type=int, default=500000)
    ap.add_argument("--bg", choices=["white", "black", "raw"], default="white",
                    help="white = SR4D protocol (needs the white-rasteriser wrapper); black = masked on black (legacy 4DAnyone best); raw = no mask")
    ap.add_argument("--fps", type=float, default=30.0, help="FTGS time unit (t=frame/fps); 30 = 4DAnyone setting")
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    split = io.load_split(root, a.setting)
    cams = io.load_cameras(root)
    frames = split["frames"]
    # ONLY training cameras are exported: ftgspp.init.points matches every camera in the memmap, so including
    # test cameras would leak them into the initialisation. Test views are rendered by our wrapper.
    cids = sorted(split["train_cams"])
    check_centered(cams, cids)
    data = out / "data"
    (data / "images").mkdir(parents=True, exist_ok=True)

    def job(arg):
        idx, cid, rel, f = arg
        img = io.read_image(root / "images" / cid / f"{f:06d}.jpg")
        m = io.read_mask(root / "masks" / cid / f"{f:06d}.png")
        p = data / "images" / str(idx) / f"{rel}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        out_img = img if a.bg == "raw" else composite(img, m, 255 if a.bg == "white" else 0)
        cv2.imwrite(str(p), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    jobs = [(i, c, r, f) for i, c in enumerate(cids) for r, f in enumerate(frames)]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(job, jobs))

    rows = []
    for cid in cids:
        c = cams[cid]
        c2w = np.linalg.inv(io.w2c_matrix(c))
        right, down, fwd, t = c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3]
        pose = np.stack([down, right, -fwd, t], 1)  # LLFF raw columns (inverse of ftgspp.data.sfm reorder)
        hwf = np.array([[c["H"]], [c["W"]], [c["K"][0, 0]]])
        dist = np.linalg.norm(t)
        rows.append(np.concatenate([np.concatenate([pose, hwf], 1).reshape(-1), [0.1, dist * 2]]))
    np.save(data / "poses_bounds.npy", np.stack(rows))

    idx = {c: i for i, c in enumerate(cids)}
    train = [idx[c] for c in split["train_cams"]]
    ev = [0]  # FTGS's own eval stage needs a non-empty list; it is not used (SR4D common evaluation instead)
    cfg = PRESET.format(video=data, work=out / "work", fps=a.fps, nframes=len(frames), eval=ev, train=train,
                        num_gaussians=a.num_gaussians, iters=a.iters, reloc_stop=int(a.iters * 25 / 30))
    (out / "config.toml").write_text(cfg)
    (out / "export_meta.json").write_text(json.dumps({"cam_index": idx, "frames": frames, "fps": a.fps, "background": a.bg}, indent=1))
    print(f"FTGS++ export: {len(cids)} cams ({len(train)} train) x {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
