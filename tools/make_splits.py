"""Build camera splits from cameras.json (CLAUDE.md 5.3).

Evaluation cameras are chosen ONCE and shared by every setting:
  test_extrap = a contiguous azimuth block (whole block removed from training)
                + the `--n_elev` cameras whose elevation deviates most from the median ring
  test_interp = `--n_interp` cameras spread in azimuth among the rest (sandwiched by training cams)
Training cameras for each setting = N cameras spread evenly in azimuth from the remaining pool.
For each eval camera the angle to the nearest training camera (viewing-direction angle seen from
the subject center) is stored.

Usage:
  python tools/make_splits.py --root data_dev/zju_mocap/387 --settings real_all:999 real12:12 real8:8 \
      --n_interp 3 --extrap_block 3 --n_elev 0
  DNA-Rendering: --settings real48:48 real24:24 real16:16 --exclude 22 --n_interp 4 --extrap_block 4 --n_elev 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402


def subject_center(root: Path, frames: list[int]) -> np.ndarray:
    ts = [io.load_body(root, f)["transl"].reshape(3) for f in frames[:: max(1, len(frames) // 10)]]
    return np.mean(ts, axis=0)


def angles(cams: dict, center: np.ndarray):
    ids = sorted(cams)
    C = np.stack([io.camera_center(cams[c]) for c in ids])
    up = -np.mean(np.stack([cams[c]["R"][1] for c in ids]), axis=0)  # OpenCV y = down
    up /= np.linalg.norm(up)
    d = C - center
    elev = np.degrees(np.arcsin(np.clip((d @ up) / np.linalg.norm(d, axis=1), -1, 1)))
    # azimuth in the plane orthogonal to up; reference = first camera
    ref = d[0] - (d[0] @ up) * up
    ref /= np.linalg.norm(ref)
    side = np.cross(up, ref)
    az = np.degrees(np.arctan2(d @ side, d @ ref)) % 360
    dirs = d / np.linalg.norm(d, axis=1, keepdims=True)
    return ids, az, elev, dirs


def even_pick(cands: list[int], az: np.ndarray, n: int, offset: float = 0.0) -> list[int]:
    """Pick n of cands whose azimuths best match n evenly spaced targets (greedy, unique)."""
    if n >= len(cands):
        return list(cands)
    targets = (offset + np.arange(n) * 360.0 / n) % 360
    chosen = []
    left = list(cands)
    for tg in targets:
        dd = [min(abs(az[i] - tg), 360 - abs(az[i] - tg)) for i in left]
        k = int(np.argmin(dd))
        chosen.append(left.pop(k))
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--settings", nargs="+", required=True, help="name:num_train")
    ap.add_argument("--exclude", nargs="*", default=[], help="camera ids excluded from train/eval (e.g. 4DAnyone source cam)")
    ap.add_argument("--n_interp", type=int, default=4)
    ap.add_argument("--extrap_block", type=int, default=4, help="contiguous azimuth block size (all elevations)")
    ap.add_argument("--block_center_deg", type=float, default=90.0, help="azimuth (rel. to first cam) of the removed block")
    ap.add_argument("--n_elev", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs=2, default=None)
    args = ap.parse_args()
    root = Path(args.root)
    meta = io.load_meta(root)
    cams = io.load_cameras(root)
    frames = list(range(*args.frames)) if args.frames else meta["frames"]
    center = subject_center(root, frames)
    ids, az, elev, dirs = angles(cams, center)
    excl = {ids.index(c) for c in args.exclude if c in ids}
    pool = [i for i in range(len(ids)) if i not in excl]

    # --- extrapolation: contiguous azimuth block around block_center_deg
    order = sorted(pool, key=lambda i: min(abs(az[i] - args.block_center_deg), 360 - abs(az[i] - args.block_center_deg)))
    # take the k nearest-by-azimuth cams of the median ring then all cams of other rings within that azimuth span
    block = order[: args.extrap_block]
    if block:
        lo = min(az[i] for i in block)
        hi = max(az[i] for i in block)
        block = [i for i in pool if lo <= az[i] <= hi] if hi - lo < 180 else block
    med = np.median(elev[pool])
    rest = [i for i in pool if i not in block]
    elev_out = sorted(rest, key=lambda i: -abs(elev[i] - med))[: args.n_elev]
    test_extrap = sorted(set(block) | set(elev_out))
    rest = [i for i in pool if i not in test_extrap]
    # --- interpolation: evenly spread, avoiding neighbors of the block
    test_interp = even_pick(rest, az, args.n_interp, offset=args.block_center_deg + 180.0 / max(args.n_interp, 1) + 90)
    train_pool = [i for i in rest if i not in test_interp]

    (root / "splits").mkdir(exist_ok=True)
    summary = {}
    for s in args.settings:
        name, n = s.split(":")
        train = even_pick(train_pool, az, int(n))
        nearest = {}
        for i in test_interp + test_extrap:
            ang = np.degrees(np.arccos(np.clip(dirs[train] @ dirs[i], -1, 1)))
            nearest[ids[i]] = {"cam": ids[train[int(np.argmin(ang))]], "angle_deg": float(ang.min())}
        split = dict(
            setting=name,
            train_cams=[ids[i] for i in train],
            test_interp_cams=[ids[i] for i in test_interp],
            test_extrap_cams=[ids[i] for i in test_extrap],
            excluded_cams=[ids[i] for i in sorted(excl)],
            frames=frames,
            nearest_train=nearest,
            cam_angles={ids[i]: {"azimuth": float(az[i]), "elevation": float(elev[i])} for i in range(len(ids))},
            subject_center=center.tolist(),
        )
        (root / "splits" / f"{name}.json").write_text(json.dumps(split, indent=1))
        summary[name] = dict(n_train=len(train),
                             interp_nearest=float(np.mean([nearest[ids[i]]["angle_deg"] for i in test_interp])) if test_interp else None,
                             extrap_nearest=float(np.mean([nearest[ids[i]]["angle_deg"] for i in test_extrap])) if test_extrap else None)
    print("test_interp:", [ids[i] for i in test_interp], " test_extrap:", [ids[i] for i in test_extrap])
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
