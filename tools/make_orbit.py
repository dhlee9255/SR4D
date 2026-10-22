"""Build a novel-view orbit (one revolution on the circle fitted to the training cameras) for NVS videos.

Writes <out>/cameras.json (o000..) and <out>/orbit.json {cams, frame_rel}: time advances with the orbit and
ping-pongs over the sequence so the motion stays continuous.
  python tools/make_orbit.py --root <unified root> --setting gen24 --out <dir> [--n 120] [--elev_deg 0] [--radius_scale 1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--elev_deg", type=float, default=0.0, help="raise the orbit above the fitted ring (extrapolation)")
    ap.add_argument("--radius_scale", type=float, default=1.0)
    ap.add_argument("--abs_elev_deg", type=float, default=None,
                    help="absolute elevation seen from the look-at target, at the mean training camera distance (overrides --elev_deg)")
    a = ap.parse_args()
    root = Path(a.root)
    split = io.load_split(root, a.setting)
    cams = io.load_cameras(root)
    tr = [cams[c] for c in split["train_cams"]]
    C = np.stack([-c["R"].T @ c["t"] for c in tr])                  # centres
    fwd = np.stack([c["R"][2] for c in tr])                          # optical axes (world)
    up = -np.stack([c["R"][1] for c in tr]).mean(0); up /= np.linalg.norm(up)
    # look-at target: least-squares point closest to all optical axes
    A = sum(np.eye(3) - np.outer(f, f) for f in fwd); b = sum((np.eye(3) - np.outer(f, f)) @ c for f, c in zip(fwd, C))
    target = np.linalg.solve(A, b)
    # ring plane: normal ~ up, centre = projection of the camera centroid onto the axis through target
    h = ((C - target) @ up).mean()
    centre = target + h * up
    rad = np.linalg.norm((C - centre) - np.outer((C - centre) @ up, up), axis=1).mean() * a.radius_scale
    e1 = (C[0] - centre) - ((C[0] - centre) @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    # orbit direction follows the camera ordering of the training rig
    if ((C[1] - centre) @ e2) < 0:
        e2 = -e2
    lift = rad * np.tan(np.radians(a.elev_deg))
    K, W, H = tr[0]["K"], tr[0]["W"], tr[0]["H"]
    out = {}
    nf = len(split["frames"])
    for i in range(a.n):
        th = 2 * np.pi * i / a.n
        p = centre + rad * (np.cos(th) * e1 + np.sin(th) * e2) + lift * up
        if a.abs_elev_deg is not None:
            D, e = np.linalg.norm(C - target, axis=1).mean() * a.radius_scale, np.radians(a.abs_elev_deg)
            p = target + D * (np.cos(e) * (np.cos(th) * e1 + np.sin(th) * e2) + np.sin(e) * up)
        z = target - p; z /= np.linalg.norm(z)
        x = np.cross(z, up); x /= np.linalg.norm(x)   # OpenCV: x right, y down, z forward
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        out[f"o{i:03d}"] = dict(K=K, R=R, t=-R @ p, W=W, H=H, dist=np.zeros(5), source="orbit")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    io.save_cameras(Path(a.out), out)
    period = 2 * (nf - 1)
    rel = [(i % period) if (i % period) < nf else period - (i % period) for i in range(a.n)]
    (Path(a.out) / "orbit.json").write_text(json.dumps(dict(cams=list(out), frame_rel=rel, radius=float(rad), elev_deg=a.elev_deg, abs_elev_deg=a.abs_elev_deg,
                                                            target=target.tolist())))
    print(f"orbit: {a.n} cams, radius {rad:.3f}, cam height above target {h:.3f}, -> {a.out}")


if __name__ == "__main__":
    main()
