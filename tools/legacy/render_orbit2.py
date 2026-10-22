"""Proper single-revolution orbiting-camera render for a 24-camera RING rig
(not a forward-facing array). render.py's own default-camera path calls
ellipse_trajectory() on the naive elementwise average of the 24 c2w matrices
(torch.inverse(c2ws.mean(0))) -- averaging rotation matrices directly like
that is not a valid rotation (verified: the resulting "forward" column had
magnitude 0.259, not 1), which produced a garbled/broken orbit for this
360-degree ring layout. This script builds the orbit analytically instead:
fits a circle to the real camera positions (all already equidistant from
their centroid, confirmed programmatically), and looks at the centroid
using the AVERAGE OF THE REAL PER-CAMERA UP UNIT VECTORS (each individually
valid, unlike averaging whole rotation matrices) as the up hint. Reuses
render.py's own render_rgb/intrinsic and math.py's look_at, unmodified."""
import argparse
import math
from pathlib import Path

import imageio.v3 as iio
import torch
import torch.nn.functional as F

from ftgspp.data.utils import MultiViewData
from ftgspp.models.gaussians import Gaussians
from ftgspp.render import intrinsic, render_rgb
from ftgspp.utils import Config
from ftgspp.utils.math import look_at

ap = argparse.ArgumentParser()
ap.add_argument("run_path", type=Path)
ap.add_argument("output", type=Path)
ap.add_argument("--radius-scale", type=float, default=1.0, help="multiply the fitted ring radius")
ap.add_argument("--fps", type=float, default=None)
args = ap.parse_args()

config = Config.load(args.run_path / "config.toml")
dataset = MultiViewData.load(config.data.memmap_path)
h, w = dataset.height, dataset.width
h -= h % 2
w -= w % 2

fx = float(dataset.intrinsic[0, 0, 0, 0])
fov = 2 * math.atan2(dataset.width, 2 * fx)

c2ws = torch.linalg.inv(dataset[0].w2c)  # (num_cameras, 4, 4)
positions = c2ws[:, :3, 3]  # (N, 3)
center = positions.mean(0)
radii = (positions - center).norm(dim=1)
radius = radii.mean().item() * args.radius_scale
print(f"ring position centroid={center.tolist()} radius={radius:.4f} (real radii min={radii.min():.4f} max={radii.max():.4f})")

# The real cameras' position centroid is NOT where they look -- verified
# directly: for dance5885633 the centroid sits at Y=1.876 but the cameras'
# actual viewing rays converge (least-squares ray intersection) at Y=1.222,
# 0.65 units lower. Using the centroid as the look-at target aims the orbit
# camera level/too high and crops to just the head. Triangulate the real
# focus point from all cameras' actual forward directions instead.
forwards = torch.nn.functional.normalize(c2ws[:, :3, 2], dim=1)  # col2 = forward
A = torch.zeros(3, 3)
b = torch.zeros(3)
for p, d in zip(positions, forwards):
    M = torch.eye(3) - torch.outer(d, d)
    A += M
    b += M @ p
focus_point = torch.linalg.solve(A, b)
print(f"triangulated focus point={focus_point.tolist()} (centroid diff={(focus_point - center).tolist()})")

# per-camera real "up" (c2w convention: col1 = down, so up = -col1), averaged
# as unit vectors (valid), not as part of a naive full-matrix average (invalid).
real_down = c2ws[:, :3, 1]
up_hint = F.normalize(-real_down.mean(0), dim=0)
print(f"up_hint={up_hint.tolist()}")

# fit the ring plane: two orthonormal in-plane axes, derived from the first
# real camera's offset from center (projected to be perpendicular to up_hint)
# and its cross product with up_hint.
ref_offset = positions[0] - center
axis1 = F.normalize(ref_offset - (ref_offset @ up_hint) * up_hint, dim=0)
axis2 = F.normalize(torch.cross(up_hint, axis1, dim=0), dim=0)

n = dataset.num_frames
t = torch.linspace(0, 1, n) * 2 * torch.pi
new_positions = center.unsqueeze(0) + radius * (
    torch.cos(t).unsqueeze(1) * axis1.unsqueeze(0) + torch.sin(t).unsqueeze(1) * axis2.unsqueeze(0)
)
look_dirs = focus_point.unsqueeze(0) - new_positions
rotation = look_at(look_dirs, z=up_hint.unsqueeze(0))

c2ws_orbit = torch.tile(torch.eye(4), (n, 1, 1))
c2ws_orbit[:, :3, :3] = rotation
c2ws_orbit[:, :3, 3] = new_positions
w2cs = torch.linalg.inv(c2ws_orbit).cuda()

gs = Gaussians.load(args.run_path / "gaussians.pt")
gs.cuda().eval()

args.output.parent.mkdir(exist_ok=True, parents=True)
with iio.imopen(args.output, "w", plugin="pyav") as out:
    out.init_video_stream("libx264", fps=args.fps or config.data.fps)
    out._video_stream.options = {"crf": "20"}
    render_rgb(
        gs=gs,
        timestamps=dataset[:, 0].time.cuda(),
        out=out,
        w2cs=w2cs,
        shape=(h, w),
        fov=fov,
    )
print(f"wrote {args.output}")
