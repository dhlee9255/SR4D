"""Side-by-side orbit mp4 (one video -> playback is synchronised by construction) for the experiment artifact.
  PYTHONPATH=. python tools/exp_orbit.py --out <card dir>/orbit_e55.mp4 --cap "55° ..." --methods "label=<dir with 0000.png...>" ...
Appends the file and caption to <card dir>/index.json (orbit, orbit_caps).
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from sr4d.data import io
from tools.cmp_figure import label_bar

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--cap", default=""); ap.add_argument("--methods", nargs="+", required=True)
ap.add_argument("--scale", type=float, default=0.6)
a = ap.parse_args()
dirs = [(l, Path(d)) for l, d in (m.split("=", 1) for m in a.methods)]
n = min(len(list(d.glob("[0-9][0-9][0-9][0-9].png"))) for _, d in dirs)
W0, H0 = (np.array(io.read_image(dirs[0][1] / "0000.png").shape[1::-1]) * a.scale).astype(int) // 2 * 2
bar = label_bar([l for l, _ in dirs], [W0] * len(dirs), 30)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
wr = imageio.get_writer(str(out), fps=24, codec="libx264", quality=None, ffmpeg_params=["-crf", "26", "-pix_fmt", "yuv420p"])
for i in range(n):
    fr = np.concatenate([cv2.resize(io.read_image(d / f"{i:04d}.png"), (W0, H0), interpolation=cv2.INTER_AREA) for _, d in dirs], 1)
    fr = np.concatenate([bar, fr], 0)
    wr.append_data(fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2])
wr.close()
idx = out.parent / "index.json"
I = json.loads(idx.read_text()) if idx.exists() else {}
I.setdefault("orbit", []); I.setdefault("orbit_caps", {})
if out.name not in I["orbit"]:
    I["orbit"].append(out.name)
I["orbit_caps"][out.name] = a.cap
idx.write_text(json.dumps(I, ensure_ascii=False, indent=1))
print("orbit", out, n, "frames")
