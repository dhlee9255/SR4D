"""Write an "orbit" dir (cameras.json + orbit.json) that lists given dataset cameras x frames, so every renderer's --orbit
path can render arbitrary (camera, frame) pairs, e.g. training views for a seen-view comparison.
  python tools/make_camlist.py --root <unified root> --setting <s> --cams g03 g15 --frames 0 30 59 --out <dir>
Output order: for cam in cams: for frame in frames -> index i (renders land at <out>/<i:04d>.png).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--setting", required=True)
ap.add_argument("--cams", nargs="+", required=True); ap.add_argument("--frames", nargs="+", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
root = Path(a.root); cams = io.load_cameras(root); frames = io.load_split(root, a.setting)["frames"]
out, names, rel = {}, [], []
for c in a.cams:
    for f in a.frames:
        k = f"{c}_f{f:03d}"; out[k] = dict(cams[c]); names.append(k); rel.append(frames.index(f))
Path(a.out).mkdir(parents=True, exist_ok=True)
io.save_cameras(Path(a.out), out)
(Path(a.out) / "orbit.json").write_text(json.dumps(dict(cams=names, frame_rel=rel, src_cams=a.cams, frames=a.frames)))
print(len(names), "views ->", a.out)
