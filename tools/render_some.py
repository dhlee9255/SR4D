"""Render a few cams x frames of an SR4D checkpoint (crop comparisons). usage: PYTHONPATH=. python tools/render_some.py --ckpt C --out DIR --cams g01 g11 --frames 0 15 30 45 59"""
import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
from sr4d.data.dataset import MultiViewSequence
from sr4d.eval.render_cams import render_model
from sr4d.train.trainer import load_model
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cams", nargs="+", required=True); ap.add_argument("--frames", type=int, nargs="+", required=True)
a = ap.parse_args()
st = torch.load(a.ckpt, map_location="cpu", weights_only=False); cfg = OmegaConf.create(st["cfg"])
data = MultiViewSequence(cfg.data.root, cfg.data.setting, list(range(*cfg.data.frames)), load_images=False)
m = load_model(Path(a.ckpt), data, cfg)
render_model(m, data, a.cams, [data.frames.index(f) for f in a.frames], Path(a.out))
print("rendered", a.cams, a.frames)
