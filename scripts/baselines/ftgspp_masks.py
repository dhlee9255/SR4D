"""Masks for the FTGS++ transparent-background run, in the export's frame / camera order:
ftgspp_masks.py <unified root> <export_meta.json> <out.npy>  ->  (num_frames, num_cameras, H, W) uint8"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sr4d.data import io  # noqa: E402

root = Path(sys.argv[1]); em = json.loads(Path(sys.argv[2]).read_text())
cams = sorted(em["cam_index"], key=lambda c: em["cam_index"][c])
M = np.stack([np.stack([io.read_mask(root / "masks" / c / f"{f:06d}.png") for c in cams]) for f in em["frames"]])
np.save(sys.argv[3], M); print(sys.argv[3], M.shape)
