"""Shared helpers for baseline exports (all read the SR4D unified format)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.data import io  # noqa: E402,F401


def composite(img: np.ndarray, mask: np.ndarray, bg: int) -> np.ndarray:
    m = (mask > 127)[..., None]
    return np.where(m, img, bg).astype(np.uint8)


def check_centered(cams: dict, cids: list[str]):
    """Baselines assume square pixels and a centred principal point (unified converters guarantee this)."""
    for c in cids:
        K, W, H = cams[c]["K"], cams[c]["W"], cams[c]["H"]
        assert abs(K[0, 0] - K[1, 1]) < 1e-6 * K[0, 0], f"cam {c}: fx != fy"
        assert abs(K[0, 2] - (W - 1) / 2) < 1e-3 and abs(K[1, 2] - (H - 1) / 2) < 1e-3, f"cam {c}: principal point not centred"
