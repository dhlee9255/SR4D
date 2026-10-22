"""Multi-view sequence dataset on the SR4D unified format. Preloads uint8 images/masks to GPU."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from sr4d.data import io


class Camera:
    def __init__(self, cid: str, cam: dict, device):
        self.id = cid
        # stored K uses the OpenCV pixel-centre convention; gsplat puts pixel centres at +0.5
        K = cam["K"].copy()
        K[0, 2] += 0.5
        K[1, 2] += 0.5
        self.K = torch.as_tensor(K, dtype=torch.float32, device=device)
        self.w2c = torch.as_tensor(io.w2c_matrix(cam), dtype=torch.float32, device=device)
        self.W, self.H = int(cam["W"]), int(cam["H"])
        self.source = cam.get("source", "real")
        self.center = torch.as_tensor(io.camera_center(cam), dtype=torch.float32, device=device)


class MultiViewSequence:
    """Holds every camera's images for the selected frames.

    frames: absolute frame ids; frame index t (0..T-1) is used everywhere inside the model.
    """

    def __init__(self, root: str | Path, setting: str, frames: list[int] | None = None, device="cuda",
                 cams: list[str] | None = None, load_images: bool = True, background=(1.0, 1.0, 1.0)):
        self.root = Path(root)
        self.meta = io.load_meta(self.root)
        self.split = io.load_split(self.root, setting)
        self.setting = setting
        self.device = device
        all_cams = io.load_cameras(self.root)
        self.frames = list(frames) if frames is not None else list(self.split["frames"])
        self.train_ids = list(self.split["train_cams"])
        self.interp_ids = list(self.split["test_interp_cams"])
        self.extrap_ids = list(self.split["test_extrap_cams"])
        wanted = cams if cams is not None else sorted(set(self.train_ids + self.interp_ids + self.extrap_ids))
        self.cameras = {cid: Camera(cid, all_cams[cid], device) for cid in wanted}
        self.cam_index = {cid: i for i, cid in enumerate(sorted(all_cams))}  # stable per-dataset index (L4 codes)
        self.background = torch.tensor(background, device=device)
        self.images: dict[str, torch.Tensor] = {}
        self.masks: dict[str, torch.Tensor] = {}
        if load_images:
            self._load(wanted)
        self.body_params = [io.load_body(self.root, f) for f in self.frames]

    def _load(self, cam_ids):
        def load_one(args):
            cid, f = args
            img = io.read_image(self.root / "images" / cid / f"{f:06d}.jpg")
            m = io.read_mask(self.root / "masks" / cid / f"{f:06d}.png")
            return cid, f, img, m

        jobs = [(c, f) for c in cam_ids for f in self.frames]
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(load_one, jobs))
        by_cam: dict[str, list] = {c: [None] * len(self.frames) for c in cam_ids}
        by_mask: dict[str, list] = {c: [None] * len(self.frames) for c in cam_ids}
        fidx = {f: i for i, f in enumerate(self.frames)}
        for cid, f, img, m in res:
            by_cam[cid][fidx[f]] = img
            by_mask[cid][fidx[f]] = m
        for c in cam_ids:
            self.images[c] = torch.as_tensor(np.stack(by_cam[c])).to(self.device)  # (T,H,W,3) uint8
            self.masks[c] = torch.as_tensor(np.stack(by_mask[c])).to(self.device)  # (T,H,W) uint8

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def get(self, cid: str, t: int):
        """Returns (rgb composited on background (H,W,3) float, mask (H,W) float, raw rgb)."""
        img = self.images[cid][t].float() / 255.0
        m = (self.masks[cid][t] > 127).float()
        comp = img * m[..., None] + self.background * (1 - m[..., None])
        return comp, m, img

    def sample(self, batch_size: int, gen: torch.Generator | None = None, cam_ids: list[str] | None = None):
        """Random (t, cid) pairs; different frames within a batch (CLAUDE.md 7.5)."""
        cam_ids = cam_ids or self.train_ids
        T = self.num_frames
        ts = torch.randperm(T, generator=gen)[:batch_size].tolist() if batch_size <= T else torch.randint(0, T, (batch_size,), generator=gen).tolist()
        cs = torch.randint(0, len(cam_ids), (batch_size,), generator=gen).tolist()
        return [(t, cam_ids[c]) for t, c in zip(ts, cs)]
