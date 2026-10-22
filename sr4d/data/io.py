"""Read/write helpers for the SR4D unified dataset format (CLAUDE.md 5.2).

Layout:
  <root>/images/<cam_id>/<frame:06d>.jpg
  <root>/masks/<cam_id>/<frame:06d>.png      (0/255)
  <root>/cameras.json   cam_id -> {K, R, t, W, H, dist, source}; R,t are OpenCV world->camera
  <root>/smplx/<frame:06d>.npz               body-model params in world coordinates
  <root>/splits/<setting>.json
  <root>/meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

FORMAT_VERSION = 1


def frame_name(frame: int) -> str:
    return f"{frame:06d}"


def save_cameras(root: Path, cams: dict[str, dict]) -> None:
    out = {}
    for cid, c in cams.items():
        out[str(cid)] = {
            "K": np.asarray(c["K"], dtype=np.float64).tolist(),
            "R": np.asarray(c["R"], dtype=np.float64).tolist(),
            "t": np.asarray(c["t"], dtype=np.float64).reshape(3).tolist(),
            "W": int(c["W"]),
            "H": int(c["H"]),
            "dist": np.asarray(c.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1).tolist(),
            "source": c.get("source", "real"),
        }
    (root / "cameras.json").write_text(json.dumps(out, indent=1))


def load_cameras(root: Path) -> dict[str, dict]:
    raw = json.loads((Path(root) / "cameras.json").read_text())
    cams = {}
    for cid, c in raw.items():
        cams[cid] = {
            "K": np.array(c["K"], dtype=np.float64),
            "R": np.array(c["R"], dtype=np.float64),
            "t": np.array(c["t"], dtype=np.float64),
            "W": c["W"],
            "H": c["H"],
            "dist": np.array(c["dist"], dtype=np.float64),
            "source": c["source"],
        }
    return cams


def camera_center(cam: dict) -> np.ndarray:
    return -cam["R"].T @ cam["t"]


def w2c_matrix(cam: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = cam["R"]
    m[:3, 3] = cam["t"]
    return m


def save_body(root: Path, frame: int, params: dict) -> None:
    d = root / "smplx"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{frame_name(frame)}.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in params.items()})


def load_body(root: Path, frame: int) -> dict:
    with np.load(Path(root) / "smplx" / f"{frame_name(frame)}.npz") as z:
        return {k: z[k] for k in z.files}


def load_meta(root: Path) -> dict:
    return json.loads((Path(root) / "meta.json").read_text())


def load_split(root: Path, setting: str) -> dict:
    return json.loads((Path(root) / "splits" / f"{setting}.json").read_text())


def write_image(path: Path, img_rgb: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 127).astype(np.uint8) * 255)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m


def centered_K(K: np.ndarray, W: int, H: int) -> np.ndarray:
    """Square-pixel pinhole with the principal point at the image centre (OpenCV pixel-centre convention).
    In gsplat / COLMAP corner convention this is cx = W/2, cy = H/2, which is what FTGS++ (SIMPLE_PINHOLE),
    DualGS and 4DGaussians (Inria FoV cameras) assume."""
    f = 0.5 * (K[0, 0] + K[1, 1])
    return np.array([[f, 0, (W - 1) / 2.0], [0, f, (H - 1) / 2.0], [0, 0, 1]], dtype=np.float64)


def undistort_and_resize(img: np.ndarray, K: np.ndarray, dist: np.ndarray, downscale: float, interp=cv2.INTER_AREA,
                         is_mask: bool = False, center: bool = True):
    """Undistort (+ re-centre principal point, square pixels) in ONE remap, then resize by 1/downscale.

    center=False keeps K (DNA-Rendering / ZJU official convention). Returns (image, K_new); K_new is in the
    OpenCV pixel-centre convention (renderers add +0.5 for gsplat's corner convention).
    """
    H0, W0 = img.shape[:2]
    has_dist = dist is not None and np.any(np.abs(dist) > 0)
    if center:
        Kc = centered_K(K, W0, H0)
        if has_dist or not np.allclose(Kc, K):
            mx, my = cv2.initUndistortRectifyMap(K, dist if has_dist else None, None, Kc, (W0, H0), cv2.CV_32FC1)
            img = cv2.remap(img, mx, my, cv2.INTER_NEAREST if is_mask else cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        K = Kc
    elif has_dist:
        img = cv2.undistort(img, K, dist)
    K_new = K.copy()
    if downscale != 1:
        H, W = img.shape[:2]
        Wn, Hn = int(round(W / downscale)), int(round(H / downscale))
        sx, sy = Wn / W, Hn / H
        img = cv2.resize(img, (Wn, Hn), interpolation=cv2.INTER_NEAREST if is_mask else interp)
        K_new[0, :] *= sx
        K_new[1, :] *= sy
        # keep pixel-center convention: (c + 0.5) * s - 0.5
        K_new[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K_new[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return img, K_new
