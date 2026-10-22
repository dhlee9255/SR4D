"""Common evaluation for every method (CLAUDE.md 8). Reads renders/<cam>/<frame>.png produced by any method
and compares against the unified dataset GT composited on white with the GT mask.

  python -m sr4d.eval.metrics --data_root data_dev/zju_mocap/387 --setting real12 \
      --renders outputs/sr4d/387/real12/renders --method sr4d [--train_min 123 --n_gauss 300000 --notes ...]
Writes <renders>/../metrics_<renders name>.json and appends rows to results/<phase>_<setting>.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from pytorch_msssim import ssim

from sr4d.data import io
from sr4d.eval.temporal import FlowWarp, temporal_error

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["method", "seq", "setting", "split", "psnr", "ssim", "lpips", "fg_psnr", "fg_ssim", "fg_lpips", "temporal",
           "train_min", "n_gauss", "notes"]


def to_t(x):
    return torch.as_tensor(x).float().cuda() / 255.0


def bbox(mask, pad=16):
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    if len(ys) == 0:
        return 0, H, 0, W
    return max(ys.min() - pad, 0), min(ys.max() + pad + 1, H), max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)


@torch.no_grad()
def evaluate(data_root: Path, setting: str, renders: Path, frames=None, splits=("test_interp", "test_extrap"), tof_stride: int = 5):
    split = io.load_split(data_root, setting)
    frames = frames or split["frames"]
    fn = lpips.LPIPS(net="vgg", verbose=False).cuda().eval()
    fw = FlowWarp()
    per = {}
    rows = []
    for sname in splits:
        cams = split[f"{sname}_cams"]
        acc = {k: [] for k in ("psnr", "ssim", "lpips", "fg_psnr", "fg_ssim", "fg_lpips", "temporal", "fgi_psnr", "band_psnr", "tof", "oom_alpha")}
        per_cam = {}
        for cid in cams:
            cam_ps = []
            prev = None
            hist = []  # (R, G, mask) history for the strided tOF
            for f in frames:
                rp = renders / cid / f"{f:06d}.png"
                if not rp.exists():
                    raise FileNotFoundError(f"missing render {rp}")
                img = io.read_image(data_root / "images" / cid / f"{f:06d}.jpg")
                m = io.read_mask(data_root / "masks" / cid / f"{f:06d}.png") > 127
                gt = np.where(m[..., None], img, 255).astype(np.uint8)
                pr = io.read_image(rp)
                if pr.shape != gt.shape:
                    raise ValueError(f"render {rp} shape {pr.shape} != GT {gt.shape}")
                G, R = to_t(gt), to_t(pr)
                mse = ((G - R) ** 2).mean()
                acc["psnr"].append(float(-10 * torch.log10(mse)))
                Gc, Rc = G.permute(2, 0, 1)[None], R.permute(2, 0, 1)[None]
                acc["ssim"].append(float(ssim(Rc, Gc, data_range=1.0)))
                acc["lpips"].append(float(fn(Rc * 2 - 1, Gc * 2 - 1)))
                mt = torch.as_tensor(m).cuda()
                fg_mse = (((G - R) ** 2).mean(-1) * mt).sum() / mt.sum().clamp_min(1)
                acc["fg_psnr"].append(float(-10 * torch.log10(fg_mse.clamp_min(1e-10))))
                # interior foreground (mask eroded by 3 px): insensitive to inconsistent mask boundaries (diagnostic, JSON only)
                mi = torch.as_tensor(cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0).cuda()
                fgi_mse = (((G - R) ** 2).mean(-1) * mi).sum() / mi.sum().clamp_min(1)
                acc["fgi_psnr"].append(float(-10 * torch.log10(fgi_mse.clamp_min(1e-10))))
                # boundary band: GT silhouette +-3 px (D-032; interior = fgi above)
                bd = torch.as_tensor((cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~(cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)).cuda()
                bd_mse = (((G - R) ** 2).mean(-1) * bd).sum() / bd.sum().clamp_min(1)
                acc["band_psnr"].append(float(-10 * torch.log10(bd_mse.clamp_min(1e-10))))
                ap = rp.with_suffix(".alpha.png")
                if ap.exists():  # floaters: mean accumulated alpha outside the GT mask dilated by 5 px
                    al = cv2.imread(str(ap), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    out = cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)) == 0
                    acc["oom_alpha"].append(float(al[out].mean()))
                y0, y1, x0, x1 = bbox(m)
                Gb, Rb = Gc[..., y0:y1, x0:x1], Rc[..., y0:y1, x0:x1]
                acc["fg_ssim"].append(float(ssim(Rb, Gb, data_range=1.0)) if min(y1 - y0, x1 - x0) > 11 else float("nan"))
                acc["fg_lpips"].append(float(fn(Rb * 2 - 1, Gb * 2 - 1)))
                if prev is not None:
                    acc["temporal"].append(temporal_error(R, G, prev[0], prev[1], mt | prev[2]))
                prev = (R, G, mt)
                hist.append(prev)
                if len(hist) > tof_stride:
                    h = hist.pop(0)
                    acc["tof"].append(fw.tof(R, h[0], G, h[1], mt | h[2]))
                cam_ps.append(acc["psnr"][-1])
            per_cam[cid] = {"psnr": float(np.mean(cam_ps)),
                            "nearest_angle": split["nearest_train"][cid]["angle_deg"]}
        res = {k: float(np.nanmean(v)) if v else float("nan") for k, v in acc.items()}
        res["per_cam"] = per_cam
        per[sname] = res
    return per


def append_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--phase", default="A")
    ap.add_argument("--train_min", default="")
    ap.add_argument("--n_gauss", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no_csv", action="store_true")
    ap.add_argument("--tof_stride", type=int, default=5, help="frame stride for tOF (ZJU-387 moves only ~0.45 px/frame)")
    a = ap.parse_args()
    data_root = Path(a.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    renders = Path(a.renders)
    per = evaluate(data_root, a.setting, renders, tof_stride=a.tof_stride)
    seq = io.load_meta(data_root).get("seq", data_root.name)
    out_json = renders.parent / f"metrics_{renders.name}.json"
    out_json.write_text(json.dumps(per, indent=1))
    rows = []
    for s, r in per.items():
        rows.append(dict(method=a.method, seq=seq, setting=a.setting, split=s, train_min=a.train_min, n_gauss=a.n_gauss,
                         notes=a.notes, **{k: f"{v:.4f}" for k, v in r.items() if k != "per_cam"}))
        print(f"{a.method:10s} {s:12s} PSNR {r['psnr']:.3f} SSIM {r['ssim']:.4f} LPIPS {r['lpips']:.4f} | "
              f"fg PSNR {r['fg_psnr']:.3f} SSIM {r['fg_ssim']:.4f} LPIPS {r['fg_lpips']:.4f} | temporal {r['temporal']:.5f} | fg-interior PSNR {r['fgi_psnr']:.3f} | band PSNR {r.get('band_psnr', float('nan')):.3f} | tOF@stride {r['tof']:.3f}px | oom-alpha {r['oom_alpha']:.4f}")
    if not a.no_csv:
        append_csv(ROOT / "results" / f"{a.phase}_{a.setting}.csv", rows)


if __name__ == "__main__":
    main()
