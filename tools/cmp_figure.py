"""10/15 comparison figures for one dataset (gen48_ho8 models).

NVS: every held-out view x frames (0, 30, 59); seen: the 4 training views of outputs/orbits/cmp1015/<k>/train4.
Each cell = full image + face / hair / body-texture crops (boxes from the SMPL-X joints), with an |error| row under it
(error x3, inferno) and the masked fg PSNR of the full image printed in the cell. One JPEG per view.
Orbit videos: methods side by side in ONE mp4 per elevation (so playback is synchronised by construction).
  PYTHONPATH=. python tools/cmp_figure.py --k dance --out <dir> [--methods "label=kind:path" ...]
kind: held = <path>/renders/<cam>/<frame>.png, orbit sub-dir name = <path> under outputs/orbits/cmp1015/<k>/<orbit>/.
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sr4d.data import io
from sr4d.model.body_model import BodyModel, params_to_tensors

ROOTS = {"dance": "data_dev/fouranyone_gen/dance5885633_gen48_2ring", "zju": "data_dev/zju_mocap_gen/387_gen48_2ring"}
FRAMES = [0, 30, 59]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FW, FH, CS = 176, 320, 106     # full image cell (half resolution) and crop size


def label_bar(names, widths, h=34):
    im = Image.new("RGB", (sum(widths), h), (255, 255, 255)); d = ImageDraw.Draw(im); f = ImageFont.truetype(FONT, 20)
    x = 0
    for n, w in zip(names, widths):
        tw = d.textlength(n, font=f); d.text((x + (w - tw) / 2, 4), n, fill=(20, 20, 20), font=f); x += w
    return np.asarray(im)


def put(img, txt, size=17):
    im = Image.fromarray(img); d = ImageDraw.Draw(im); f = ImageFont.truetype(FONT, size)
    d.rectangle([0, 0, d.textlength(txt, font=f) + 8, size + 8], fill=(255, 255, 255)); d.text((4, 2), txt, fill=(0, 0, 0), font=f)
    return np.asarray(im)


def cut(img, cx, cy, s):
    H, W = img.shape[:2]; x0, y0 = int(cx - s / 2), int(cy - s / 2)
    pad = np.full((s, s) + img.shape[2:], 255 if img.ndim == 3 else 0, img.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    if xb > xa and yb > ya:
        pad[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return pad


def errmap(a, b):
    e = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1) * 3
    return cv2.cvtColor(cv2.applyColorMap(np.clip(e, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def cell(img, gt, boxes, m, show_psnr):
    crops = [cv2.resize(cut(img, *b), (CS, CS), interpolation=cv2.INTER_AREA) for b in boxes]
    full = cv2.resize(img, (FW, FH), interpolation=cv2.INTER_AREA)
    top = np.concatenate([full, np.concatenate(crops + [np.full((FH - 3 * CS, CS, 3), 255, np.uint8)], 0)], 1)
    if gt is None:
        bot = np.full_like(top, 255)
    else:
        if show_psnr:
            e2 = ((img.astype(np.float32) - gt) / 255) ** 2
            top = put(top, f"{-10 * np.log10(max(e2.mean(-1)[m].mean(), 1e-10)):.2f} dB")
        ec = [cv2.resize(errmap(cut(img, *b), cut(gt, *b)), (CS, CS), interpolation=cv2.INTER_AREA) for b in boxes]
        bot = np.concatenate([cv2.resize(errmap(img, gt), (FW, FH), interpolation=cv2.INTER_AREA),
                              np.concatenate(ec + [np.full((FH - 3 * CS, CS, 3), 255, np.uint8)], 0)], 1)
    return np.concatenate([top, bot], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", required=True); ap.add_argument("--out", required=True); ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--orbit_scale", type=float, default=0.6)
    a = ap.parse_args()
    R = Path(ROOTS[a.k]); O = Path("outputs/orbits/cmp1015") / a.k; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meths = [(lab, *spec.split(":", 1)) for lab, spec in (m.split("=", 1) for m in a.methods)]   # (label, held path, orbit sub)
    cams = io.load_cameras(R); meta = io.load_meta(R); split = io.load_split(R, "gen48_ho8")
    bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
    tr4 = json.loads((O / "train4/orbit.json").read_text())
    third = "점무늬" if a.k == "dance" else "몸통"

    def boxes(c, f):
        b, p, t, _ = params_to_tensors([io.load_body(R, f)])
        with torch.no_grad():
            J = bm(b, p, t)[1][0].numpy()
        Rm, tt, K = np.array(cams[c]["R"]), np.array(cams[c]["t"]).reshape(3), np.array(cams[c]["K"])
        uv = lambda j: (lambda x: x[:2] / x[2])(K @ (Rm @ J[j] + tt))  # noqa: E731
        hl = max(np.linalg.norm(uv(15) - uv(12)), 8); h = uv(15)
        return [(h[0], h[1] + 0.15 * hl, int(1.4 * hl)), (h[0], h[1] - 0.7 * hl, int(2.6 * hl)), (*uv(3 if a.k == "dance" else 6), 80)]

    names = ["생성 GT"] + [m[0] for m in meths]; widths = [FW + CS] * len(names)
    index = {"held": [], "seen": [], "orbit": []}
    for grp, views in (("held", split["test_interp_cams"]), ("seen", tr4["src_cams"])):
        for c in views:
            rows = [label_bar(names, widths)]
            for fi, f in enumerate(FRAMES):
                img = io.read_image(R / f"images/{c}/{f:06d}.jpg"); m = io.read_mask(R / f"masks/{c}/{f:06d}.png") > 127
                gt = np.where(m[..., None], img, 255).astype(np.uint8); bx = boxes(c, f)
                row = [cell(gt, None, bx, m, False)]
                for lab, held, orb in meths:
                    if grp == "held":
                        p = Path(held) / "renders" / c / f"{f:06d}.png"
                    else:
                        p = O / "train4" / orb / f"{tr4['src_cams'].index(c) * len(tr4['frames']) + tr4['frames'].index(f):04d}.png"
                    r = io.read_image(p) if p.exists() else None
                    row.append(cell(r, gt, bx, m, True) if r is not None else np.full((2 * FH, FW + CS, 3), 160, np.uint8))
                rows.append(put(np.concatenate(row, 1), f"{c} · frame {f}", 18))
            name = f"{grp}_{c}.jpg"
            cv2.imwrite(str(out / name), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 78])
            index[grp].append(name)
    for orb in ("orbit_e55", "orbit_e05"):
        sched = json.loads((O / orb / "orbit.json").read_text()); n = len(sched["cams"])
        dirs = [(lab, O / orb / sub) for lab, _, sub in meths]
        if not all((d / "0000.png").exists() for _, d in dirs):
            print("missing orbit renders", orb, [lab for lab, d in dirs if not (d / "0000.png").exists()]); continue
        W0, H0 = (np.array(io.read_image(dirs[0][1] / "0000.png").shape[1::-1]) * a.orbit_scale).astype(int) // 2 * 2
        bar = label_bar([lab for lab, _ in dirs], [W0] * len(dirs), 30)
        wr = imageio.get_writer(str(out / f"{orb}.mp4"), fps=24, codec="libx264", quality=None, ffmpeg_params=["-crf", "24", "-pix_fmt", "yuv420p"])
        for i in range(n):
            fr = np.concatenate([cv2.resize(io.read_image(d / f"{i:04d}.png"), (W0, H0), interpolation=cv2.INTER_AREA) for _, d in dirs], 1)
            fr = np.concatenate([bar, fr], 0)
            wr.append_data(fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2])
        wr.close(); index["orbit"].append(f"{orb}.mp4")
    (out / "index.json").write_text(json.dumps(index))
    print(json.dumps(index))


if __name__ == "__main__":
    main()
