"""Head-pose check (user 2026-10-13): is the SMPL-X head where the generated face is?
ViTPose-H (GVHMR copy in 4Danyone, read-only; run in the 4danyone env) detects COCO face keypoints (nose, eyes, ears)
on (a) the generated GT and (b) a render; both are compared with the projected SMPL-X face landmark vertices
(smplx vertex_ids: nose 9120, reye 9929, leye 9448, rear 616, lear 6). Error is normalised by the head size in pixels.
If GT keypoints are far from SMPL-X while the render's are close to SMPL-X, the scaffold head pose is off and the
model (tied to the scaffold head) averages the face -> blur.
usage (cwd anywhere): conda activate 4danyone; python tools/face_kp_check.py --root <data> --setting gen48_all \
       --render <run>/renders --frames 0 2 4 ... --out <json> [--device cpu]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

GV = "/data/AI-Graphics/Applications/4Danyone/4DAnyone/third_party/GVHMR"
FACE_V = {"nose": 9120, "leye": 9448, "reye": 9929, "lear": 6, "rear": 616}
COCO = {"nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--setting", required=True); ap.add_argument("--render", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--landmarks", required=True); ap.add_argument("--cams_every", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.root).resolve(); rd = Path(a.render).resolve()
    import importlib.util as iu

    def load(name, path, pkg=False):  # import by file path: hmr4d.utils.preproc/__init__ pulls pytorch3d (absent in 4danyone)
        spec = iu.spec_from_file_location(name, path, submodule_search_locations=[str(Path(path).parent)] if pkg else None)
        mod = iu.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
    vp = load("vitpose_pytorch", f"{GV}/hmr4d/utils/preproc/vitpose_pytorch/__init__.py", pkg=True)
    keypoints_from_heatmaps = load("kp2d_utils", f"{GV}/hmr4d/utils/kpts/kp2d_utils.py").keypoints_from_heatmaps
    MEAN, STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

    def flip_heatmap_coco17(h):  # same as GVHMR hmr4d/utils/geo/flip_utils.py
        b = h.clone()
        for l, r in [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]:
            b[:, l], b[:, r] = h[:, r], h[:, l]
        return b.flip(3)

    def get_batch(imgs, bb):  # same crop as GVHMR get_batch (enlarge 1.0, 256 px, ImageNet norm)
        out = []
        for img, (cx, cy, s) in zip(imgs, bb.numpy()):
            d = s / 256 / 2.0
            if d > 1.1:
                img = cv2.GaussianBlur(img, (5, 5), (d - 1) / 2)
            hs = s / 2
            src = np.float32([[cx - hs, cy - hs], [cx + hs, cy - hs], [cx, cy]])
            dst = np.float32([[0, 0], [255, 0], [127.5, 127.5]])
            out.append(cv2.warpAffine(img, cv2.getAffineTransform(src, dst), (256, 256), flags=cv2.INTER_LINEAR))
        x = torch.from_numpy(np.stack(out)).float()
        return ((x / 255.0 - MEAN) / STD).permute(0, 3, 1, 2), bb
    cwd = os.getcwd(); os.chdir(GV)
    model = vp.build_model("ViTPose_huge_coco_256x192", "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth").to(a.device).eval()
    os.chdir(cwd)
    split = json.load(open(root / f"splits/{a.setting}.json")); cams = json.load(open(root / "cameras.json"))
    LM = json.load(open(a.landmarks))  # "frame/cam" -> 5x2 projected SMPL-X face landmarks (tools/face_landmarks.py)
    tr = split["train_cams"][::a.cams_every]

    @torch.no_grad()
    def detect(imgs, masks):
        boxes = []
        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, max(np.ptp(xs), np.ptp(ys)) * 1.2])
        bb = torch.tensor(boxes, dtype=torch.float32)
        x, _ = get_batch(imgs, bb)
        x = x[:, :, :, 32:224].to(a.device)
        h, hf = model(torch.cat([x, x.flip(3)], 0)).chunk(2)
        h = ((h + flip_heatmap_coco17(hf)) * 0.5).cpu().numpy()
        scale = (torch.cat((bb[:, [2]] * 24 / 32, bb[:, [2]]), 1) / 200).numpy()
        p, c = keypoints_from_heatmaps(heatmaps=h, center=bb[:, :2].numpy(), scale=scale, use_udp=True)
        return np.concatenate([p, c], -1)

    res = {}
    for f in a.frames:
        gts, rens, ms, smp, hs = [], [], [], [], []
        for c in tr:
            uv = np.array(LM[f"{f}/{c}"])
            img = cv2.imread(str(root / f"images/{c}/{f:06d}.jpg"))[..., ::-1]; m = cv2.imread(str(root / f"masks/{c}/{f:06d}.png"), 0) > 127
            gts.append(np.where(m[..., None], img, 255).astype(np.uint8)); ms.append(m)
            rens.append(cv2.imread(str(rd / c / f"{f:06d}.png"))[..., ::-1].copy())
            smp.append(uv); hs.append(np.linalg.norm(uv[3] - uv[4]) + 1e-6)  # ear-ear distance as head size
        kg, kr = detect(gts, ms), detect(rens, ms)
        eg, er, egr = [], [], []
        for i in range(len(tr)):
            for n, j in COCO.items():
                vi = list(FACE_V).index(n)
                if kg[i, j, 2] > 0.5 and kr[i, j, 2] > 0.5:
                    eg.append(np.linalg.norm(kg[i, j, :2] - smp[i][vi]) / hs[i])
                    er.append(np.linalg.norm(kr[i, j, :2] - smp[i][vi]) / hs[i])
                    egr.append(np.linalg.norm(kg[i, j, :2] - kr[i, j, :2]) / hs[i])
        res[f] = dict(gt_vs_smplx=float(np.median(eg)), render_vs_smplx=float(np.median(er)), gt_vs_render=float(np.median(egr)), n=len(eg))
        print(f, res[f], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
