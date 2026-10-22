"""Export a trained 4DGaussians model as per-frame Inria PLYs at t = rel / T (the Panoptic training time grid).
Mirrors 4DGaussians/export_perframe_3DGS.py (which iterates test cameras via `.time`, unavailable for
PanopticSports dict cameras). Run in the hust4dgs env with cwd = 4DGaussians.

  python fourdgs_export_frames.py -m <model_path> --num_frames 60 --out <dir>
"""
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.getcwd())
from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.render_utils import get_state_at_time  # noqa: E402


def write_ply(path, xyz, scales, rots, opac, shs, n_dc):
    f_dc = shs[:, :n_dc].transpose(1, 2).flatten(1).cpu().numpy()
    f_rest = shs[:, n_dc:].transpose(1, 2).flatten(1).cpu().numpy()
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(f_dc.shape[1])] + [f"f_rest_{i}" for i in range(f_rest.shape[1])] \
        + ["opacity"] + [f"scale_{i}" for i in range(scales.shape[1])] + [f"rot_{i}" for i in range(rots.shape[1])]
    x = xyz.cpu().numpy()
    attrs = np.concatenate([x, np.zeros_like(x), f_dc, f_rest, opac.cpu().numpy(), scales.cpu().numpy(), rots.cpu().numpy()], 1)
    el = np.empty(x.shape[0], dtype=[(n, "f4") for n in names])
    el[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyper = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", default="", help="official preset used for training (D-038); cfg_args does not keep its merged values")
    args = get_combined_args(parser)
    if getattr(args, "configs", ""):
        from utils.params_utils import merge_hparams
        g_ = {}
        exec(compile(open(args.configs).read(), args.configs, "exec"), g_)
        args = merge_hparams(args, {k: v for k, v in g_.items() if k.endswith("Params")})
    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        ds, hp = model.extract(args), hyper.extract(args)
        g = GaussianModel(ds.sh_degree, hp)
        scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
        for rel in range(args.num_frames):
            v = SimpleNamespace(time=rel / args.num_frames)
            # D-038: 4DGaussians' get_state_at_time returns the RAW opacity and drops opacity_final; with an opacity-deforming
            # preset (dynerf: no_do=False) that leaves Gaussians visible when they should fade. Call the deformation directly.
            t_ = torch.tensor(v.time, device=g.get_xyz.device).repeat(g.get_xyz.shape[0], 1)
            p, s, r, o, sh = g._deformation(g.get_xyz, g._scaling, g._rotation, g._opacity, g.get_features, t_)
            write_ply(os.path.join(args.out, f"time_{rel:05d}.ply"), p, s, r, o, sh, g._features_dc.shape[1])
    print("exported", args.num_frames, "frames ->", args.out)
