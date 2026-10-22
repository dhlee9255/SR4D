"""Run FreeTimeGS++'s RoMa point initialisation with ONE fix applied in-memory (no file of the external repo
is modified; docs/DECISIONS.md D-012):

ftgspp/init/points.py:matches_to_points computes cheirality as  w2c[2,:3] . (xyz - w2c[:3,3])  instead of the
camera depth  w2c[2,:3] . xyz + w2c[2,3].  Whether points survive then depends on the world origin: on
ZJU-MoCap 0-9% survive per camera; on the 4DAnyone ring all points from about half of the cameras are dropped.
The replacement below is identical except for the depth test (positive depth in both cameras).

  <ftgs venv python> ftgspp_points_fixed.py <config.toml>
"""
import argparse

import torch
from torch.nn.functional import grid_sample

import ftgspp.init.points as P
from ftgspp.utils.math import chw


def matches_to_points(*, rgb_0, pts_0, pts_1, intrinsic_0, intrinsic_1, w2c_0, w2c_1):
    image_shape = rgb_0.shape[:2]
    projmat0 = intrinsic_0 @ w2c_0[:3]
    projmat1 = intrinsic_1 @ w2c_1[:3]
    normalizer = torch.tensor([[2 / image_shape[1], 0, -1], [0, 2 / image_shape[0], -1], [0, 0, 1]])
    xyz = P.triangulate(
        torch.stack([pts_0 @ normalizer[:2, :2] + normalizer[:2, 2], pts_1 @ normalizer[:2, :2] + normalizer[:2, 2]]),
        torch.stack([normalizer @ projmat0, normalizer @ projmat1]),
    )
    d0 = xyz @ w2c_0[2, :3] + w2c_0[2, 3]
    d1 = xyz @ w2c_1[2, :3] + w2c_1[2, 3]
    valid = (d0 > 0) & (d1 > 0)
    valid *= torch.amax(xyz.abs(), dim=-1) < 1e3
    pts_0_norm = pts_0.clone()
    pts_0_norm[..., 0] /= image_shape[1]
    pts_0_norm[..., 1] /= image_shape[0]
    pts_0_norm = pts_0_norm * 2 - 1
    rgb = grid_sample(chw(rgb_0).reshape(1, 3, *image_shape).float(), pts_0_norm.reshape(1, 1, -1, 2),
                      align_corners=True, padding_mode="border").reshape(3, -1).T
    return xyz[valid], rgb[valid].byte()


if __name__ == "__main__":
    P.matches_to_points = matches_to_points
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    from pathlib import Path
    a = ap.parse_args()
    P.main(argparse.Namespace(config=Path(a.config)))
