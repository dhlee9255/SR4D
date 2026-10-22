"""L3 — per-frame instant residual: latent z_t (R^24) + shared decoder.

Outputs (i) small SE(3) residuals per scaffold joint (axis-angle added to the joint rotation, plus a
global translation) and per free bone, (ii) a per-frame colour affine (3x3 + 3).
Joint residuals are applied inside forward kinematics (equivalent to a pose residual on the scaffold;
docs/DECISIONS.md D-008). Used only in `replay` render mode; dropped with probability p during training.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mean0(x: torch.Tensor) -> torch.Tensor:
    # K=0 (no free bones, V0): mean of an empty tensor is NaN
    return x.mean() if x.numel() else x.sum()


class InstantResidual(nn.Module):
    def __init__(self, n_frames: int, n_joints: int, n_bones: int, cfg):
        super().__init__()
        self.J, self.K = n_joints, n_bones
        self.z = nn.Parameter(torch.randn(n_frames, cfg.z_dim) * 1e-3)
        out = n_joints * 3 + 3 + n_bones * 6 + 12
        self.dec = nn.Sequential(nn.Linear(cfg.z_dim, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, out))
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, t: int):
        o = self.dec(self.z[t])
        J, K = self.J, self.K
        i = 0
        joint_rot = o[i:i + 3 * J].reshape(J, 3) * 0.1; i += 3 * J
        transl = o[i:i + 3] * 0.01; i += 3
        bone_rot = o[i:i + 3 * K].reshape(K, 3) * 0.1; i += 3 * K
        bone_trans = o[i:i + 3 * K].reshape(K, 3) * 0.01; i += 3 * K
        aff = o[i:i + 12]
        A = torch.eye(3, device=o.device) + aff[:9].reshape(3, 3) * 0.1
        b = aff[9:12] * 0.1
        return {"joint_rot": joint_rot, "transl": transl, "bone_rot": bone_rot, "bone_trans": bone_trans, "color_A": A, "color_b": b}

    def mag_loss(self, out: dict) -> torch.Tensor:
        return (out["joint_rot"] ** 2).sum(-1).mean() + (out["transl"] ** 2).sum() * 100 + _mean0((out["bone_rot"] ** 2).sum(-1)) \
            + _mean0((out["bone_trans"] ** 2).sum(-1)) * 100 + ((out["color_A"] - torch.eye(3, device=out["color_A"].device)) ** 2).sum() \
            + (out["color_b"] ** 2).sum()

    def smooth_loss(self, t: int) -> torch.Tensor:
        T = self.z.shape[0]
        nb = [s for s in (t - 1, t + 1) if 0 <= s < T]
        return sum(((self.z[t] - self.z[s]) ** 2).sum() for s in nb) / max(len(nb), 1)
