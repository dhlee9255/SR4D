"""Thin SMPL / SMPL-X wrapper exposing the buffers SR4D needs (own LBS, no pose blendshapes by default).

Unified per-frame parameters (stored in <root>/smplx/<frame>.npz):
  betas (10,), full_pose (J,3) axis-angle *absolute* (hand mean already added), transl (3,), expression (10,) optional.
Joint order follows the body model's kintree (SMPL: 24, SMPL-X: 55).
Vertex formula (same as smplx): v = sum_j w_j A_j(full_pose, betas) [v_shaped; 1] + transl
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import smplx
import torch
import torch.nn as nn
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "assets" / "body_models"

NUM_JOINTS = {"smpl": 24, "smplx": 55}
# body joints used for the pose code (excludes global orient, hands, face)
BODY_JOINTS = {"smpl": list(range(1, 24)), "smplx": list(range(1, 22))}


class BodyModel(nn.Module):
    def __init__(self, model_type: str = "smplx", gender: str = "neutral", model_dir: str | Path | None = None, num_betas: int = 10):
        super().__init__()
        model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.model_type = model_type
        kw = dict(gender=gender, num_betas=num_betas)
        if model_type == "smplx":
            kw.update(ext="npz", use_pca=False, flat_hand_mean=True, num_expression_coeffs=10)
        else:
            kw.update(ext="pkl")
        m = smplx.create(str(model_dir), model_type=model_type, **kw)
        self.num_joints = NUM_JOINTS[model_type]
        self.register_buffer("v_template", m.v_template.float().clone())
        self.register_buffer("shapedirs", m.shapedirs[..., :num_betas].float().clone())
        if model_type == "smplx":
            self.register_buffer("exprdirs", m.expr_dirs.float().clone())
        self.register_buffer("posedirs", m.posedirs.float().clone())
        self.register_buffer("J_regressor", m.J_regressor.float().clone())
        self.register_buffer("lbs_weights", m.lbs_weights.float().clone())
        self.register_buffer("parents", m.parents.long().clone())
        self.register_buffer("faces", torch.as_tensor(np.asarray(m.faces, dtype=np.int64)))
        if model_type == "smplx":
            # hand means, for converting datasets stored with flat_hand_mean=False (e.g. DNA-Rendering)
            mm = smplx.create(str(model_dir), model_type="smplx", gender=gender, ext="npz", use_pca=False, flat_hand_mean=False)
            self.register_buffer("left_hand_mean", mm.left_hand_mean.float().reshape(15, 3).clone())
            self.register_buffer("right_hand_mean", mm.right_hand_mean.float().reshape(15, 3).clone())
            del mm
        del m

    @property
    def num_verts(self) -> int:
        return self.v_template.shape[0]

    def shaped(self, betas: torch.Tensor, expression: torch.Tensor | None = None):
        """betas (B,10) -> v_shaped (B,V,3), J (B,J,3)."""
        v = self.v_template[None] + blend_shapes(betas, self.shapedirs)
        if expression is not None and self.model_type == "smplx":
            v = v + blend_shapes(expression, self.exprdirs)
        J = vertices2joints(self.J_regressor, v)
        return v, J

    def joint_transforms(self, full_pose: torch.Tensor, J: torch.Tensor):
        """full_pose (B,J,3), J rest joints (B,J,3) -> A (B,J,4,4) mapping rest(T-pose) points to posed (no transl)."""
        B = full_pose.shape[0]
        rot = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(B, -1, 3, 3)
        _, A = batch_rigid_transform(rot, J, self.parents)
        return A, rot

    def forward(self, betas, full_pose, transl, expression=None, pose_blend: bool = True):
        v_shaped, J = self.shaped(betas, expression)
        A, rot = self.joint_transforms(full_pose, J)
        v = v_shaped
        if pose_blend:
            B = full_pose.shape[0]
            ident = torch.eye(3, device=rot.device, dtype=rot.dtype)
            pose_feature = (rot[:, 1:] - ident).reshape(B, -1)
            v = v + torch.matmul(pose_feature, self.posedirs).reshape(B, -1, 3)
        T = torch.einsum("vj,bjxy->bvxy", self.lbs_weights, A)
        vh = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        verts = torch.einsum("bvxy,bvy->bvx", T, vh)[..., :3] + transl[:, None]
        # posed joint = A·J_rest (A maps rest points to posed; its translation column alone is NOT the joint, D-019)
        joints = torch.einsum("bjxy,bjy->bjx", A[..., :3, :3], J) + A[..., :3, 3] + transl[:, None]
        return verts, joints, A


def canonical_pose(model_type: str, kind: str = "da", device="cpu") -> torch.Tensor:
    """Canonical full_pose (J,3). 'da' = legs apart (~30deg) like SAGA/GauHuman big-pose, 't' = zeros."""
    J = NUM_JOINTS[model_type]
    p = torch.zeros(J, 3, device=device)
    if kind == "da":
        p[1, 2] = np.deg2rad(30.0)  # left hip
        p[2, 2] = -np.deg2rad(30.0)  # right hip
    elif kind != "t":
        raise ValueError(kind)
    return p


def params_to_tensors(params: list[dict], device="cpu"):
    """List of per-frame unified dicts -> stacked tensors."""
    betas = torch.as_tensor(np.stack([p["betas"].reshape(-1)[:10] for p in params])).float().to(device)
    full_pose = torch.as_tensor(np.stack([p["full_pose"] for p in params])).float().to(device)
    transl = torch.as_tensor(np.stack([p["transl"].reshape(3) for p in params])).float().to(device)
    expr = None
    if all("expression" in p for p in params):
        expr = torch.as_tensor(np.stack([p["expression"].reshape(-1)[:10] for p in params])).float().to(device)
    return betas, full_pose, transl, expr
