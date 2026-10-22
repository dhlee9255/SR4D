"""Rotation / quaternion helpers (wxyz quaternions, matching gsplat)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def rotvec_to_rotmat(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues, safe at 0. v (...,3) -> (...,3,3)."""
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    k = v / theta
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    Kx = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(v.shape[:-1] + (3, 3))
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(Kx.shape)
    R = eye + s * Kx + (1 - c) * (Kx @ Kx)
    # first-order fallback for tiny angles keeps gradients well-defined
    small = (theta2 < 1e-10)[..., None]
    vx, vy, vz = v.unbind(-1)
    V = torch.stack([zero, -vz, vy, vz, zero, -vx, -vy, vx, zero], -1).reshape(Kx.shape)
    return torch.where(small, eye + V, R)


def se3_matrix(rotvec: torch.Tensor, trans: torch.Tensor, center: torch.Tensor | None = None) -> torch.Tensor:
    """4x4 transform rotating about `center` then translating. Shapes (...,3)."""
    R = rotvec_to_rotmat(rotvec)
    T = torch.zeros(rotvec.shape[:-1] + (4, 4), device=rotvec.device, dtype=rotvec.dtype)
    T[..., :3, :3] = R
    t = trans
    if center is not None:
        t = t + center - (R @ center[..., None])[..., 0]
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Robust matrix -> wxyz quaternion."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.stack(
        [
            1 + t,
            1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ],
        -1,
    )
    idx = q.argmax(-1)
    w = torch.stack([1 + t, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1)
    x = torch.stack([m[..., 2, 1] - m[..., 1, 2], q[..., 1], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]], -1)
    y = torch.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], q[..., 2], m[..., 1, 2] + m[..., 2, 1]], -1)
    z = torch.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], q[..., 3]], -1)
    cand = torch.stack([w, x, y, z], -2)
    out = torch.gather(cand, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    return F.normalize(out, dim=-1)


def triangle_frames(v: torch.Tensor, faces: torch.Tensor):
    """Per-triangle orthonormal frame (cols: t1, t2, n) and area. v (B,V,3) or (V,3)."""
    a, b, c = v[..., faces[:, 0], :], v[..., faces[:, 1], :], v[..., faces[:, 2], :]
    e1 = b - a
    e2 = c - a
    n = torch.cross(e1, e2, dim=-1)
    area = 0.5 * n.norm(dim=-1)
    n = F.normalize(n, dim=-1)
    t1 = F.normalize(e1, dim=-1)
    t2 = torch.cross(n, t1, dim=-1)
    return torch.stack([t1, t2, n], dim=-1), area
