"""L0 — SMPL(-X) scaffold. Not rendered; provides canonical mesh, skinning weights and per-frame joint transforms.

G_t[j] maps a canonical-space point (canonical pose, e.g. da-pose) to frame t's world space:
    G_t[j] = [A(theta_t)[j] @ A(theta_c)[j]^-1] + transl_t

Pose correction modes (D-025, B-stage):
    free   — per-frame pose / transl parameters (original)
    global — pose fixed to the input (GVHMR) poses; only betas and one sequence-shared translation offset are learned
    spline — pose / transl = input + uniform cubic B-spline correction (control point every `spline_step` frames)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sr4d.model.body_model import BODY_JOINTS, BodyModel, canonical_pose, params_to_tensors


HEAD_NECK = [12, 15]  # SMPL-X joints: neck, head


def scaffold_kw(cfg) -> dict:
    """Scaffold pose-correction mode (D-025): model.scaffold_pose = free | global | spline | global_headspline (D-033)."""
    return dict(pose_mode=cfg.model.get("scaffold_pose", "free"), spline_step=int(cfg.model.get("scaffold_spline_step", 4)))


class Scaffold(nn.Module):
    def __init__(self, body_model: BodyModel, body_params: list[dict], canon_kind: str = "da", pose_mode: str = "free",
                 spline_step: int = 4):
        super().__init__()
        self.bm = body_model
        betas, pose, transl, expr = params_to_tensors(body_params)
        self.T = pose.shape[0]
        self.betas = nn.Parameter(betas.mean(0, keepdim=True))
        self.pose_mode, self.spline_step = pose_mode, spline_step
        if pose_mode == "free":
            self.pose_p = nn.Parameter(pose.clone())  # (T,J,3)
            self.transl_p = nn.Parameter(transl.clone())  # (T,3)
        elif pose_mode == "global":
            self.transl_off = nn.Parameter(torch.zeros(1, 3))
        elif pose_mode == "global_headspline":  # D-033: body pose fixed (as global), neck + head get a temporal B-spline
            nc = (self.T - 1) // spline_step + 4
            self.transl_off = nn.Parameter(torch.zeros(1, 3))
            self.register_buffer("spline_joints", torch.tensor(HEAD_NECK))
            self.pose_ctrl = nn.Parameter(torch.zeros(nc, len(HEAD_NECK), 3))
        elif pose_mode == "spline":
            nc = (self.T - 1) // spline_step + 4
            self.pose_ctrl = nn.Parameter(torch.zeros(nc, pose.shape[1], 3))
            self.transl_ctrl = nn.Parameter(torch.zeros(nc, 3))
        else:
            raise ValueError(pose_mode)
        self.register_buffer("pose_init", pose.clone())
        self.register_buffer("transl_init", transl.clone())
        self.register_buffer("betas_init", betas.mean(0, keepdim=True).clone())
        self.register_buffer("canon_pose", canonical_pose(body_model.model_type, canon_kind)[None])
        self.body_joint_ids = BODY_JOINTS[body_model.model_type]
        self._canon_cache = None

    def _load_from_state_dict(self, sd, prefix, *args, **kw):
        # checkpoints before D-025 stored the per-frame parameters as "pose" / "transl"
        for old, new in (("pose", "pose_p"), ("transl", "transl_p")):
            if prefix + old in sd and prefix + new not in sd:
                sd[prefix + new] = sd.pop(prefix + old)
        super()._load_from_state_dict(sd, prefix, *args, **kw)

    def _spline_w(self, dev):
        """(T, nc) uniform cubic B-spline basis matrix."""
        T, k = self.T, self.spline_step
        W = torch.zeros(T, (T - 1) // k + 4, device=dev)
        for t in range(T):
            u = t / k; i = int(u); f = u - i
            W[t, i:i + 4] = torch.tensor([(1 - f) ** 3, 3 * f ** 3 - 6 * f ** 2 + 4, -3 * f ** 3 + 3 * f ** 2 + 3 * f + 1, f ** 3], device=dev) / 6
        return W

    @property
    def pose(self):  # (T,J,3) current pose for every frame
        if self.pose_mode == "free":
            return self.pose_p
        if self.pose_mode == "global":
            return self.pose_init
        if self.pose_mode == "global_headspline":
            d = torch.einsum("tc,cjx->tjx", self._spline_w(self.pose_ctrl.device), self.pose_ctrl)
            return self.pose_init.index_add(1, self.spline_joints, d)
        return self.pose_init + torch.einsum("tc,cjx->tjx", self._spline_w(self.pose_ctrl.device), self.pose_ctrl)

    @property
    def transl(self):  # (T,3)
        if self.pose_mode == "free":
            return self.transl_p
        if self.pose_mode in ("global", "global_headspline"):
            return self.transl_init + self.transl_off
        return self.transl_init + self._spline_w(self.transl_ctrl.device) @ self.transl_ctrl

    def pose_params(self):
        return {"free": lambda: [self.pose_p], "global": lambda: [], "spline": lambda: [self.pose_ctrl],
                "global_headspline": lambda: [self.pose_ctrl]}[self.pose_mode]()

    def transl_params(self):
        return {"free": lambda: [self.transl_p], "global": lambda: [self.transl_off], "spline": lambda: [self.transl_ctrl],
                "global_headspline": lambda: [self.transl_off]}[self.pose_mode]()

    def reset_to_input(self):
        """Set the pose correction to zero (input poses); returns a snapshot for restore()."""
        snap = [q.data.clone() for q in self.pose_params() + self.transl_params()]
        if self.pose_mode == "free":
            self.pose_p.data.copy_(self.pose_init); self.transl_p.data.copy_(self.transl_init)
        else:
            for q in self.pose_params() + self.transl_params():
                q.data.zero_()
        self.invalidate()
        return snap

    def restore(self, snap):
        for q, v in zip(self.pose_params() + self.transl_params(), snap):
            q.data.copy_(v)
        self.invalidate()

    @property
    def faces(self):
        return self.bm.faces

    @property
    def lbs_weights(self):
        return self.bm.lbs_weights

    @property
    def num_joints(self):
        return self.bm.num_joints

    def invalidate(self):
        self._canon_cache = None

    def canonical(self):
        """Canonical vertices (V,3), joints (J,3), canonical joint transforms A_c (J,4,4)."""
        if self._canon_cache is not None and not self.betas.requires_grad:
            return self._canon_cache
        v_shaped, J = self.bm.shaped(self.betas)
        A_c, _ = self.bm.joint_transforms(self.canon_pose, J)
        Tv = torch.einsum("vj,bjxy->bvxy", self.bm.lbs_weights, A_c)
        vh = torch.cat([v_shaped, torch.ones_like(v_shaped[..., :1])], -1)
        v_c = torch.einsum("bvxy,bvy->bvx", Tv, vh)[0, :, :3]
        j_c = torch.einsum("jxy,jy->jx", A_c[0, :, :3, :3], J[0]) + A_c[0, :, :3, 3]  # canonical joint positions (D-019)
        out = (v_c, j_c, A_c[0], J[0])
        if not self.betas.requires_grad:
            self._canon_cache = tuple(o.detach() for o in out)
        return out

    def frame_transforms(self, t: torch.Tensor | int, pose_delta=None, transl_delta=None):
        """Per-joint canonical->frame transforms G (B,J,4,4) for frame indices t."""
        t = torch.as_tensor(t, device=self.pose_init.device).reshape(-1)
        pose = self.pose[t]
        transl = self.transl[t]
        if pose_delta is not None:
            pose = pose + pose_delta
        if transl_delta is not None:
            transl = transl + transl_delta
        _, _, A_c, J = self.canonical()
        A_t, _ = self.bm.joint_transforms(pose, J[None].expand(pose.shape[0], -1, -1))
        G = A_t @ torch.linalg.inv(A_c)[None]
        G = G.clone()
        G[..., :3, 3] = G[..., :3, 3] + transl[:, None]
        return G

    def posed_vertices(self, G: torch.Tensor, v_c: torch.Tensor | None = None):
        """LBS of canonical vertices with G (B,J,4,4) -> (B,V,3)."""
        if v_c is None:
            v_c = self.canonical()[0]
        Tv = torch.einsum("vj,bjxy->bvxy", self.bm.lbs_weights, G)
        return torch.einsum("bvxy,vy->bvx", Tv[..., :3, :3], v_c) + Tv[..., :3, 3]

    def pose_feature(self, t: torch.Tensor | int) -> torch.Tensor:
        """Body pose (axis-angle of body joints, B x 3*nb) used as the pose code input of L2."""
        t = torch.as_tensor(t, device=self.pose_init.device).reshape(-1)
        return self.pose[t][:, self.body_joint_ids].reshape(t.shape[0], -1)

    def smooth_loss(self, t: int) -> torch.Tensor:
        """Second-order temporal smoothness (acceleration) of pose / translation around frame t (optional; D-017)."""
        if self.T < 3:
            return self.betas.sum() * 0
        t = min(max(t, 1), self.T - 2)
        P, Tr = self.pose, self.transl
        acc_p = P[t - 1] - 2 * P[t] + P[t + 1]
        acc_t = Tr[t - 1] - 2 * Tr[t] + Tr[t + 1]
        return (acc_p ** 2).sum(-1).mean() + (acc_t ** 2).sum() * 100.0

    def prior_loss(self):
        return ((self.pose - self.pose_init) ** 2).sum(-1).mean() + ((self.transl - self.transl_init) ** 2).sum(-1).mean() * 10 \
            + ((self.betas - self.betas_init) ** 2).mean() * 0.1


def sample_surface(v: torch.Tensor, faces: torch.Tensor, n: int, gen: torch.Generator | None = None):
    """Area-proportional sampling. Returns (tri_id (n,), bary (n,3))."""
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    area = 0.5 * torch.cross(b - a, c - a, dim=-1).norm(dim=-1)
    tri = torch.multinomial(area / area.sum(), n, replacement=True, generator=gen)
    r1 = torch.rand(n, device=v.device, generator=gen)
    r2 = torch.rand(n, device=v.device, generator=gen)
    s = r1.sqrt()
    bary = torch.stack([1 - s, s * (1 - r2), s * r2], -1)
    return tri, bary


def nearest_vertex(points: torch.Tensor, verts: torch.Tensor, chunk: int = 8192):
    """Brute-force nearest vertex (GPU, chunked). Returns (idx (N,), dist (N,))."""
    idx, dist = [], []
    for i in range(0, points.shape[0], chunk):
        d = torch.cdist(points[i:i + chunk], verts)
        dd, ii = d.min(1)
        idx.append(ii)
        dist.append(dd)
    if not idx:
        return torch.zeros(0, dtype=torch.long, device=points.device), torch.zeros(0, device=points.device)
    return torch.cat(idx), torch.cat(dist)


def dense_surface_points(v: torch.Tensor, faces: torch.Tensor, n: int = 200000, seed: int = 0):
    g = torch.Generator(device=v.device).manual_seed(seed)
    tri, bary = sample_surface(v, faces, n, g)
    return (v[faces[tri]] * bary[..., None]).sum(1)

