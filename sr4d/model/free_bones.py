"""Free bones: K Gaussian bones (k-means of free Gaussians' canonical points) with per-frame SE(3)
relative to the parent SMPL(-X) joint frame (applied in canonical space before LBS, so it is relative to
the parent joint). Skinning weight = softmax of negative squared Mahalanobis distance to each bone.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from sr4d.utils.geometry import se3_matrix


def kmeans(x: torch.Tensor, k: int, iters: int = 30, seed: int = 0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    a = torch.cdist(x, c).argmin(1)
    return c, a


BONE_MODES = ("free", "pose_mlp", "spline", "mlp_spline")  # D-023: free = per-frame independent SE(3) (jitters)


class FreeBones(nn.Module):
    """mode (D-023):
      free       : per-frame independent SE(3) (original; jitters, kept for reference)
      pose_mlp   : V1  B_t = MLP(pose_t, pose_t - pose_{t-1}, pose_t - pose_{t-2})
      spline     : V2  uniform cubic B-spline over time, control point every `spline_step` frames
      mlp_spline : V3  V1 + V2 as a small residual (L2-penalised)
    """

    def __init__(self, points: torch.Tensor, n_bones: int, n_frames: int, joints_c: torch.Tensor | None,
                 mode: str = "free", pose_dim: int = 63, spline_step: int = 4, hidden: int = 128):
        super().__init__()
        assert mode in BONE_MODES, mode
        self.mode = mode
        self.n_frames = n_frames
        self.spline_step = spline_step
        K = max(1, min(n_bones, points.shape[0] // 50)) if points.shape[0] > 0 else 1
        if points.shape[0] >= K:
            centers, assign = kmeans(points, K)
            std = torch.stack([points[assign == j].std(0) if (assign == j).sum() > 2 else torch.full((3,), 0.05, device=points.device)
                               for j in range(K)])
        else:
            centers = torch.zeros(K, 3, device=points.device)
            std = torch.full((K, 3), 0.1, device=points.device)
        self.K = K
        self.register_buffer("centers", centers)
        self.log_scale = nn.Parameter(torch.log(std.clamp(0.02, 0.5) * 1.5))
        dev = points.device
        if mode == "free":
            self.rotvec = nn.Parameter(torch.zeros(n_frames, K, 3, device=dev))
            self.trans = nn.Parameter(torch.zeros(n_frames, K, 3, device=dev))
        if mode in ("pose_mlp", "mlp_spline"):
            self.mlp = nn.Sequential(nn.Linear(3 * pose_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, K * 6)).to(dev)
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        if mode in ("spline", "mlp_spline"):
            self.ctrl = nn.Parameter(torch.zeros((n_frames - 1) // spline_step + 4, K, 6, device=dev))
        parent = torch.cdist(centers, joints_c).argmin(1) if joints_c is not None else torch.zeros(K, dtype=torch.long, device=points.device)
        self.register_buffer("parent", parent)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """(N,3) canonical points -> (N,K) skinning weights."""
        d = (x[:, None, :] - self.centers[None]) / self.log_scale.exp()[None]
        return torch.softmax(-0.5 * (d * d).sum(-1), dim=-1)

    def motion_parameters(self):
        return [p for n, p in self.named_parameters() if n != "log_scale"]

    def _spline(self, t: int) -> torch.Tensor:
        u = t / self.spline_step
        i = int(u)
        f = u - i
        w = torch.tensor([(1 - f) ** 3, 3 * f ** 3 - 6 * f ** 2 + 4, -3 * f ** 3 + 3 * f ** 2 + 3 * f + 1, f ** 3],
                         device=self.ctrl.device) / 6.0
        return torch.einsum("c,ckx->kx", w, self.ctrl[i:i + 4])  # (K,6)

    def _mlp(self, t: int, scaffold) -> torch.Tensor:
        p = scaffold.pose_feature(torch.tensor([t, max(t - 1, 0), max(t - 2, 0)])).detach()  # (3, D)
        return self.mlp(torch.cat([p[0], p[0] - p[1], p[0] - p[2]], -1)).view(self.K, 6)

    def params(self, t: int, scaffold=None):
        """-> rotvec (K,3), trans (K,3), spline residual (K,6) or None"""
        if self.mode == "free":
            return self.rotvec[t], self.trans[t], None
        res = None
        if self.mode == "spline":
            v = self._spline(t)
        elif self.mode == "pose_mlp":
            v = self._mlp(t, scaffold)
        else:
            res = self._spline(t)
            v = self._mlp(t, scaffold) + res
        return v[:, :3], v[:, 3:] * 0.1, res  # translation in 10 cm units

    def transforms(self, t: int, rot_delta: torch.Tensor | None = None, trans_delta: torch.Tensor | None = None, scaffold=None) -> torch.Tensor:
        rv, tr, _ = self.params(t, scaffold)
        if rot_delta is not None:
            rv = rv + rot_delta
            tr = tr + trans_delta
        return se3_matrix(rv, tr, self.centers)  # (K,4,4)

    def reg_loss(self, t: int, scaffold=None) -> torch.Tensor:
        """||B_t - I|| (DressRecon-style clothing-deformation minimisation)."""
        rv, tr, _ = self.params(int(t), scaffold)
        return (rv ** 2).sum(-1).mean() + (tr ** 2).sum(-1).mean() * 100.0  # 10cm ~ 1 rad

    def residual_loss(self, t: int, scaffold=None) -> torch.Tensor:
        """V3: L2 size of the spline residual on top of the pose MLP."""
        res = self.params(int(t), scaffold)[2]
        return (res ** 2).sum(-1).mean() if res is not None else torch.zeros((), device=self.centers.device)

    def smooth_loss(self, t: int) -> torch.Tensor:
        """Temporal smoothness of the per-frame SE(3) (optional, train.loss.bone_smooth; D-015)."""
        if self.mode != "free":
            return torch.zeros((), device=self.centers.device)
        T = self.rotvec.shape[0]
        nb = [s for s in (t - 1, t + 1) if 0 <= s < T]
        l = sum(((self.rotvec[t] - self.rotvec[s]) ** 2).sum(-1).mean() + ((self.trans[t] - self.trans[s]) ** 2).sum(-1).mean() * 100.0
                for s in nb)
        return l / max(len(nb), 1)

    def extra_repr(self):
        return f"mode={self.mode}, K={self.K}, sigma0~{math.exp(self.log_scale.mean().item()):.3f}m"
