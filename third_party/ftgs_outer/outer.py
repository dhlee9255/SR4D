"""Outer layer of the 2-layer model (docs/DECISIONS.md D-040): FreeTimeGS++ time Gaussians, adapted copy.

Source (read-only, copied verbatim to ./orig/ on 2026-10-17): 4Danyone/4dgs_recon/FreeTimeGSPlusPlus/ftgspp
  models/gaussians.py      -> TimeGaussians (means_t, temporal_opacity, opacities_t, marginal gate)
  init/__init__.py         -> init_dynamic (keyframe points + KNN velocity, time/duration/scale/opacity init)
  train/relocation.py      -> relocate_mcmc (3d_mcmc branch)
  train/state.py           -> TrainState (2D gradient accumulation; adapted to packed=False meta)
  train/train.py           -> make_optimizers (per-parameter Adam, sqrt(batch) lr scaling, scene-extent means/velocity lr)
  utils/math.py            -> knn_dist2_mean, nearest, scene_extent, rgb_to_sh
FreeTimeGS++ Research and Educational Use License (./orig/LICENSE), (c) 2026 SNU VGI Lab / ETRI.
Changes: explicit velocities only (no tinycudann VelocityField, no colour correctors); rendering is done by the caller
(sr4d/layered/render.py) together with the inner layer, so this module exposes per-time attributes instead of forward().
"""
from __future__ import annotations

import math

import numpy as np
import scipy.spatial
import torch
from gsplat.relocation import compute_relocation
from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer
from torch import Tensor, nn

SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0


def knn_dist2_mean(xyz: Tensor, k: int = 3) -> Tensor:  # FTGS++: query against itself (includes the point itself)
    x = xyz.detach().cpu().numpy()
    d, _ = scipy.spatial.KDTree(x).query(x, k=k)
    return torch.tensor(d ** 2).float().mean(1)


def nearest(x0: Tensor, x1: Tensor, k: int = 1) -> Tensor:
    _, idx = scipy.spatial.KDTree(x0.detach().cpu().numpy()).query(x1.detach().cpu().numpy(), k=k)
    return torch.as_tensor(np.asarray(idx)).long()


def scene_extent(w2c: Tensor) -> float:
    pos = torch.inverse(w2c)[:, :3, 3]
    return torch.norm(pos - pos.mean(0), dim=1).max().item()


def quat_to_rotmat(q: Tensor) -> Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


class TimeGaussians(nn.Module):
    PARAMS = ("means", "scales", "quats", "opacities", "sh_0", "sh_n", "times", "durations", "velocities", "marginal_gates")

    def __init__(self, max_duration: float = 1.0, **p):
        super().__init__()
        for k in self.PARAMS:
            setattr(self, k, nn.Parameter(p[k].float().contiguous()))
        self.max_duration = max_duration
        self.sh_degree = math.isqrt(self.sh_n.shape[1] + 1) - 1

    def __len__(self):
        return self.means.shape[0]

    def means_t(self, t):
        return self.means + (t - self.times) * self.velocities

    def gate(self):
        return torch.sigmoid(20 * self.marginal_gates)

    def sigma_t(self):
        return self.max_duration / 6 * torch.sigmoid(self.durations)

    def temporal_opacity(self, t):
        g = self.gate()
        return g + (1 - g) * torch.exp(-0.5 * ((t - self.times) / self.sigma_t()) ** 2)

    def opacities_t(self, t):
        return self.opacities.sigmoid() * self.temporal_opacity(t)

    def covars(self):
        R = quat_to_rotmat(self.quats)
        s = self.scales.exp()
        return R @ torch.diag_embed(s * s) @ R.transpose(-1, -2)

    def shs(self):
        return torch.cat([self.sh_0, self.sh_n], 1)

    def state(self):
        return {k: getattr(self, k).detach().cpu() for k in self.PARAMS} | {"max_duration": self.max_duration}

    @classmethod
    def from_state(cls, st):
        st = dict(st)
        md = st.pop("max_duration")
        return cls(max_duration=md, **st)


@torch.no_grad()
def init_dynamic(n: int, xyz: Tensor, rgb: Tensor, xyz_next: Tensor, time: float, delta_time: float, *, scale=0.1,
                 opacity=0.5, duration=0.1, max_duration=1.0, sh_degree=3, num_velocity_nns=3, gen=None) -> dict:
    """FTGS++ init_dynamic (explicit velocities). xyz/rgb (M,3) cpu, rgb in [0,1]."""
    n = min(n, len(xyz), len(xyz_next))
    # FTGS++ samples with replacement from a 700k-point cloud; our keyframe sets are already subsampled -> no replacement
    mask = torch.randperm(len(xyz), generator=gen)[:n]
    means = xyz[mask].clone()
    times = (torch.rand((n, 1), generator=gen) - 0.5) * delta_time + time
    k = max(1, min(num_velocity_nns, len(xyz_next)))
    nn_idx = nearest(xyz_next, means, k).reshape(n, -1)
    vel = ((xyz_next[nn_idx] - means[:, None]) / delta_time).mean(1)
    spacing = torch.sqrt(torch.clamp(knn_dist2_mean(means), min=1e-7))
    durations = torch.logit(torch.full((n, 1), duration) * 6 / max_duration)
    quats = torch.zeros((n, 4)); quats[:, 0] = 1
    sh = torch.zeros((n, (sh_degree + 1) ** 2, 3)); sh[:, 0] = rgb_to_sh(rgb[mask])
    return dict(means=means, scales=torch.log(scale * spacing[:, None].repeat(1, 3)), quats=quats,
                opacities=torch.logit(torch.full((n, 1), opacity)), sh_0=sh[:, :1], sh_n=sh[:, 1:], times=times,
                durations=durations, velocities=vel, marginal_gates=torch.full((n, 1), -1.0))


def cat_states(parts: list[dict]) -> dict:
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


DEFAULT_LRS = dict(means=1.6e-4, quats=1e-3, scales=5e-3, opacities=5e-2, sh_0=2.5e-3, sh_n=1.25e-4, durations=5e-3,
                   times=1.6e-4, velocities=1.6e-3, marginal_gates=1e-3)
DEFAULT_SCHED = dict(means=0.01, velocities=0.01)


def make_optimizers(gs: TimeGaussians, lrs: dict, sched: dict, iterations: int, extent: float, batch_size: int = 1):
    opts = {}
    for name, p in gs.named_parameters(recurse=False):
        if name in lrs and p.requires_grad:
            lr = lrs[name] * batch_size ** 0.5 * (extent if name in ("means", "velocities") else 1.0)
            opts[name] = torch.optim.Adam([{"params": p, "lr": lr, "name": name}], eps=1e-15 / batch_size ** 0.5,
                                          betas=(0.9 ** batch_size, 0.999 ** batch_size))
    scheds = [torch.optim.lr_scheduler.ExponentialLR(opts[k], v ** (1 / iterations)) for k, v in sched.items() if k in opts]
    return opts, scheds


class TrainState:
    """2D mean-gradient accumulation for relocation scores (FTGS++ state.py, for packed=False gsplat meta)."""

    def __init__(self, n, dev="cuda"):
        self.grad2d = torch.zeros(n, device=dev)
        self.count = torch.zeros(n, device=dev)

    @torch.no_grad()
    def update_(self, means2d_grad: Tensor, radii: Tensor, W: int, H: int):
        g = means2d_grad.clone()  # (N,2) for the outer slice
        g[:, 0] *= W / 2.0
        g[:, 1] *= H / 2.0
        vis = (radii > 0).all(-1) if radii.dim() == 2 else radii > 0
        self.grad2d[vis] += g[vis].norm(dim=-1)
        self.count[vis] += 1

    def grad2d_acc(self):
        return self.grad2d / self.count.clamp_min(1)

    def zero_(self):
        self.grad2d.zero_(); self.count.zero_()


def relocation_binoms(n_max=51):
    b = torch.zeros((n_max, n_max))
    for n in range(n_max):
        for k in range(n + 1):
            b[n, k] = math.comb(n, k)
    return b


@torch.no_grad()
def relocate_mcmc(gs: TimeGaussians, opts: dict, state: TrainState, binoms: Tensor, min_opacity: float = 0.005,
                  src_ok: Tensor | None = None, rows: tuple = ()) -> int:
    """FTGS++ relocate_gaussians(score_mode=default) + _relocate_mcmc.
    SR4D additions (EXP-005, D-042): src_ok (N,) bool excludes Gaussians from being relocation sources (no cloning);
    rows: extra per-Gaussian tensors whose rows are copied src -> dead in place (view-coverage statistics)."""
    op = torch.sigmoid(gs.opacities).flatten()
    g2 = state.grad2d_acc()
    probs = op / op.sum() + g2 / g2.sum().clamp_min(1e-12)
    if not torch.isfinite(probs).all():
        probs = torch.ones_like(probs)
    dead = op <= min_opacity
    nd = int(dead.sum())
    if nd == 0 or nd == len(op):
        return nd
    dead_idx = dead.nonzero(as_tuple=True)[0]
    alive_idx = (~dead).nonzero(as_tuple=True)[0]
    if src_ok is not None and bool(src_ok[alive_idx].any()):
        probs = torch.where(src_ok, probs, torch.zeros_like(probs))
    src = alive_idx[_multinomial_sample(probs[alive_idx].flatten(), nd, replacement=True)]
    eps = torch.finfo(torch.float32).eps
    new_op, new_sc = compute_relocation(opacities=op[src], scales=torch.exp(gs.scales)[src],
                                        ratios=torch.bincount(src)[src] + 1, binoms=binoms)
    new_op = torch.clamp(new_op, max=1.0 - eps, min=min_opacity)

    def param_fn(name, p):
        u = p.detach().clone()
        if name == "opacities":
            u[src] = torch.logit(new_op)[:, None] if u.dim() == 2 and new_op.dim() == 1 else torch.logit(new_op)
        elif name == "scales":
            u[src] = torch.log(new_sc)
        u[dead_idx] = u[src]
        return nn.Parameter(u, requires_grad=p.requires_grad)

    def opt_fn(_, v):
        v = v.clone(); v[src] = 0
        return v

    _update_param_with_optimizer(param_fn=param_fn, optimizer_fn=opt_fn, params=gs._parameters, optimizers=opts)
    state.grad2d[src] = 0; state.count[src] = 0
    for r in rows:
        r[dead_idx] = r[src]
    return nd
