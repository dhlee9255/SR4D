"""L1 — canonical Gaussians.

Two sets share one canonical space:
  body : surfels attached to scaffold triangles (tri id + barycentric logits + local offset), flat (3rd scale = eps)
  free : unconstrained 3D Gaussians for clothing / hair / props, driven by free bones then LBS

Each set is a GaussianSet: a dict of per-Gaussian nn.Parameters (first dim = N) + per-Gaussian buffers,
with one Adam per parameter so relocation / growth can edit rows and optimizer state together.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from sr4d.utils.geometry import quat_mul


def sh_num(deg: int) -> int:
    return (deg + 1) ** 2


C0 = 0.28209479177387814


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


class GaussianSet(nn.Module):
    def __init__(self, name: str, params: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor]):
        super().__init__()
        self.name = name
        self.params = nn.ParameterDict({k: nn.Parameter(v.contiguous()) for k, v in params.items()})
        self.buffer_names = list(buffers)
        for k, v in buffers.items():
            self.register_buffer(k, v.contiguous())
        self.optimizers: dict[str, torch.optim.Adam] = {}

    def __len__(self):
        return next(iter(self.params.values())).shape[0]

    def __getitem__(self, k):
        if k in self.params:
            return self.params[k]
        return getattr(self, k)

    # ---- optimizers
    def setup_optimizers(self, lrs: dict[str, float]):
        self.optimizers = {k: torch.optim.Adam([p], lr=lrs.get(k, 0.0), eps=1e-15) for k, p in self.params.items()}

    def set_lr(self, key: str, lr: float):
        if key in self.optimizers:
            for g in self.optimizers[key].param_groups:
                g["lr"] = lr

    def step(self, keys: set[str] | None = None):
        for k, opt in self.optimizers.items():
            if keys is None or k in keys:
                opt.step()
            opt.zero_grad(set_to_none=True)

    def zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)

    # ---- row edits
    @torch.no_grad()
    def set_rows(self, idx: torch.Tensor, values: dict[str, torch.Tensor]):
        """Overwrite rows idx (params and buffers) and reset Adam moments of those rows."""
        for k, v in values.items():
            if k in self.params:
                self.params[k].data[idx] = v
                st = self.optimizers.get(k)
                if st is not None:
                    s = st.state.get(self.params[k], None)
                    if s:
                        s["exp_avg"][idx] = 0
                        s["exp_avg_sq"][idx] = 0
            else:
                getattr(self, k)[idx] = v

    @torch.no_grad()
    def append(self, values: dict[str, torch.Tensor]):
        """Append rows; every param/buffer must be provided."""
        n_new = next(iter(values.values())).shape[0]
        if n_new == 0:
            return
        for k in list(self.params.keys()):
            old = self.params[k]
            new = nn.Parameter(torch.cat([old.data, values[k]], 0))
            new.requires_grad_(old.requires_grad)
            opt = self.optimizers.get(k)
            if opt is not None:
                st = opt.state.pop(old, None)
                opt.param_groups[0]["params"] = [new]
                if st:
                    pad = torch.zeros((n_new,) + old.shape[1:], device=old.device, dtype=old.dtype)
                    st["exp_avg"] = torch.cat([st["exp_avg"], pad], 0)
                    st["exp_avg_sq"] = torch.cat([st["exp_avg_sq"], pad], 0)
                    opt.state[new] = st
            self.params[k] = new
        for k in self.buffer_names:
            setattr(self, k, torch.cat([getattr(self, k), values[k]], 0))

    def gather(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {k: p.data[idx].clone() for k, p in self.params.items()}
        out.update({k: getattr(self, k)[idx].clone() for k in self.buffer_names})
        return out

    def state_dict_full(self):
        return {"params": {k: p.data for k, p in self.params.items()},
                "buffers": {k: getattr(self, k) for k in self.buffer_names}}

    @classmethod
    def from_state(cls, name, st):
        return cls(name, st["params"], st["buffers"])


def init_body_set(n: int, tri: torch.Tensor, bary: torch.Tensor, sh_max: int, init_scale: float, colors: torch.Tensor | None = None) -> GaussianSet:
    dev = tri.device
    K = sh_num(sh_max) - 1
    rgb = colors if colors is not None else torch.full((n, 3), 0.5, device=dev)
    params = {
        "bary_logits": torch.log(bary.clamp_min(1e-4)),
        "offset": torch.zeros(n, 3, device=dev),  # local frame (t1,t2,n) coords
        "rot_theta": torch.rand(n, 1, device=dev) * 2 * math.pi,
        "rot_res": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1),
        "log_scales": torch.full((n, 2), math.log(init_scale), device=dev),
        "opacity": torch.full((n, 1), math.log(0.9 / 0.1), device=dev),
        "sh0": rgb_to_sh0(rgb)[:, None, :],
        "shN": torch.zeros(n, K, 3, device=dev),
    }
    return GaussianSet("body", params, {"tri": tri})


def init_free_set(points: torch.Tensor, lbs_vidx: torch.Tensor, sh_max: int, init_scale: float, colors: torch.Tensor | None = None) -> GaussianSet:
    n = points.shape[0]
    dev = points.device
    K = sh_num(sh_max) - 1
    rgb = colors if colors is not None else torch.full((n, 3), 0.5, device=dev)
    params = {
        "means": points.clone(),
        "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1),
        "log_scales": torch.full((n, 3), math.log(init_scale), device=dev),
        "opacity": torch.full((n, 1), math.log(0.5 / 0.5), device=dev),
        "sh0": rgb_to_sh0(rgb)[:, None, :],
        "shN": torch.zeros(n, K, 3, device=dev),
    }
    return GaussianSet("free", params, {"lbs_vidx": lbs_vidx})


def body_local_rotation(theta: torch.Tensor, rot_res: torch.Tensor) -> torch.Tensor:
    """Quaternion of the surfel in its triangle frame: residual * rotation about the frame normal (local z)."""
    half = 0.5 * theta[:, 0]
    q_theta = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], -1)
    return quat_mul(rot_res, q_theta)
