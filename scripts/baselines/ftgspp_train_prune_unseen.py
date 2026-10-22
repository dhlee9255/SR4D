"""Run `python -m ftgspp.train <cfg> <run> [args]` with the D-029 appendix rule (user 2026-10-12), in memory only:
before every MCMC relocation, Gaussians whose centre at time t (means_t) is outside the image of every training camera
at every training time are marked dead (opacity below FTGS++'s relocation threshold) so relocation reuses them.

Training cameras / times are taken from the training renders themselves (Gaussians.forward is wrapped to record each
(w2c, intrinsic, shape) and each t), so FTGS++'s own coordinate normalisation is used as-is. Cameras are a static rig;
times seen before the first relocation are used (logged). Optional env FTGS_WHITE=1 also applies the white-background
patch of ftgspp_train_whitebg.py. External repo untouched.
"""
import os
import runpy
import sys

import torch

if os.environ.get("FTGS_WHITE") == "1":
    import gsplat

    _orig_r = gsplat.rasterization

    def _white(*args, **kw):
        img, alpha, meta = _orig_r(*args, **kw)
        if kw.get("sh_degree") is not None and kw.get("backgrounds") is None:
            img = img + (1.0 - alpha)
        return img, alpha, meta

    gsplat.rasterization = _white

import importlib  # noqa: E402

T = importlib.import_module("ftgspp.train.train")  # the package attribute `train` is the function, not the module
from ftgspp.models import Gaussians  # noqa: E402

_CAMS, _TIMES = {}, set()
_EVAL = {"on": False}
_orig_fwd = Gaussians.forward
_orig_eval = T.eval_rgb


def _eval_rgb(*a, **kw):  # held-out evaluation renders must never feed the rule
    _EVAL["on"] = True
    try:
        return _orig_eval(*a, **kw)
    finally:
        _EVAL["on"] = False


T.eval_rgb = _eval_rgb


def _fwd(self, *a, **kw):
    w2c, K, shape, t = kw.get("w2c"), kw.get("intrinsic"), kw.get("shape"), kw.get("t")
    if not _EVAL["on"] and w2c is not None and K is not None:
        for i in range(w2c.shape[0]):
            key = tuple(torch.round(w2c[i].detach().flatten()[:12] * 1e4).long().tolist())
            if key not in _CAMS:
                H, W = (int(shape[0][i]) if torch.is_tensor(shape[0]) and shape[0].ndim else int(shape[0]),
                        int(shape[1][i]) if torch.is_tensor(shape[1]) and shape[1].ndim else int(shape[1]))
                _CAMS[key] = (w2c[i].detach().float(), K[i].detach().float(), H, W)
        if t is not None:
            _TIMES.add(round(float(t), 6))
    return _orig_fwd(self, *a, **kw)


Gaussians.forward = _fwd
_orig_reloc = T.relocate_gaussians
_N = {"calls": 0}


@torch.no_grad()
def _reloc(*a, **kw):
    gs = kw["gs"] if "gs" in kw else a[1]
    opacity_threshold = kw["opacity_threshold"] if "opacity_threshold" in kw else a[2]
    seen = torch.zeros(len(gs), dtype=torch.bool, device=gs.means.device)
    for t in sorted(_TIMES):
        x = gs.means_t(t)
        for w2c, K, H, W in _CAMS.values():
            pc = x @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            u = pc[:, 0] / z.clamp_min(1e-6) * K[0, 0] + K[0, 2]
            v = pc[:, 1] / z.clamp_min(1e-6) * K[1, 1] + K[1, 2]
            seen |= (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    un = ~seen
    if un.any() and _TIMES and _CAMS:
        gs.opacities.data[un] = torch.logit(torch.tensor(opacity_threshold * 0.5)).item()
    _N["calls"] += 1
    if _N["calls"] % 10 == 1:
        print(f"[prune_unseen] call {_N['calls']}: cams {len(_CAMS)} times {len(_TIMES)} killed {int(un.sum())}/{len(gs)}", flush=True)
    return _orig_reloc(*a, **kw)


T.relocate_gaussians = _reloc
if __name__ == "__main__":
    sys.argv = ["ftgspp.train"] + sys.argv[1:]
    runpy.run_module("ftgspp.train", run_name="__main__", alter_sys=True)
