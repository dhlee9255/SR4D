"""Run 4DGaussians train.py with a WHITE background for PanopticSports cameras (setup_camera hard-codes black).
In-memory patch only (docs/DECISIONS.md D-011). Run with cwd = 4DGaussians, same CLI as train.py.
"""
import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seed_offset  # noqa: E402  (TRAIN_SEED env: shift the hard-coded 6666, item 6)

seed_offset.install()
import scene.dataset_readers as dr  # noqa: E402

_orig = dr.setup_camera


def _setup_camera_white(*a, **k):
    cam = _orig(*a, **k)
    return cam._replace(bg=torch.ones(3, dtype=torch.float32, device="cuda"))


dr.setup_camera = _setup_camera_white

# D-038 fairness fix: official 4DGaussians presets are mmcv python configs (--configs arguments/dynerf/default.py), but
# mmcv is not installed in hust4dgs and the env must not be modified. Minimal in-memory stand-in for mmcv.Config.fromfile:
# exec the plain-python preset and expose its dicts; train.py then merges them (utils/params_utils.merge_hparams), so list
# arguments such as multires take the official values.
if "--configs" in sys.argv and "mmcv" not in sys.modules:
    import types

    class _Cfg(dict):
        @staticmethod
        def fromfile(path):
            g = {}
            exec(compile(open(path).read(), path, "exec"), g)
            return _Cfg({k: v for k, v in g.items() if k.endswith("Params")})

    sys.modules["mmcv"] = types.SimpleNamespace(Config=_Cfg)

# D-029 appendix (user 2026-10-12): same "never inside any training image" rule as SR4D. Env PRUNE_UNSEEN=<train_meta.json>.
# After every densify() call, Gaussians whose (deformed, fine stage) centre is outside the image of every training camera
# at every training time are pruned. In-memory patch only.
if os.environ.get("PRUNE_UNSEEN"):
    import json
    from types import SimpleNamespace

    from scene.gaussian_model import GaussianModel
    from utils.render_utils import get_state_at_time

    _M = json.load(open(os.environ["PRUNE_UNSEEN"]))
    _T = len(_M["fn"])
    _CAMS = [(torch.tensor(k, device="cuda").float(), torch.tensor(w2c, device="cuda").float())
             for k, w2c in zip(_M["k"][0], _M["w2c"][0])]
    _W, _H = _M["w"], _M["h"]
    _orig_densify = GaussianModel.densify

    @torch.no_grad()
    def _unseen(g, stage):
        n = g.get_xyz.shape[0]
        seen = torch.zeros(n, dtype=torch.bool, device="cuda")
        for r in range(_T if stage == "fine" else 1):
            x = get_state_at_time(g, SimpleNamespace(time=r / _T))[0] if stage == "fine" else g.get_xyz
            for K, w2c in _CAMS:
                pc = x @ w2c[:3, :3].T + w2c[:3, 3]
                z = pc[:, 2]
                u = pc[:, 0] / z.clamp_min(1e-6) * K[0, 0] + K[0, 2]
                v = pc[:, 1] / z.clamp_min(1e-6) * K[1, 1] + K[1, 2]
                seen |= (z > 1e-3) & (u >= 0) & (u < _W) & (v >= 0) & (v < _H)
        return ~seen

    def _densify_prune_unseen(self, *a, **k):
        out = _orig_densify(self, *a, **k)
        stage = k.get("stage", a[-1] if a else "fine")
        m = _unseen(self, stage)
        if m.any():
            self.prune_points(m)
            print(f"[prune_unseen] {stage}: pruned {int(m.sum())} / {len(m)}", flush=True)
        return out

    GaussianModel.densify = _densify_prune_unseen
if __name__ == "__main__":
    sys.argv = ["train.py"] + sys.argv[1:]
    runpy.run_path("train.py", run_name="__main__")
