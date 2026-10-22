"""Dump a FreeTimeGS++ gaussians.pt into the TimeGaussians state layout of third_party/ftgs_outer (read-only use of the
external code; run with the FTGS++ venv and PYTHONPATH=<FTGS++ repo>). usage: ftgspp_dump_state.py <run>/gaussians.pt <out.pt>"""
import sys

import torch
from ftgspp.models.gaussians import Gaussians

gs = Gaussians.load(sys.argv[1])
v = gs.velocity_model
st = {k: getattr(gs, k).detach().cpu() for k in ("means", "scales", "quats", "opacities", "sh_0", "sh_n", "times", "durations", "marginal_gates")}
st["velocities"] = (v if isinstance(v, torch.Tensor) else v.detach() if hasattr(v, "detach") else None).detach().cpu()
st["max_duration"] = float(gs.max_duration)
torch.save({"outer": st}, sys.argv[2]); print(sys.argv[2], len(st["means"]))
