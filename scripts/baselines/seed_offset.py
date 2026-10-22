"""In-memory seed offset for baselines whose training seed is hard-coded (4DGS: 6666, DualGS: 0). Item 6 (user
2026-10-12): 3 training seeds per method. If env TRAIN_SEED is set (non-zero), every random/np/torch seed call is shifted
by it, so the original code path is otherwise unchanged. External repos stay untouched (CLAUDE.md rule 1).
usage (as a launcher, cwd = the repo): TRAIN_SEED=1 python seed_offset.py train.py <train args...>
"""
import os
import random
import runpy
import sys

import numpy as np
import torch

OFF = int(os.environ.get("TRAIN_SEED", "0") or 0)


def install():
    if not OFF:
        return
    for mod, name in ((random, "seed"), (np.random, "seed"), (torch, "manual_seed"), (torch.cuda, "manual_seed_all"), (torch.cuda, "manual_seed")):
        orig = getattr(mod, name)
        setattr(mod, name, (lambda o: (lambda s=None, *a, **k: o((s or 0) + OFF, *a, **k)))(orig))
    print(f"[seed_offset] all seeds shifted by {OFF}", flush=True)


if __name__ == "__main__":
    install()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.getcwd())
    runpy.run_path(script, run_name="__main__")
