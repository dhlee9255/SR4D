"""Run `python -m ftgspp.train <cfg> <run>` with the SR4D transparent-background copy of the training loop
(third_party/ftgspp_transparent/train.py, D-043). The external repo is not modified: the package-level `train`
function used by ftgspp.train.Trainer is swapped in memory. Needs FTGS_MASKS (see scripts/baselines/ftgspp_masks.py)."""
import importlib.util
import runpy
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("ftgspp_transparent_train", Path(__file__).resolve().parents[2] / "third_party/ftgspp_transparent/train.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import ftgspp.train as P  # noqa: E402

P.train = mod.train
if __name__ == "__main__":
    sys.argv = ["ftgspp.train"] + sys.argv[1:]
    runpy.run_module("ftgspp.train", run_name="__main__", alter_sys=True)
