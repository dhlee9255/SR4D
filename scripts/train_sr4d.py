"""Train SR4D.

  python scripts/train_sr4d.py --config configs/zju_dev.yaml [--ablation configs/ablations/no_l3.yaml] \
      --out outputs/sr4d/387/real12 [--resume_stage s3] [key=value ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr4d.train.trainer import Trainer  # noqa: E402
from sr4d.utils.config import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ablation", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume_stage", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config([a.config] + a.ablation, a.overrides)
    tr = Trainer(cfg, a.out)
    tr.setup(a.resume_stage)
    tr.train()
