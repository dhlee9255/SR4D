#!/usr/bin/env bash
# Run SR4D ablations sequentially on one GPU. usage: run_ablation.sh <config> <out_root> <name> [<name> ...]
# names: s2_only no_l2 no_l3 no_l4 joint_training no_scaffold  (configs/ablations/<name>.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=$1; OUT=$2; shift 2
for A in "$@"; do
  METHOD="sr4d_$A" NOTES="ablation $A" scripts/run_sr4d.sh "$CFG" "$OUT/$A" "configs/ablations/$A.yaml" > "logs/ablation_$(basename "$OUT")_$A.log" 2>&1 || echo "ablation $A failed"
done
