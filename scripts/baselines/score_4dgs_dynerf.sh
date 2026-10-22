#!/usr/bin/env bash
# Render + score an already-trained 4DGS dynerf model (second half of run_4dgs_dynerf.sh; used after the 10-18 render crash).
# usage: score_4dgs_dynerf.sh <unified_root> <setting> <out_dir>   (env PHASE, NOTES)
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$(realpath "$1"); SET=$2; OUT=$(realpath -m "$3"); PRESET=$SR4D/configs/fourdgs/dynerf_default_nodl.py
source ~/miniconda3/etc/profile.d/conda.sh
export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
conda activate hust4dgs; cd "$SR4D/third_party/4DGaussians"
python "$SR4D/scripts/baselines/fourdgs_render_native.py" -m "$OUT/model" --configs "$PRESET" --root "$ROOT" --setting "$SET" --out "$OUT/renders"
conda activate sr4d; cd "$SR4D"
python -m sr4d.eval.metrics --phase "${PHASE:-B_g48}" --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" --method 4dgs_dynerf \
  --train_min 0 --n_gauss 0 --no_csv --notes "${NOTES:-official dynerf preset, dataloader off, native renderer (D-038)}"
if [ "$SET" = r15h35 ]; then PYTHONPATH=. python -m sr4d.eval.covis_metrics --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" | grep extrap; fi
rm -rf "$OUT/data/ims"
echo "SCOREDONE $(date +%H:%M)"
