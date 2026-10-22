#!/usr/bin/env bash
# 4DGS official dynerf preset done the D-038 way: dataloader-off copy (configs/fourdgs/dynerf_default_nodl.py, otherwise the
# official preset) + scoring from the 4DGS native renderer (the PLY-export path is invalid for this preset, D-038).
# usage: run_4dgs_dynerf.sh <unified_root> <setting> <out_dir>   (env PHASE, NOTES)
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$(realpath "$1"); SET=$2; OUT=$(realpath -m "$3"); PRESET=$SR4D/configs/fourdgs/dynerf_default_nodl.py
FD=$SR4D/third_party/4DGaussians
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sr4d; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
cd "$SR4D/tools" && python export_4dgs.py --root "$ROOT" --setting "$SET" --out "$OUT" && cd "$SR4D"
conda activate hust4dgs
cd "$FD"
T0=$(date +%s)
IT=$(python -c "g={};exec(open('$PRESET').read(),g);print(g['OptimizationParams']['iterations'])")
python "$SR4D/scripts/baselines/fourdgs_train_whitebg.py" -s "$OUT/data" -m "$OUT/model" --port $((6000 + RANDOM % 3000)) --expname sr4d \
  --configs "$PRESET" --save_iterations "$IT" --test_iterations "$IT"
T1=$(date +%s)
python "$SR4D/scripts/baselines/fourdgs_render_native.py" -m "$OUT/model" --configs "$PRESET" --root "$ROOT" --setting "$SET" --out "$OUT/renders"
conda activate sr4d
cd "$SR4D"
python -m sr4d.eval.metrics --phase "${PHASE:-B_g48}" --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" --method 4dgs_dynerf \
  --train_min $(( (T1 - T0) / 60 )) --n_gauss 0 --notes "${NOTES:-official dynerf preset, dataloader off, native renderer (D-038)}"
[ "$SET" = r15h35 ] && PYTHONPATH=. python -m sr4d.eval.covis_metrics --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" | grep extrap
rm -rf "$OUT/data/ims"
echo "DYNDONE $(date +%H:%M)"
