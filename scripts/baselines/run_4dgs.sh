#!/usr/bin/env bash
# HUST 4DGaussians baseline (dnerf_default hyper-parameters passed as CLI flags; mmcv is not in hust4dgs).
# usage: run_4dgs.sh <unified_root> <setting> <out_dir> [iterations] [coarse_iterations]
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$(realpath "$1"); SET=$2; OUT=$(realpath -m "$3"); IT=${4:-20000}; CIT=${5:-3000}
FD=$SR4D/third_party/4DGaussians
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sr4d; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
cd "$SR4D/tools" && python export_4dgs.py --root "$ROOT" --setting "$SET" --out "$OUT" && cd "$SR4D"
NF=$(python -c "import json;print(len(json.load(open('$OUT/export_meta.json'))['frames']))")
conda activate hust4dgs
[ "${PRUNE_UNSEEN:-}" = 1 ] && export PRUNE_UNSEEN="$OUT/data/train_meta.json"   # D-029 appendix
cd "$FD"
T0=$(date +%s)
python "$SR4D/scripts/baselines/fourdgs_train_whitebg.py" -s "$OUT/data" -m "$OUT/model" --port $((6000 + RANDOM % 3000)) --expname sr4d \
  --iterations "$IT" --coarse_iterations "$CIT" --deformation_lr_init 0.00016 --deformation_lr_final 0.0000016 \
  --deformation_lr_delay_mult 0.01 --grid_lr_init 0.0016 --grid_lr_final 0.000016 --pruning_interval 8000 --percent_dense 0.01 \
  --defor_depth 0 --net_width 64 --plane_tv_weight 0.0001 --time_smoothness_weight 0.01 --l1_time_planes 0.0001 \
  --weight_decay_iteration 0 --bounds 1.6 --save_iterations "$IT" --test_iterations "$IT"
T1=$(date +%s)
python "$SR4D/scripts/baselines/fourdgs_export_frames.py" -m "$OUT/model" --num_frames "$NF" --out "$OUT/frames"
conda activate sr4d
cd "$SR4D"
python scripts/baselines/render_ply.py --ply_pattern "$OUT/frames/time_{rel:05d}.ply" --root "$ROOT" --setting "$SET" --out "$OUT/renders"
NG=$(python -c "from plyfile import PlyData;print(PlyData.read('$OUT/frames/time_00000.ply')['vertex'].count)")
python -m sr4d.eval.metrics --phase "${PHASE:-A}" --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" --method "${METHOD:-4dgs}" \
  --train_min $(( (T1 - T0) / 60 )) --n_gauss "$NG" --notes "${NOTES:-dnerf_default hparams (multires default [1,2,4,8]), PanopticSports format}"
rm -rf "$OUT/data/ims" "$OUT/frames"   # regenerable (export / fourdgs_export_frames.py); disk is tight
