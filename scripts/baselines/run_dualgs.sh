#!/usr/bin/env bash
# DualGS baseline. usage: run_dualgs.sh <unified_root> <setting> <out_dir> [iterations] [subseq_iters]
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$(realpath "$1"); SET=$2; OUT=$(realpath -m "$3"); IT=${4:-30000}; SUB=${5:-15000}
DG=/data/AI-Graphics/Applications/DualGS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sr4d; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
cd "$SR4D/tools" && python export_dualgs.py --root "$ROOT" --setting "$SET" --out "$OUT" && cd "$SR4D"
EV=$(python -c "import json;print(','.join(map(str,json.load(open('$OUT/export_meta.json'))['eval_views'])))")
NF=$(python -c "import json;print(len(json.load(open('$OUT/export_meta.json'))['frames']))")
conda activate dualgs
cd "$DG"
T0=$(date +%s)
python train.py -s "$OUT/data" -m "$OUT/model" --frame_st 0 --frame_ed "$NF" --iterations "$IT" --subseq_iters "$SUB" \
  --training_mode 0 --parallel_load -r 1 --eval --eval_views "$EV"
T1=$(date +%s)
conda activate sr4d
cd "$SR4D"
python scripts/baselines/render_ply.py --ply_pattern "$OUT/model/ckt/point_cloud_{rel}.ply" --root "$ROOT" --setting "$SET" --out "$OUT/renders"
NG=$(python -c "from plyfile import PlyData;print(PlyData.read('$OUT/model/ckt/point_cloud_0.ply')['vertex'].count)")
python -m sr4d.eval.metrics --phase "${PHASE:-A}" --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" --method dualgs \
  --train_min $(( (T1 - T0) / 60 )) --n_gauss "$NG" --notes "${NOTES:-iters=$IT subseq=$SUB, init=visual hull+SMPL verts}"
