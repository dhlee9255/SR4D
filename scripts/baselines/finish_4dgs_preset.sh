#!/usr/bin/env bash
# Finish a run_4dgs_preset.sh run whose training completed but whose frame export failed (D-038): export frames with the
# same preset, render, evaluate. usage: PRESET=<file> finish_4dgs_preset.sh <unified_root> <setting> <out_dir>
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/../.." && pwd); ROOT=$(realpath "$1"); SET=$2; OUT=$(realpath -m "$3"); FD=$SR4D/third_party/4DGaussians
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8; cd "$SR4D"
NF=$(python -c "import json;print(len(json.load(open('$OUT/export_meta.json'))['frames']))")
[ -d $OUT/data/ims ] || (cd tools && python export_4dgs.py --root "$ROOT" --setting "$SET" --out "$OUT")  # images were removed after the first finish
conda activate hust4dgs; cd "$FD"
python "$SR4D/scripts/baselines/fourdgs_export_frames.py" -m "$OUT/model" --num_frames "$NF" --out "$OUT/frames" --configs "$PRESET"
conda activate sr4d; cd "$SR4D"
python scripts/baselines/render_ply.py --ply_pattern "$OUT/frames/time_{rel:05d}.ply" --root "$ROOT" --setting "$SET" --out "$OUT/renders"
NG=$(python -c "from plyfile import PlyData;print(PlyData.read('$OUT/frames/time_00000.ply')['vertex'].count)")
TMIN=$(grep -o "train_min [0-9]*" $OUT/train_min.txt 2>/dev/null | cut -d' ' -f2 || true)
python -m sr4d.eval.metrics --phase "${PHASE:-B_g48}" --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" --method "${METHOD:-4dgs_dynerf}" \
  --n_gauss "$NG" --notes "${NOTES:-official dynerf preset via mmcv stand-in, dataloader off}"
rm -rf "$OUT/data/ims" "$OUT/frames"
