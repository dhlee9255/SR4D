#!/usr/bin/env bash
# Item 4 (B-stage held-out views): re-run 4DAnyone with a second, higher ring (pitch 15 + 35, 24 views each) using the
# SAME GVHMR motion and seed as the original 24-view run, so (a) we can check whether co-generation changes the
# original 15-deg ring and (b) the 35-deg ring becomes held-out GT. 4Danyone repo is only read; outputs go to SR4D.
# usage: gen_4danyone_2ring.sh <video> <orig_run_dir> <out_data_dir> <gpu> [seed=42]
set -euo pipefail
VID=$(realpath "$1"); ORIG=$(realpath "$2"); OUT=$(realpath -m "$3"); G=$4; SEED=${5:-42}
FA=/data/AI-Graphics/Applications/4Danyone
CLIP=$(basename "$VID" .mp4)
mkdir -p "$OUT/gvhmr/results"
[ -d "$OUT/gvhmr/results/$CLIP" ] || cp -r "$ORIG/gvhmr/results/$CLIP" "$OUT/gvhmr/results/"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate 4danyone
cd $FA/4DAnyone
CUDA_VISIBLE_DEVICES=$G python inference.py --video_path "$VID" --views_per_layer 24 --layer_pitches '[15,35]' \
  --seed $SEED --data_dir "$OUT" --model_dir $FA/models --gpu_ids '[0]'
