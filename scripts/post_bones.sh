#!/usr/bin/env bash
# Post-process one finished bone variant: jitter diag + orbit video. usage: post_bones.sh <run_dir> <orbit_dir> <tag>
set -euo pipefail
cd "$(dirname "$0")/.."; source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
R=$1; ORB=$2; TAG=$3
python tools/diag_jitter.py --ckpt $R/ckpt_final.pt --json $R/jitter.json
OUT=$ORB/bones_$TAG; mkdir -p $OUT
python -m sr4d.eval.render_cams --ckpt $R/ckpt_final.pt --out $OUT --sub sr4d --orbit $ORB
ffmpeg -y -loglevel error -framerate 24 -i $OUT/sr4d/%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 $ORB/sr4d_$TAG.mp4
echo post_done $TAG
