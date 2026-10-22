#!/usr/bin/env bash
# D-037 detail layer on top of an EXISTING SR4D checkpoint (shared model not retrained): resume at its final iteration with
# stage s7 appended, then render held-out + training crops and evaluate. usage: run_s7.sh <base_run_dir> <tag> <gpu> [ablation.yaml ...]
set -uo pipefail
cd "$(dirname "$0")/.."
BASE=$1; TAG=$2; export CUDA_VISIBLE_DEVICES=$3 PHASE=B_g48; shift 3
OUT=${BASE}_$TAG; L=logs/s7_$(basename $OUT).log; mkdir -p $OUT
ln -f $BASE/ckpt_final.pt $OUT/ckpt_base.pt 2>/dev/null || cp $BASE/ckpt_final.pt $OUT/ckpt_base.pt
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
python scripts/train_sr4d.py --config $BASE/config.yaml --ablation configs/ablations/detail_s7.yaml "$@" --out $OUT --resume_stage base > $L 2>&1
python -m sr4d.eval.render_cams --ckpt $OUT/ckpt_final.pt --out $OUT >> $L 2>&1
ROOT=$(python3 -c "import yaml;print(yaml.safe_load(open('$OUT/config.yaml'))['data']['root'])"); SET=$(python3 -c "import yaml;print(yaml.safe_load(open('$OUT/config.yaml'))['data']['setting'])")
python -m sr4d.eval.metrics --phase B_g48 --data_root $ROOT --setting $SET --renders $OUT/renders --method sr4d_$TAG --notes "s7 detail layer on $(basename $BASE)" >> $L 2>&1
python tools/diag_jitter.py --ckpt $OUT/ckpt_final.pt --json $OUT/jitter.json >> $L 2>&1
PYTHONPATH=. python tools/render_some.py --ckpt $OUT/ckpt_final.pt --out $OUT/renders_train --cams g01 g11 --frames 0 15 30 45 59 >> $L 2>&1
rm -f $OUT/ckpt_latest.pt
echo "done $OUT rc=$? $(date +%H:%M)"
