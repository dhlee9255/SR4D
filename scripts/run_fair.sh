#!/usr/bin/env bash
# D-038 fairness re-runs on gen48_ho8: 4DGS with the official multi-view preset (dynerf) and FTGS++ on a white background.
# usage: [SFX=_s123] run_fair.sh <zju|dance> <4dgs_dynerf|ftgspp_white> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
K=$1; M=$2; SFX=${SFX:-}; export CUDA_VISIBLE_DEVICES=$3 PHASE=B_g48
case $K in zju) R=data_dev/zju_mocap_gen/387_gen48_2ring$SFX ;; dance) R=data_dev/fouranyone_gen/dance5885633_gen48_2ring$SFX ;; esac
TAG=${K}_gen48_ho8$SFX; L=logs/fair_${TAG}_$M.log
case $M in
  4dgs_dynerf) METHOD=4dgs_dynerf NOTES="$TAG, official dynerf preset (14k it, batch 4, width 128, multires [1,2]) via mmcv stand-in" PRESET=arguments/dynerf/default.py \
                 scripts/baselines/run_4dgs_preset.sh $R gen48_ho8 outputs/4dgs_dynerf/$TAG > $L 2>&1 ;;
  ftgspp_white) BG=white INIT_FIX=0 METHOD=ftgspp_white_official NOTES="$TAG, white bg (D-011 wrapper), official init" \
                 scripts/baselines/run_ftgspp.sh $R gen48_ho8 outputs/ftgspp_white_official/$TAG > $L 2>&1 ;;
esac
echo "done $TAG $M rc=$? $(date +%H:%M)"
