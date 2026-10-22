#!/usr/bin/env bash
# D-029 appendix (user 2026-10-12): 4DGS / FTGS++ on r15h35 with the same "never inside any training image" pruning rule
# as SR4D (in-memory hooks: scripts/baselines/fourdgs_train_whitebg.py, ftgspp_train_prune_unseen.py). Same data, split,
# protocol and outputs layout as run_r15h35.sh (FTGS++ legacy black bg, INIT_FIX=0).
# usage: [SFX=_s123] run_r15h35_unseen.sh <zju|dance> <4dgs|ftgspp> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
K=$1; M=$2; SFX=${SFX:-}; export CUDA_VISIBLE_DEVICES=$3 PHASE=B_r15h35 PRUNE_UNSEEN=1
case $K in zju) R=data_dev/zju_mocap_gen/387_gen48_2ring ;; dance) R=data_dev/fouranyone_gen/dance5885633_gen48_2ring ;; esac
R=$R$SFX
L=logs/r15h35${SFX}_${K}_${M}_unseen.log
case $M in
  4dgs) METHOD=4dgs_unseen NOTES="$K r15h35$SFX held-out 35deg, prune unseen (D-029 appendix)" scripts/baselines/run_4dgs.sh $R r15h35 outputs/4dgs_unseen/${K}_r15h35$SFX > $L 2>&1 ;;
  ftgspp) BG=black INIT_FIX=0 METHOD=ftgspp_black_official_unseen NOTES="$K r15h35$SFX held-out 35deg, legacy, prune unseen (D-029 appendix)" scripts/baselines/run_ftgspp.sh $R r15h35 outputs/ftgspp_black_official_unseen/${K}_r15h35$SFX > $L 2>&1 ;;
esac
echo "done $K $M rc=$?"
