#!/usr/bin/env bash
# Item 6 (user 2026-10-12): A-stage final setting (ZJU-387 mask_cihp, real12) with extra TRAINING seeds.
# Same commands/settings as the original seed runs (results/Acihp_real12.csv, Abones_real12.csv); only the seed differs.
# usage: run_A_seed.sh <sr4d|4dgs|ftgspp|dualgs> <seed> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
M=$1; SEED=$2; export CUDA_VISIBLE_DEVICES=$3 PHASE=Aseeds TRAIN_SEED=$2
R=data_dev/zju_mocap/387_cihp; L=logs/Aseed_${M}_s$SEED.log
case $M in
  sr4d) METHOD=sr4d_v0 NOTES="mask_cihp, bones V0 (D-026), train seed $SEED" scripts/run_sr4d.sh outputs/sr4d_bones/387_cihp_real12/v0/config.yaml outputs/sr4d_bones/387_cihp_real12/v0_seed$SEED -- seed=$SEED > $L 2>&1 ;;
  4dgs) NOTES="mask_cihp, train seed offset $SEED" scripts/baselines/run_4dgs.sh $R real12 outputs/4dgs/387_cihp/real12_seed$SEED > $L 2>&1 ;;
  ftgspp) BG=black INIT_FIX=1 METHOD=ftgspp_black NOTES="mask_cihp, masked black bg, init fix, train seed offset $SEED" scripts/baselines/run_ftgspp.sh $R real12 outputs/ftgspp_black/387_cihp/real12_seed$SEED > $L 2>&1 ;;
  dualgs) NOTES="mask_cihp, train seed offset $SEED" scripts/baselines/run_dualgs_seed.sh $R real12 outputs/dualgs/387_cihp/real12_seed$SEED > $L 2>&1 ;;
esac
echo "done $M seed $SEED rc=$?"
