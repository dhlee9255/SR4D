#!/usr/bin/env bash
# B-stage dev queue (baselines) on ZJU-387 gen24, source-camera-registered evaluation cams (D-014)
set -uo pipefail
cd "$(dirname "$0")/.."
export PHASE=B NOTES="gen24 (4DAnyone from B5), source-camera registration"
R=data_dev/zju_mocap_gen/387_gen24_reg
scripts/baselines/run_4dgs.sh $R gen24 outputs/4dgs/387_gen/gen24 > logs/b_4dgs_387_gen24.log 2>&1 || echo "4dgs failed"
scripts/baselines/run_ftgspp.sh $R gen24 outputs/ftgspp/387_gen/gen24 > logs/b_ftgspp_387_gen24.log 2>&1 || echo "ftgspp failed"
