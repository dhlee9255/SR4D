#!/usr/bin/env bash
# Item 5: after a seed-123 2-ring generation finishes, convert it and train SR4D (pose a) + 4DGS + FTGS++ on it sequentially.
# usage: chain_s123.sh <zju|dance> <gpu> <gen_pidfile>
set -uo pipefail
cd "$(dirname "$0")/.."
K=$1; G=$2; P=$(cat $3)
while kill -0 $P 2>/dev/null; do sleep 60; done
case $K in
  zju) bash scripts/prep_2ring.sh outputs/fouranyone_runs/zju_2ring_s123 zju_387_camB5_121f data_dev/zju_mocap_gen/387_gen48_2ring_s123 50 $G ;;
  dance) bash scripts/prep_2ring.sh outputs/fouranyone_runs/dance_2ring_s123 5885633-hd_1080_1920_25fps data_dev/fouranyone_gen/dance5885633_gen48_2ring_s123 25 $G ;;
esac > logs/prep2ring_${K}_s123.log 2>&1 || { echo prep failed; exit 1; }
for M in sr4d_pa 4dgs ftgspp; do SFX=_s123 bash scripts/run_r15h35.sh $K $M $G; done
echo chain_s123 done $K
