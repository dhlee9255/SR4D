#!/usr/bin/env bash
# EXP-009: render + score the 6 trained 4DGS dynerf models (native render crashed on a.orbit, fixed 10-18), then orbits.
# usage: queue_exp009_dynscore.sh <gpuA> <gpuB>; done -> logs/queue_exp009.log
cd "$(dirname "$0")/.."
root() { [ $1 = zju ] && echo data_dev/zju_mocap_gen/387_gen48_2ring$2 || echo data_dev/fouranyone_gen/dance5885633_gen48_2ring$2; }
one() { local g=$1 K=$2 S=$3 X=${4:-}; local T=${K}_${S}$X; local L=logs/e9_dynscore_$T.log rc
  echo "$(date +%H:%M) gpu$g start dynscore $T" >> logs/queue_exp009.log
  ( echo $BASHPID > $L.pid; CUDA_VISIBLE_DEVICES=$g PHASE=$([ $S = r15h35 ] && echo B_r15h35 || echo B_g48) NOTES="$T, official dynerf preset, dataloader off, native renderer (EXP-009)" \
      bash scripts/baselines/score_4dgs_dynerf.sh $(root $K "$X") $S outputs/4dgs_dynerf/$T ) > $L 2>&1; rc=$?
  echo "$(date +%H:%M) gpu$g done dynscore $T rc=$rc log $L" >> logs/queue_exp009.log; }
( one $1 dance r15h35; one $1 dance r15h35 _s123; one $1 dance gen48_ho8 _s123; bash scripts/orbits_exp009.sh $1 > logs/orbits_exp009.log 2>&1; echo "LANE dynscore$1 DONE $(date +%H:%M)" >> logs/queue_exp009.log ) &
( one $2 zju r15h35; one $2 zju r15h35 _s123; one $2 zju gen48_ho8 _s123; echo "LANE dynscore$2 DONE $(date +%H:%M)" >> logs/queue_exp009.log ) &
wait; echo "SCOREQUEUEDONE $(date +%H:%M)" >> logs/queue_exp009.log
