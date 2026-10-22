#!/usr/bin/env bash
# EXP-009 4DGS dynerf re-queue (D-038 path: scripts/baselines/run_4dgs_dynerf.sh). The first try via run_4dgs_preset.sh
# deadlocked (dataloader on). usage: queue_exp009_dyn.sh <gpuA> <gpuB>; done -> logs/queue_exp009.log
set -uo pipefail
cd "$(dirname "$0")/.."
root() { case $1 in zju) echo data_dev/zju_mocap_gen/387_gen48_2ring$2 ;; dance) echo data_dev/fouranyone_gen/dance5885633_gen48_2ring$2 ;; esac; }
one() { local g=$1 K=$2 S=$3 X=${4:-}; local T=${K}_${S}$X; local L=logs/e9_dyn_$T.log
  echo "$(date +%H:%M) gpu$g start dyn2 $T" >> logs/queue_exp009.log
  ( echo $BASHPID > $L.pid; CUDA_VISIBLE_DEVICES=$g PHASE=$([ $S = r15h35 ] && echo B_r15h35 || echo B_g48) NOTES="$T, official dynerf preset, dataloader off, native renderer (EXP-009)" \
      bash scripts/baselines/run_4dgs_dynerf.sh $(root $K "$X") $S outputs/4dgs_dynerf/$T ) > $L 2>&1
  echo "$(date +%H:%M) gpu$g done dyn2 $T rc=$? log $L" >> logs/queue_exp009.log; }
( one $1 dance r15h35; one $1 dance r15h35 _s123; one $1 dance gen48_ho8 _s123; echo "LANE dyn$1 DONE $(date +%H:%M)" >> logs/queue_exp009.log ) &
( one $2 zju r15h35; one $2 zju r15h35 _s123; one $2 zju gen48_ho8 _s123; echo "LANE dyn$2 DONE $(date +%H:%M)" >> logs/queue_exp009.log ) &
wait; echo "DYNQUEUEDONE $(date +%H:%M)" >> logs/queue_exp009.log
