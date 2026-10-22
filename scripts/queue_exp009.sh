#!/usr/bin/env bash
# EXP-009 (D-044) job pool on 4 GPUs: New-T x (dance, zju) x (gen48_ho8, r15h35) x (gen seed 42, 123) + missing baselines
# (DualGS gen48_ho8 s123, FTGS++ transparent, 4DGS dynerf). One job per GPU; each lane pulls the next job from a locked list.
# usage: queue_exp009.sh <gpu> ...   run inside tmux; done -> logs/queue_exp009.log
set -uo pipefail
cd "$(dirname "$0")/.."; source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
J=logs/exp009_jobs.txt; LK=logs/exp009_jobs.lock
root() { case $1 in zju) echo data_dev/zju_mocap_gen/387_gen48_2ring$2 ;; dance) echo data_dev/fouranyone_gen/dance5885633_gen48_2ring$2 ;; esac; }
phase() { [ $1 = r15h35 ] && echo B_r15h35 || echo B_g48; }
# job lines: <type> <k> <split> <sfx>   (longest first)
{ for K in dance zju; do echo "dualgs $K gen48_ho8 _s123"; done
  for X in "" _s123; do for K in dance zju; do for S in r15h35 gen48_ho8; do
    [ "$K$S$X" = dancegen48_ho8 ] || echo "newt $K $S $X"; done; done; done
  for X in "" _s123; do for K in dance zju; do for S in r15h35 gen48_ho8; do
    [ "$K$X" = dance ] || echo "ftt $K $S $X"; done; done; done
  for X in "" _s123; do for K in dance zju; do echo "dyn $K r15h35 $X"; done; done
  for K in dance zju; do echo "dyn $K gen48_ho8 _s123"; done
} > $J
run() {  # gpu type k split sfx
  local g=$1 T=$2 K=$3 S=$4 X=${5:-}; local R=$(root $K "$X") TAG=${K}_${S}${X}; local L=logs/e9_${T}_$TAG.log
  echo "$(date +%H:%M) gpu$g start $T $TAG" >> logs/queue_exp009.log
  case $T in
    newt) bash scripts/run_newt.sh $g exp009_$TAG $R $S outputs/sr4d_final/${TAG}_nol4/ckpt_final.pt 0; L=logs/lf_exp009_$TAG.log ;;
    ftt) ( echo $BASHPID > $L.pid; CUDA_VISIBLE_DEVICES=$g PHASE=$(phase $S) bash scripts/baselines/run_ftgspp_transparent.sh $R $S outputs/ftgspp_transparent/$TAG ) > $L 2>&1 ;;
    dyn) ( echo $BASHPID > $L.pid; CUDA_VISIBLE_DEVICES=$g PHASE=$(phase $S) METHOD=4dgs_dynerf NOTES="$TAG, official dynerf preset (EXP-009)" PRESET=arguments/dynerf/default.py \
             bash scripts/baselines/run_4dgs_preset.sh $R $S outputs/4dgs_dynerf/$TAG
           [ $S = r15h35 ] && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python -m sr4d.eval.covis_metrics --data_root $R --setting $S --renders outputs/4dgs_dynerf/$TAG/renders | grep extrap ) > $L 2>&1 ;;
    dualgs) ( echo $BASHPID > $L.pid; CUDA_VISIBLE_DEVICES=$g PHASE=$(phase $S) bash scripts/baselines/run_dualgs.sh $R $S outputs/dualgs/$TAG ) > $L 2>&1 ;;
  esac
  echo "$(date +%H:%M) gpu$g done $T $TAG rc=$? log $L" >> logs/queue_exp009.log
}
lane() { local g=$1 line
  while true; do
    line=$(flock $LK bash -c "l=\$(head -1 $J); [ -n \"\$l\" ] && sed -i 1d $J; echo \"\$l\"")
    [ -z "$line" ] && break
    run $g $line
  done; echo "LANE $g DONE $(date +%H:%M)" >> logs/queue_exp009.log; }
for g in "$@"; do lane $g & sleep 5; done
wait; echo "QUEUEDONE $(date +%H:%M)" >> logs/queue_exp009.log
