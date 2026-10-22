#!/usr/bin/env bash
# EXP-005..007 (D-042): train one layered run (New-drop + optional view-coverage pruning), then New-fade (10,25) render + score
# (+ plain render if PLAIN=1, + co-visible metrics for far views).
# usage: run_exp005.sh <gpu> <name> <data root> <setting> <inner ckpt> <N min views, 0 = New-drop> <seed> ; run inside tmux.
# log logs/exp005_<name>.log, pid logs/exp005_<name>.pid
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; G=$1; N=$2; R=$3; S=$4; I=$5; MV=$6; SEED=$7
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=.
L=logs/exp005_$N.log; echo $$ > logs/exp005_$N.pid; O=outputs/layered/$N
CV=$([ "$S" = r15h35 ] && echo 1 || echo "")
{
echo "START $(date +%H:%M) gpu $G $R $S $I N=$MV seed=$SEED"
if [ ! -f $O/outer_final.pt ]; then
  python -m sr4d.layered.train --root $R --setting $S --inner $I --out $O --inner_drop 0.5 --seed $SEED --cov_min_views $MV || { echo "EXIT train failed $(date +%H:%M)"; exit 1; }
fi
[ -f $O/vis_outer_final.pt ] || python -m sr4d.layered.fade --run $O
SUBS="renders_fade10_25"; [ "${PLAIN:-0}" = 1 ] && SUBS="$SUBS renders_both"
for SUB in $SUBS; do
  FA=$([ $SUB = renders_fade10_25 ] && echo "--fade 10 25" || echo "")
  python -m sr4d.layered.render --run $O --mode both --out $O/$SUB $FA && \
  python -m sr4d.eval.metrics --data_root $R --setting $S --renders $O/$SUB --method layered_${N}_$SUB --phase B --no_csv
  [ -n "$CV" ] && python -m sr4d.eval.covis_metrics --data_root $R --setting $S --renders $O/$SUB | grep extrap
  echo "SUBDONE $SUB $(date +%H:%M)"
done
echo "EXIT 0 $(date +%H:%M)"
} > $L 2>&1
