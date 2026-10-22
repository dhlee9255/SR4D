#!/usr/bin/env bash
# EXP-009 (D-044): New-T = New-drop (inner_drop 0.5) + transparent-background training (--rand_bg --mask_w 0.05), no fade.
# Train -> plain render (renders_both) -> scores (+ co-visible for far) -> background-streak count.
# usage: run_newt.sh <gpu> <name> <data root> <setting> <inner ckpt> <seed>   log logs/lf_<name>.log, pid logs/lf_<name>.pid
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; G=$1; N=$2; R=$3; S=$4; I=$5; SEED=$6
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=.
L=logs/lf_$N.log; echo $$ > logs/lf_$N.pid; O=outputs/layered/$N
{
echo "START $(date +%H:%M) gpu $G $R $S $I seed=$SEED"
if [ ! -f $O/outer_final.pt ]; then
  python -m sr4d.layered.train --root $R --setting $S --inner $I --out $O --inner_drop 0.5 --seed $SEED --rand_bg --mask_w 0.05 || { echo "EXIT train failed $(date +%H:%M)"; exit 1; }
fi
python -m sr4d.layered.render --run $O --mode both --out $O/renders_both && \
python -m sr4d.eval.metrics --data_root $R --setting $S --renders $O/renders_both --method layered_${N}_renders_both --phase B --no_csv
[ "$S" = r15h35 ] && python -m sr4d.eval.covis_metrics --data_root $R --setting $S --renders $O/renders_both | grep extrap
python tools/count_bg_streaks.py --root $R --setting $S --state $O/outer_final.pt --json $O/bg_streaks.json | grep outer_final
echo "EXIT 0 $(date +%H:%M)"
} > $L 2>&1
