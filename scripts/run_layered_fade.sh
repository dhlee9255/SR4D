#!/usr/bin/env bash
# EXP-008+ (D-043): train one layered run with extra train.py args, then New-fade (10,25) + plain renders, scores
# (+ co-visible for far), and the background-streak count. usage: run_layered_fade.sh <gpu> <name> <data root> <setting> <inner ckpt> <seed> [train args...]
# run inside tmux; log logs/lf_<name>.log, pid logs/lf_<name>.pid
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; G=$1; N=$2; R=$3; S=$4; I=$5; SEED=$6; shift 6
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=.
L=logs/lf_$N.log; echo $$ > logs/lf_$N.pid; O=outputs/layered/$N
CV=$([ "$S" = r15h35 ] && echo 1 || echo "")
{
echo "START $(date +%H:%M) gpu $G $R $S $I seed=$SEED args: $*"
if [ ! -f $O/outer_final.pt ]; then
  python -m sr4d.layered.train --root $R --setting $S --inner $I --out $O --inner_drop 0.5 --seed $SEED "$@" || { echo "EXIT train failed $(date +%H:%M)"; exit 1; }
fi
[ -f $O/vis_outer_final.pt ] || python -m sr4d.layered.fade --run $O
for SUB in renders_fade10_25 renders_both; do
  FA=$([ $SUB = renders_fade10_25 ] && echo "--fade 10 25" || echo "")
  python -m sr4d.layered.render --run $O --mode both --out $O/$SUB $FA && \
  python -m sr4d.eval.metrics --data_root $R --setting $S --renders $O/$SUB --method layered_${N}_$SUB --phase B --no_csv
  [ -n "$CV" ] && python -m sr4d.eval.covis_metrics --data_root $R --setting $S --renders $O/$SUB | grep extrap
  echo "SUBDONE $SUB $(date +%H:%M)"
done
python tools/count_bg_streaks.py --root $R --setting $S --state $O/outer_final.pt --json $O/bg_streaks.json | grep outer_final
echo "EXIT 0 $(date +%H:%M)"
} > $L 2>&1
