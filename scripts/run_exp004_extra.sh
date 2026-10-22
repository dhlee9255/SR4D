#!/usr/bin/env bash
# EXP-004 New-fade (D-041): record view coverage of one New-drop run, then render + score 3 (th1, th2) fades.
# usage: run_exp004_extra.sh <gpu> <run> "<th1 th2>" [covis]   (extra (th1, th2) beyond the 3 asked ones); run inside tmux. log logs/exp004_<run>.log, pid logs/exp004_<run>.pid
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; G=$1; N=$2; TH=$3; CV=${4:-}
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=.
L=logs/exp004x_$N.log; echo $$ > logs/exp004x_$N.pid; O=outputs/layered/$N
R=$(python -c "import json;print(json.load(open('$O/args.json'))['root'])"); S=$(python -c "import json;print(json.load(open('$O/args.json'))['setting'])")
{
[ -f $O/vis_outer_final.pt ] || python -m sr4d.layered.fade --run $O
for th in "$TH"; do set -- $th; SUB=renders_fade$1_$2
  python -m sr4d.layered.render --run $O --mode both --out $O/$SUB --fade $1 $2 && \
  python -m sr4d.eval.metrics --data_root $R --setting $S --renders $O/$SUB --method layered_${N}_fade$1_$2 --phase B --no_csv
  [ -n "$CV" ] && python -m sr4d.eval.covis_metrics --data_root $R --setting $S --renders $O/$SUB | grep extrap
  echo "FADEDONE $1 $2 $(date +%H:%M)"
done
echo "EXIT $? $(date +%H:%M)"
} > $L 2>&1
