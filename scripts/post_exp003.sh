#!/usr/bin/env bash
# EXP-003 post-processing for one run: wait for training, score (main + co-visible), render 35-deg train-adjacent views + orbits.
# usage: post_exp003.sh <gpu> <run> <mode>
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; G=$1; N=$2; M=$3
while kill -0 $(cat logs/layered_$N.pid 2>/dev/null) 2>/dev/null; do sleep 20; done
[ -f outputs/layered/$N/metrics_renders_$M.json ] || scripts/eval_layered.sh $G $N $M
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
CUDA_VISIBLE_DEVICES=$G PYTHONPATH=. python -m sr4d.eval.covis_metrics --data_root data_dev/fouranyone_gen/dance5885633_gen48_2ring --setting r15h35 --renders outputs/layered/$N/renders_$M | grep extrap
while ! grep -q ORBDONE logs/orb_exp003.log 2>/dev/null; do sleep 20; done
scripts/render_layered_figs.sh $G $N $M "g03 g09 g15 g21" outputs/orbits/exp003/orbit_e35 outputs/orbits/exp003/orbit_e05
echo "POSTDONE $N"
