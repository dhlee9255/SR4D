#!/usr/bin/env bash
# Co-visible masked metrics for every finished r15h35 run (all methods, all generation seeds). Skips runs already done.
# usage: covis_all.sh [gpu]
cd "$(dirname "$0")/.."; source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
export CUDA_VISIBLE_DEVICES=${1:-0}
for M in outputs/*/{zju,dance}_r15h35*/metrics_renders.json outputs/sr4d_bones/{zju,dance}_r15h35*/*/metrics_renders.json; do
  [ -f "$M" ] || continue; D=$(dirname $M); [ -f $D/metrics_covis_renders.json ] && continue
  N=$(basename $(dirname $D))/$(basename $D); [[ $D == outputs/sr4d_bones/* ]] && N=$(basename $(dirname $D))
  case $N in *zju_r15h35*) R=data_dev/zju_mocap_gen/387_gen48_2ring ;; *dance_r15h35*) R=data_dev/fouranyone_gen/dance5885633_gen48_2ring ;; esac
  S=$(echo $N | grep -o "_s[0-9]*$" || true); R=$R$S
  [ -d $R/covis_r15h35 ] || PYTHONPATH=. python tools/covis_masks.py --root $R --setting r15h35 | tail -n 1
  [ -d $D/renders ] || { echo "no renders: $D"; continue; }
  echo "== $D ($R)"; python -m sr4d.eval.covis_metrics --data_root $R --setting r15h35 --renders $D/renders | grep extrap
done
