#!/usr/bin/env bash
# D-032: boundary-band / interior split metrics for every finished main-protocol and appendix run (skips done ones).
cd "$(dirname "$0")/.."; source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export PYTHONPATH=.
root_of(){ case $1 in
  *zju_gen24_all*) echo data_dev/zju_mocap_gen/387_gen24_genonly gen24 ;; *dance_gen24_all*) echo data_dev/fouranyone_gen/dance5885633_gen24 gen24 ;;
  *zju_gen48_*|*zju_r15h35*) s=$(echo $1 | grep -o "_s[0-9]*\b" | head -1); echo data_dev/zju_mocap_gen/387_gen48_2ring$s ;;
  *dance_gen48_*|*dance_r15h35*) s=$(echo $1 | grep -o "_s[0-9]*\b" | head -1); echo data_dev/fouranyone_gen/dance5885633_gen48_2ring$s ;; esac; }
for M in outputs/*/{zju,dance}_{gen48_ho8,gen24_all,r15h35}*/metrics_renders.json outputs/sr4d_bones/{zju,dance}_r15h35*/*/metrics_renders.json; do
  [ -f "$M" ] || continue; D=$(dirname $M); [ -f $D/metrics_band_renders.json ] && continue
  read R SET <<< "$(root_of $D)"; [ -z "$SET" ] && { case $D in *gen48_ho8*) SET=gen48_ho8 ;; *r15h35*) SET=r15h35 ;; esac; }
  echo "== $D ($R $SET)"; python tools/rescore_band.py --data_root $R --setting $SET --renders $D/renders
done
