#!/usr/bin/env bash
# EXP-009 card figures (D-044), generation seed 42: frame 30, near g12 / far g28 for dance and ZJU + dance crown crops
# (observe red smear on New-T, purple smear on FTGS++ transparent).
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sr4d
export PYTHONPATH=.
O=outputs; G=docs/gallery/pub2/exp/exp009; mkdir -p $G
ms() { local K=$1 S=$2 NT=$O/layered/exp009_$1_$2; [ $K$S = dancegen48_ho8 ] && NT=$O/layered/exp008a_dance_ho8
  echo "SR4D=$O/sr4d_final/${K}_${S}_nol4/renders|New-T=$NT/renders_both:share|DualGS=$O/dualgs/${K}_$S/renders|FTGS++ 투명=$O/ftgspp_transparent/${K}_$S/renders|4DGS dynerf=$O/4dgs_dynerf/${K}_$S/renders"; }
DD=data_dev/fouranyone_gen/dance5885633_gen48_2ring; DZ=data_dev/zju_mocap_gen/387_gen48_2ring
python tools/exp_top.py --out $G/sum_dance.jpg --frame 30 \
  --row "dance 가까운 시점 · gen48_ho8 홀드아웃 g12|$DD|g12|$(ms dance gen48_ho8)" --row "dance 먼 시점 · r15h35 35° 링 g28|$DD|g28|$(ms dance r15h35)"
python tools/exp_top.py --out $G/sum_ZJU.jpg --frame 30 \
  --row "ZJU 가까운 시점 · gen48_ho8 홀드아웃 g12|$DZ|g12|$(ms zju gen48_ho8)" --row "ZJU 먼 시점 · r15h35 35° 링 g28|$DZ|g28|$(ms zju r15h35)"
for KS in "dance gen48_ho8 g12 $DD" "dance r15h35 g28 $DD" "zju r15h35 g28 $DZ"; do set -- $KS
  ms $1 $2 | sed "s/:share//g" | tr "|" "\n" | xargs -d '\n' python tools/crown_crop.py --root $4 --cam $3 --frames 10 30 50 --out $G/crown_$1_$2_$3.jpg --methods
done
echo FIGDONE
# output files renamed after run: crown_<K>_<S>_<cam>.jpg -> crown_dance먼g28.jpg etc. (gallery button = text after last _)
