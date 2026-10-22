#!/usr/bin/env bash
# EXP-008 card figures (D-043): summary frame 30 (near g12 / far g28) + crown crops.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sr4d
export PYTHONPATH=.
D=data_dev/fouranyone_gen/dance5885633_gen48_2ring; O=outputs; L=outputs/layered; G=docs/gallery/pub2/exp/exp008; mkdir -p $G
NEAR="SR4D=$O/sr4d_final/dance_gen48_ho8_nol4/renders|New-fade 기존=$L/exp002b_dance_ho8_both_drop05/renders_fade10_25:share|(a) 투명=$L/exp008a_dance_ho8/renders_fade10_25:share|(b) 투명+모양=$L/exp008b_dance_ho8/renders_fade10_25:share|DualGS=$O/dualgs/dance_gen48_ho8/renders|FTGS++ 검정=$O/ftgspp_black_official/dance_gen48_ho8/renders|FTGS++ 흰=$O/ftgspp_white_official/dance_gen48_ho8/renders|FTGS++ 투명=$O/ftgspp_transparent/dance_gen48_ho8/renders"
FAR="SR4D=$O/sr4d_bones/dance_r15h35/v0_posea/renders|New-fade 기존=$L/exp003b_dance_r15h35_both_drop05/renders_fade10_25:share|(a) 투명=$L/exp008a_dance_r15h35/renders_fade10_25:share|(a) 투명, fade 없음=$L/exp008a_dance_r15h35/renders_both:share|(b) 투명+모양=$L/exp008b_dance_r15h35/renders_fade10_25:share|DualGS=$O/dualgs/dance_r15h35/renders|FTGS++ 검정=$O/ftgspp_black_official/dance_r15h35/renders|FTGS++ 투명=$O/ftgspp_transparent/dance_r15h35/renders"
python tools/exp_top.py --out $G/sum_frame30.jpg --frame 30 \
  --row "dance 먼 시점 · r15h35 35° 링 g28|$D|g28|$FAR" --row "dance 가까운 시점 · gen48_ho8 홀드아웃 g12|$D|g12|$NEAR"
sed "s/:share//g" <<<"$FAR" | tr "|" "\n" | xargs -d '\n' python tools/crown_crop.py --root $D --cam g28 --frames 10 30 50 --out $G/crown_far_g28.jpg --methods
sed "s/:share//g" <<<"$FAR" | tr "|" "\n" | xargs -d '\n' python tools/crown_crop.py --root $D --cam g26 --frames 10 30 50 --out $G/crown_far_g26.jpg --methods
sed "s/:share//g" <<<"$NEAR" | tr "|" "\n" | xargs -d '\n' python tools/crown_crop.py --root $D --cam g12 --frames 10 30 50 --out $G/crown_near_g12.jpg --methods
