#!/usr/bin/env bash
# D-041: rebuild the card figures + orbit videos of CHECK / EXP-001..003 with the unified names and orange/blue share colours.
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export PYTHONPATH=.
D=data_dev/fouranyone_gen/dance5885633_gen48_2ring; L=outputs/layered; K=$L/_links/dance_train4; O2=outputs/orbits/cmp1015/dance; O3=outputs/orbits/exp003
FW="FTGS++ (공식, 흰 배경)"; FB="FTGS++ (공식, 검정 배경)"
# CHECK
G=docs/gallery/pub2/exp/check2k; rm -f $G/index.json
python tools/exp_figure.py --root $D --cams g12 g33 g15 g42 --frames 30 --out $G --prefix chk \
 --methods "SR4D=$L/check2k_both/chk_inner" "FTGS-only 2k=$L/check2k_outer/chk_outer" "New 2k=$L/check2k_both/chk_both" "New-drop 2k=$L/check2k_drop05/chk_both" \
 --share "New=$L/check2k_both/chk_both" "New-drop=$L/check2k_drop05/chk_both" | tail -c 100
# EXP-001
E=$L/exp001_dance_ho8_outer; G=docs/gallery/pub2/exp/exp001; rm -f $G/index.json
python tools/exp_figure.py --root $D --cams g00 g06 g12 g18 g27 g33 g39 g45 --out $G --prefix held --methods "FTGS-only=$E/renders_outer" "$FW=outputs/ftgspp_white_official/dance_gen48_ho8/renders" "$FB=outputs/ftgspp_black_official/dance_gen48_ho8/renders" "SR4D=outputs/sr4d_final/dance_gen48_ho8_nol4/renders" | tail -c 100
python tools/exp_figure.py --root $D --cams g03 g15 g30 g42 --out $G --prefix train --methods "FTGS-only=$E/train_outer" "$FW=$K/ftgspp_white" "$FB=$K/ftgspp_black" "SR4D=$K/nol4" | tail -c 100
for e in 55 05; do c=$([ $e = 55 ] && echo "55° (위 링 35°보다 20° 위)" || echo "5° (아래 링 15°보다 10° 아래)")
python tools/exp_orbit.py --out $G/orbit_e$e.mp4 --cap "$c" --methods "FTGS-only=$E/orbit_e${e}_outer" "FTGS++ 공식 흰=$O2/orbit_e$e/ftgspp_white" "FTGS++ 공식 검정=$O2/orbit_e$e/ftgspp_black" "SR4D=$O2/orbit_e$e/nol4"; done
# EXP-002
B=$L/exp002b_dance_ho8_both_drop05; A=$L/exp002_dance_ho8_both; G=docs/gallery/pub2/exp/exp002; rm -f $G/index.json
python tools/exp_figure.py --root $D --cams g00 g06 g12 g18 g27 g33 g39 g45 --out $G --prefix held --methods "New-drop=$B/renders_both" "New=$A/renders_both" "FTGS-only=$E/renders_outer" "SR4D=outputs/sr4d_final/dance_gen48_ho8_nol4/renders" "DualGS=outputs/dualgs/dance_gen48_ho8/renders" "$FW=outputs/ftgspp_white_official/dance_gen48_ho8/renders" "4DGS dynerf=outputs/4dgs_dynerf/dance_gen48_ho8/renders" --share "New-drop=$B/renders_both" "New=$A/renders_both" | tail -c 100
python tools/exp_figure.py --root $D --cams g03 g15 g30 g42 --out $G --prefix train --methods "New-drop=$B/train_both" "FTGS-only=$E/train_outer" "SR4D=$K/nol4" "DualGS=$K/dualgs" "$FW=$K/ftgspp_white" "4DGS dynerf=$K/4dgs_dynerf" --share "New-drop=$B/train_both" | tail -c 100
for e in 55 05; do python tools/share_frames.py $B/orbit_e${e}_both $B/orbit_e${e}_share; c=$([ $e = 55 ] && echo "55° (위 링 35°보다 20° 위)" || echo "5° (아래 링 15°보다 10° 아래)")
python tools/exp_orbit.py --out $G/orbit_e$e.mp4 --cap "$c" --methods "New-drop=$B/orbit_e${e}_both" "층 기여 New-drop (주황 바깥·파랑 안쪽)=$B/orbit_e${e}_share" "FTGS-only=$E/orbit_e${e}_outer" "SR4D=$O2/orbit_e$e/nol4" "DualGS=$O2/orbit_e$e/dualgs" "FTGS++ 공식 흰=$O2/orbit_e$e/ftgspp_white" "4DGS dynerf=$O2/orbit_e$e/4dgs_dynerf"; done
# EXP-003
X=$L/exp003b_dance_r15h35_both_drop05; Y=$L/exp003_dance_r15h35_both; Z=$L/exp003_dance_r15h35_outer; G=docs/gallery/pub2/exp/exp003; rm -f $G/index.json
python tools/exp_figure.py --root $D --cams g24 g28 g32 g36 g40 g44 --out $G --prefix held --methods "New-drop=$X/renders_both" "New=$Y/renders_both" "FTGS-only=$Z/renders_outer" "SR4D=outputs/sr4d_bones/dance_r15h35/v0_posea/renders" "DualGS=outputs/dualgs/dance_r15h35/renders" "$FB=outputs/ftgspp_black_official/dance_r15h35/renders" "4DGS=outputs/4dgs/dance_r15h35/renders" --share "New-drop=$X/renders_both" "New=$Y/renders_both" | tail -c 100
python tools/exp_figure.py --root $D --cams g03 g09 g15 g21 --out $G --prefix train --methods "New-drop=$X/train_both" "New=$Y/train_both" "FTGS-only=$Z/train_outer" --share "New-drop=$X/train_both" "New=$Y/train_both" | tail -c 100
for e in 35 05; do for r in $X $Y; do python tools/share_frames.py $r/orbit_e${e}_both $r/orbit_e${e}_share; done; c=$([ $e = 35 ] && echo "35° (학습 링 15°보다 20° 위 = 채점 링 높이)" || echo "5° (학습 링 15°보다 10° 아래)")
python tools/exp_orbit.py --scale 0.5 --out $G/orbit_e$e.mp4 --cap "$c" --methods "New-drop=$X/orbit_e${e}_both" "층 기여 New-drop=$X/orbit_e${e}_share" "New=$Y/orbit_e${e}_both" "FTGS-only=$Z/orbit_e${e}_outer" "SR4D=$O3/orbit_e$e/sr4d" "DualGS=$O3/orbit_e$e/dualgs" "FTGS++ 공식 검정=$O3/orbit_e$e/ftgspp_black" "4DGS=$O3/orbit_e$e/4dgs"; done
echo REBUILDDONE
