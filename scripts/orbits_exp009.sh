#!/usr/bin/env bash
# EXP-009 orbit videos with New-T (dance, gen seed 42): near (gen48_ho8, rings 15/35 -> 55 and 5 deg) and far (r15h35, ring 15 -> 35 and 5 deg).
# Renders skipped when present. usage: orbits_exp009.sh <gpu>   (4DGS dynerf far needs outputs/4dgs_dynerf/dance_r15h35/model)
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
export CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=.
FT=/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus
D=data_dev/fouranyone_gen/dance5885633_gen48_2ring; G=docs/gallery/pub2/exp/exp009; mkdir -p $G
for V in near far; do
  if [ $V = near ]; then S=gen48_ho8; O=outputs/orbits/cmp1015/dance; NT=outputs/layered/exp008a_dance_ho8; SR=outputs/sr4d_final/dance_gen48_ho8_nol4; ES="55 05"
  else S=r15h35; O=outputs/orbits/exp003; NT=outputs/layered/exp009_dance_r15h35; SR=outputs/sr4d_final/dance_r15h35_nol4; ES="35 05"; fi
  for e in $ES; do OR=$O/orbit_e$e
    [ -f $NT/orbit_e${e}_both/0000.png ] || python -m sr4d.layered.render --run $NT --mode both --out $NT/orbit_e${e}_both --orbit $OR
    python tools/share_frames.py $NT/orbit_e${e}_both $NT/orbit_e${e}_share
    [ -f $OR/sr4d_nol4/0000.png ] || python -m sr4d.eval.render_cams --ckpt $SR/ckpt_final.pt --out $OR --sub sr4d_nol4 --orbit $OR || echo "FAIL sr4d $V $e"
    [ -f $OR/dualgs/0000.png ] || python scripts/baselines/render_ply.py --ply_pattern "outputs/dualgs/dance_$S/model/ckt/point_cloud_{rel}.ply" --root $D --setting $S --out $OR/dualgs --orbit $OR || echo "FAIL dualgs $V $e"
    F=outputs/ftgspp_transparent/dance_$S
    [ -f $OR/ftgspp_transparent/0000.png ] || PATH=$FT/.venv/bin:$PATH PYTHONPATH=$FT:$SR4D $FT/.venv/bin/python scripts/baselines/ftgspp_render.py --run $F/run --root $D --setting $S --export_meta $F/export_meta.json --out $OR/ftgspp_transparent --orbit $OR || echo "FAIL ftt $V $e"
    if [ ! -f $OR/4dgs_dynerf/0000.png ] && [ -f outputs/4dgs_dynerf/dance_$S/renders/g24/000000.png -o $S = gen48_ho8 ]; then
      ( conda activate hust4dgs; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8; cd third_party/4DGaussians
        python $SR4D/scripts/baselines/fourdgs_render_native.py -m $SR4D/outputs/4dgs_dynerf/dance_$S/model --configs $SR4D/configs/fourdgs/dynerf_default_nodl.py \
          --root $SR4D/$D --setting $S --out $SR4D/$OR/4dgs_dynerf --orbit $SR4D/$OR ) || echo "FAIL 4dgs $V $e"
    fi
    case $V$e in near55) c="55° (위 학습 링 35°보다 20° 위)";; near05) c="5° (아래 학습 링 15°보다 10° 아래)";; far35) c="35° (학습 링 15°보다 20° 위 = 채점 링)";; far05) c="5° (학습 링 15°보다 10° 아래)";; esac
    python tools/exp_orbit.py --scale 0.5 --out $G/orbit_${V}_e$e.mp4 --cap "dance $([ $V = near ] && echo 가까운 || echo 먼) 시점 · $c" --methods \
      "New-T=$NT/orbit_e${e}_both" "New-T 층 기여=$NT/orbit_e${e}_share" "SR4D=$OR/sr4d_nol4" "DualGS=$OR/dualgs" "FTGS++ 투명=$OR/ftgspp_transparent" "4DGS dynerf=$OR/4dgs_dynerf"
  done
done
echo ORBITDONE
