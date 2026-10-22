#!/usr/bin/env bash
# 10/15 comparison renders: orbit_e55, orbit_e05 (novel elevations) and train4 (4 training views x frames 0/30/59) for one
# dataset and a group of methods. usage: render_cmp1015.sh <zju|dance> <sr4d|base> ; log to file, run inside tmux.
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; K=$1; GRP=$2
FT=/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus
[ $K = zju ] && R=data_dev/zju_mocap_gen/387_gen48_2ring || R=data_dev/fouranyone_gen/dance5885633_gen48_2ring
O=outputs/orbits/cmp1015/$K; source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
if [ $K = dance ]; then SR="nol4=dance_gen48_ho8_nol4 nol4x2=dance_gen48_ho8_nol4x2 s7a=dance_gen48_ho8_nol4_s7a s7b=dance_gen48_ho8_nol4_s7b ${EXTRA:-}"
else SR="base=zju_gen48_ho8 cam0=zju_gen48_ho8_cam0"; fi
for orb in train4 orbit_e55 orbit_e05; do
  if [ $GRP = sr4d ]; then
    for kv in $SR; do n=${kv%%=*}; d=outputs/sr4d_final/${kv#*=}
      [ -f $O/$orb/$n/0000.png ] || python -m sr4d.eval.render_cams --ckpt $d/ckpt_final.pt --out $O/$orb --sub $n --orbit $O/$orb || echo "FAIL $n $orb"; done
    [ -f $O/$orb/dualgs/0000.png ] || python scripts/baselines/render_ply.py --ply_pattern "outputs/dualgs/${K}_gen48_ho8/model/ckt/point_cloud_{rel}.ply" --root $R --setting gen48_ho8 --out $O/$orb/dualgs --orbit $O/$orb || echo "FAIL dualgs $orb"
  else
    for bg in black white; do F=outputs/ftgspp_${bg}_official/${K}_gen48_ho8
      [ -f $O/$orb/ftgspp_$bg/0000.png ] || PATH=$FT/.venv/bin:$PATH PYTHONPATH=$FT:$SR4D $FT/.venv/bin/python scripts/baselines/ftgspp_render.py --run $F/run --root $R --setting gen48_ho8 --export_meta $F/export_meta.json --out $O/$orb/ftgspp_$bg --orbit $O/$orb || echo "FAIL ftgspp_$bg $orb"; done
    if [ ! -f $O/$orb/4dgs_dynerf/0000.png ]; then
      ( conda activate hust4dgs; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8; cd third_party/4DGaussians
        python $SR4D/scripts/baselines/fourdgs_render_native.py -m $SR4D/outputs/4dgs_dynerf/${K}_gen48_ho8/model --configs $SR4D/configs/fourdgs/dynerf_default_nodl.py \
          --root $SR4D/$R --setting gen48_ho8 --out $SR4D/$O/$orb/4dgs_dynerf --orbit $SR4D/$O/$orb ) || echo "FAIL 4dgs $orb"
    fi
  fi
done
echo "ALL DONE $K $GRP $(date +%H:%M)"
