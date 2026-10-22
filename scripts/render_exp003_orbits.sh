#!/usr/bin/env bash
# EXP-003 (D-040): orbits at 35 deg (+20 above the 15-deg training ring) and 5 deg (-10) for the r15h35 baselines.
# usage: render_exp003_orbits.sh <gpu>; run inside tmux. -> outputs/orbits/exp003/orbit_e{35,05}/<method>/<i>.png
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; export CUDA_VISIBLE_DEVICES=$1
R=data_dev/fouranyone_gen/dance5885633_gen48_2ring; O=outputs/orbits/exp003; FT=/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
for e in 35 05; do [ -f $O/orbit_e$e/orbit.json ] || python tools/make_orbit.py --root $R --setting r15h35 --out $O/orbit_e$e --abs_elev_deg $e; done
D4=outputs/4dgs/dance_r15h35
if [ ! -f $D4/frames/time_00059.ply ]; then
  ( conda activate hust4dgs; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8; cd third_party/4DGaussians
    python $SR4D/scripts/baselines/fourdgs_export_frames.py -m $SR4D/$D4/model --num_frames 60 --out $SR4D/$D4/frames ) || echo "FAIL 4dgs export"
fi
for e in 35 05; do orb=$O/orbit_e$e
  [ -f $orb/sr4d/0000.png ] || PYTHONPATH=. python -m sr4d.eval.render_cams --ckpt outputs/sr4d_bones/dance_r15h35/v0_posea/ckpt_final.pt --out $orb --sub sr4d --orbit $orb || echo "FAIL sr4d $e"
  [ -f $orb/dualgs/0000.png ] || PYTHONPATH=. python scripts/baselines/render_ply.py --ply_pattern "outputs/dualgs/dance_r15h35/model/ckt/point_cloud_{rel}.ply" --root $R --setting r15h35 --out $orb/dualgs --orbit $orb || echo "FAIL dualgs $e"
  [ -f $orb/4dgs/0000.png ] || PYTHONPATH=. python scripts/baselines/render_ply.py --ply_pattern "$D4/frames/time_{rel:05d}.ply" --root $R --setting r15h35 --out $orb/4dgs --orbit $orb || echo "FAIL 4dgs $e"
  F=outputs/ftgspp_black_official/dance_r15h35
  [ -f $orb/ftgspp_black/0000.png ] || PATH=$FT/.venv/bin:$PATH PYTHONPATH=$FT:$SR4D $FT/.venv/bin/python scripts/baselines/ftgspp_render.py --run $F/run --root $R --setting r15h35 --export_meta $F/export_meta.json --out $orb/ftgspp_black --orbit $orb || echo "FAIL ftgspp $e"
done
rm -rf $D4/frames   # regenerable; disk is tight
echo "ORBDONE $(date +%H:%M)"
