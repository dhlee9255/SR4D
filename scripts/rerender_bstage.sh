#!/usr/bin/env bash
# Re-render already-trained B-stage models at re-registered REAL evaluation cameras (training cams g* are unchanged,
# so no retraining is needed) and evaluate into results/<PHASE>_gen24.csv.
# usage: rerender_bstage.sh <data_root> <tag>      e.g. data_dev/zju_mocap_gen/387_gen24_v2 v2
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"
ROOT=$(realpath "$1"); TAG=$2; PHASE=${PHASE:-B_$TAG}; SET=gen24
FT=/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus
source ~/miniconda3/etc/profile.d/conda.sh
ev() { conda activate sr4d; python -m sr4d.eval.metrics --phase "$PHASE" --data_root "$ROOT" --setting $SET --renders "$1" --method "$2" --notes "eval cams: $TAG registration"; }
ONLY=${ONLY:-all}
conda activate sr4d
if [ "$ONLY" = all ]; then
# SR4D
S=outputs/sr4d/387_gen/gen24; python -m sr4d.eval.render_cams --ckpt $S/ckpt_final.pt --out $S/eval_$TAG --data_root "$ROOT"
ev $S/eval_$TAG/renders sr4d
# FTGS++
F=outputs/ftgspp/387_gen/gen24
PYTHONPATH=$FT:$SR4D $FT/.venv/bin/python scripts/baselines/ftgspp_render.py --run $F/run --root "$ROOT" --setting $SET --export_meta $F/export_meta.json --out $F/renders_$TAG
ev $F/renders_$TAG ftgspp
fi
# 4DGS (per-frame PLYs regenerated from the trained deformation model)
D=$(realpath outputs/4dgs/387_gen/gen24)
NF=$(python -c "import json;print(len(json.load(open('$D/export_meta.json'))['frames']))")
[ -d $D/data/ims ] || (cd tools && python export_4dgs.py --root "$ROOT" --setting $SET --out "$D")   # training images for the scene loader
conda activate hust4dgs; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
[ -f $D/frames/time_00000.ply ] || (cd third_party/4DGaussians && python "$SR4D/scripts/baselines/fourdgs_export_frames.py" -m "$D/model" --num_frames "$NF" --out "$D/frames")
conda activate sr4d
python scripts/baselines/render_ply.py --ply_pattern "$D/frames/time_{rel:05d}.ply" --root "$ROOT" --setting $SET --out $D/renders_$TAG
ev $D/renders_$TAG 4dgs
[ "${KEEP_FRAMES:-0}" = 1 ] || rm -rf $D/frames $D/data/ims
