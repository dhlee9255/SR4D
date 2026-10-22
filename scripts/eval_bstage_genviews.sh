#!/usr/bin/env bash
# B-stage self-evaluation: render the trained gen24 models at the 24 GENERATED training views and score them against
# the generated frames (GT = what each model was optimised on). usage: scripts/eval_bstage_genviews.sh
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"
ROOT=$(realpath data_dev/zju_mocap_gen/387_gen24_gv); SET=gen24; PHASE=${PHASE:-B_genview}; TAG=gv
FT=/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus
source ~/miniconda3/etc/profile.d/conda.sh
ev() { conda activate sr4d; python -m sr4d.eval.metrics --phase "$PHASE" --data_root "$ROOT" --setting $SET --renders "$1" --method "$2" --notes "GT = generated views (train views, self-eval)"; }
conda activate sr4d
S=outputs/sr4d/387_gen/gen24; python -m sr4d.eval.render_cams --ckpt $S/ckpt_final.pt --out $S/eval_$TAG --data_root "$ROOT" --splits test_interp
ev $S/eval_$TAG/renders sr4d
F=outputs/ftgspp_black_official/387_gen/gen24
PATH=$FT/.venv/bin:$PATH PYTHONPATH=$FT:$SR4D $FT/.venv/bin/python scripts/baselines/ftgspp_render.py --run $F/run --root "$ROOT" --setting $SET --export_meta $F/export_meta.json --out $F/renders_$TAG
ev $F/renders_$TAG ftgspp_black_official
D=$(realpath outputs/4dgs/387_gen/gen24)
NF=$(python -c "import json;print(len(json.load(open('$D/export_meta.json'))['frames']))")
[ -d $D/data/ims ] || (cd tools && python export_4dgs.py --root "$ROOT" --setting $SET --out "$D")
conda activate hust4dgs; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
[ -f $D/frames/time_00000.ply ] || (cd third_party/4DGaussians && python "$SR4D/scripts/baselines/fourdgs_export_frames.py" -m "$D/model" --num_frames "$NF" --out "$D/frames")
conda activate sr4d
python scripts/baselines/render_ply.py --ply_pattern "$D/frames/time_{rel:05d}.ply" --root "$ROOT" --setting $SET --out $D/renders_$TAG
ev $D/renders_$TAG 4dgs
rm -rf $D/frames $D/data/ims
