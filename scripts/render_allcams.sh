#!/usr/bin/env bash
# Render trained A-stage models at EVERY camera (train + held-out) for the viewer. usage: render_allcams.sh <setting>
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"
SET=$1; ROOT=$(realpath data_dev/zju_mocap/387_cihp_allcams)
FT=/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
S=outputs/sr4d/387_cihp/$SET
[ -f $S/ckpt_final.pt ] && python -m sr4d.eval.render_cams --ckpt $S/ckpt_final.pt --out $S --data_root "$ROOT" --splits test_interp --sub renders_all
F=outputs/ftgspp_black/387_cihp/$SET
[ -d $F/run ] && PATH=$FT/.venv/bin:$PATH PYTHONPATH=$FT:$SR4D $FT/.venv/bin/python scripts/baselines/ftgspp_render.py --run $F/run --root "$ROOT" --setting $SET --export_meta $F/export_meta.json --out $F/renders_all
G=outputs/dualgs/387_cihp/$SET
[ -f $G/model/ckt/point_cloud_59.ply ] && python scripts/baselines/render_ply.py --ply_pattern "$G/model/ckt/point_cloud_{rel}.ply" --root "$ROOT" --setting $SET --out $G/renders_all
D=outputs/4dgs/387_cihp/$SET
if [ -f $D/metrics_renders.json ]; then
  D=$(realpath $D); NF=$(python -c "import json;print(len(json.load(open('$D/export_meta.json'))['frames']))")
  [ -d $D/data/ims ] || (cd tools && python export_4dgs.py --root "$(realpath ../data_dev/zju_mocap/387_cihp)" --setting $SET --out "$D")
  conda activate hust4dgs; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
  [ -f $D/frames/time_00000.ply ] || (cd third_party/4DGaussians && python "$SR4D/scripts/baselines/fourdgs_export_frames.py" -m "$D/model" --num_frames "$NF" --out "$D/frames")
  conda activate sr4d
  python scripts/baselines/render_ply.py --ply_pattern "$D/frames/time_{rel:05d}.ply" --root "$ROOT" --setting $SET --out $D/renders_all
  rm -rf $D/frames $D/data/ims
fi
echo "done $SET"
