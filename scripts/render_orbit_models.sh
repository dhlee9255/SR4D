#!/usr/bin/env bash
# Orbit (NVS) videos for trained models. usage: render_orbit_models.sh <orbit_dir> <data_root> <setting> <sr4d_dir> <4dgs_dir> <ftgspp_dir> <out_dir>
# any model dir may be '-' to skip. Frames -> <out_dir>/<method>/<i:04d>.png, plus <out_dir>/<method>.mp4
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"
ORB=$(realpath "$1"); ROOT=$(realpath "$2"); SET=$3; S=$4; D=$5; F=$6; OUT=$(realpath -m "$7")
FT=/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
mkdir -p "$OUT"
[ "$S" = - ] || python -m sr4d.eval.render_cams --ckpt $S/ckpt_final.pt --out "$OUT" --sub sr4d --orbit "$ORB"
[ "$F" = - ] || PATH=$FT/.venv/bin:$PATH PYTHONPATH=$FT:$SR4D $FT/.venv/bin/python scripts/baselines/ftgspp_render.py --run $F/run --root "$ROOT" --setting $SET --export_meta $F/export_meta.json --out "$OUT/ftgspp" --orbit "$ORB"
if [ "$D" != - ]; then
  D=$(realpath $D); NF=$(python -c "import json;print(len(json.load(open('$D/export_meta.json'))['frames']))")
  [ -d $D/data/ims ] || (cd tools && python export_4dgs.py --root "$ROOT" --setting $SET --out "$D")
  conda activate hust4dgs; export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
  [ -f $D/frames/time_00000.ply ] || (cd third_party/4DGaussians && python "$SR4D/scripts/baselines/fourdgs_export_frames.py" -m "$D/model" --num_frames "$NF" --out "$D/frames")
  conda activate sr4d
  python scripts/baselines/render_ply.py --ply_pattern "$D/frames/time_{rel:05d}.ply" --root "$ROOT" --setting $SET --out "$OUT/4dgs" --orbit "$ORB"
  rm -rf $D/frames $D/data/ims
fi
for m in sr4d 4dgs ftgspp; do
  [ -d "$OUT/$m" ] && ffmpeg -y -loglevel error -framerate 24 -i "$OUT/$m/%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/$m.mp4" || true
done
echo done
