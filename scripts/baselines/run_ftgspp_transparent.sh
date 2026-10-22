#!/usr/bin/env bash
# EXP-008 (D-043) "FTGS++ (투명 배경)": official FTGS++ pipeline (masked-on-black export, official init INIT_FIX=0, official
# config) but trained with random background + alpha-vs-mask loss (third_party/ftgspp_transparent). Renders on white.
# usage: run_ftgspp_transparent.sh <unified_root> <setting> <out_dir> [iters]
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$(realpath "$1"); SET=$2; OUT=$(realpath -m "$3"); ITERS=${4:-30000}
FT=/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus; PY=$FT/.venv/bin/python
export PATH=$FT/.venv/bin:/usr/local/cuda/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sr4d
$HOME/miniconda3/envs/sr4d/bin/python "$SR4D/tools/export_ftgspp.py" --root "$ROOT" --setting "$SET" --out "$OUT" --iters "$ITERS" --bg black
$HOME/miniconda3/envs/sr4d/bin/python "$SR4D/scripts/baselines/ftgspp_masks.py" "$ROOT" "$OUT/export_meta.json" "$OUT/masks.npy"
CFG=$OUT/config.toml; RUN=$OUT/run
cd "$FT"
T0=$(date +%s)
for m in ftgspp.data.extract ftgspp.data.sfm ftgspp.data.prep; do $PY -m $m "$CFG"; done
$PY -m ftgspp.init.points "$CFG"
$PY -m ftgspp.init "$CFG" "$RUN"
FTGS_MASKS=$OUT/masks.npy PYTHONPATH=$FT $PY "$SR4D/scripts/baselines/ftgspp_train_transparent.py" "$CFG" "$RUN" --seed-offset "${TRAIN_SEED:-0}"
T1=$(date +%s)
PYTHONPATH=$FT:$SR4D $PY "$SR4D/scripts/baselines/ftgspp_render.py" --run "$RUN" --root "$ROOT" --setting "$SET" \
  --export_meta "$OUT/export_meta.json" --out "$OUT/renders" --composite white
cd "$SR4D"
$HOME/miniconda3/envs/sr4d/bin/python -m sr4d.eval.metrics --phase "${PHASE:-B_g48}" --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" --method ftgspp_transparent \
  --train_min $(( (T1 - T0) / 60 )) --n_gauss 500000 --notes "${NOTES:-FTGS++ official pipeline + random bg + alpha mask loss 0.05 (D-043)}"
[ "$SET" = r15h35 ] && $HOME/miniconda3/envs/sr4d/bin/python -m sr4d.eval.covis_metrics --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" | grep extrap
rm -rf "$OUT/work/_memmap" "$OUT/work/_extracted" "$OUT/data/images" "$OUT/masks.npy"
echo "FTGSDONE $(date +%H:%M)"
