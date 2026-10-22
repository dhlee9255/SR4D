#!/usr/bin/env bash
# FreeTimeGS++ baseline. usage: run_ftgspp.sh <unified_root> <setting> <out_dir> [iters]
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$(realpath "$1"); SET=$2; OUT=$(realpath -m "$3"); ITERS=${4:-30000}
BG=${BG:-white}           # white (protocol, patched rasteriser) | black (masked, official training) | raw (no mask)
INIT_FIX=${INIT_FIX:-1}   # 1 = cheirality-fixed RoMa init (D-012), 0 = official ftgspp.init.points
FT=/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus
PY=$FT/.venv/bin/python
export PATH=$FT/.venv/bin:/usr/local/cuda/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sr4d
$HOME/miniconda3/envs/sr4d/bin/python "$SR4D/tools/export_ftgspp.py" --root "$ROOT" --setting "$SET" --out "$OUT" --iters "$ITERS" --bg "$BG"
CFG=$OUT/config.toml; RUN=$OUT/run
cd "$FT"
T0=$(date +%s)
for m in ftgspp.data.extract ftgspp.data.sfm ftgspp.data.prep; do $PY -m $m "$CFG"; done
if [ "$INIT_FIX" = 1 ]; then PYTHONPATH=$FT $PY "$SR4D/scripts/baselines/ftgspp_points_fixed.py" "$CFG"   # cheirality fix, D-012
else $PY -m ftgspp.init.points "$CFG"; fi
$PY -m ftgspp.init "$CFG" "$RUN"
if [ "${PRUNE_UNSEEN:-}" = 1 ]; then FTGS_WHITE=$([ "$BG" = white ] && echo 1 || echo 0) PYTHONPATH=$FT $PY "$SR4D/scripts/baselines/ftgspp_train_prune_unseen.py" "$CFG" "$RUN" --seed-offset "${TRAIN_SEED:-0}"   # D-029 appendix
elif [ "$BG" = white ]; then PYTHONPATH=$FT $PY "$SR4D/scripts/baselines/ftgspp_train_whitebg.py" "$CFG" "$RUN"   # white bg, D-011
else $PY -m ftgspp.train "$CFG" "$RUN" --seed-offset "${TRAIN_SEED:-0}"; fi
T1=$(date +%s)
PYTHONPATH=$FT:$SR4D $PY "$SR4D/scripts/baselines/ftgspp_render.py" --run "$RUN" --root "$ROOT" --setting "$SET" \
  --export_meta "$OUT/export_meta.json" --out "$OUT/renders" --composite $([ "$BG" = raw ] && echo none || echo white)
cd "$SR4D"
NG=$($PY -c "import torch;s=torch.load('$RUN/gaussians.pt',map_location='cpu',weights_only=False);print(next(v.shape[0] for k,v in (s.items() if isinstance(s,dict) else s.state_dict().items()) if 'means' in k))" 2>/dev/null || echo "")
$HOME/miniconda3/envs/sr4d/bin/python -m sr4d.eval.metrics --phase "${PHASE:-A}" --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" --method "${METHOD:-ftgspp}" \
  --train_min $(( (T1 - T0) / 60 )) --n_gauss "$NG" --notes "${NOTES:-4DAnyone ftgspp preset, white bg (patched), init cheirality fix}"
rm -rf "$OUT/work/_memmap" "$OUT/work/_extracted" "$OUT/data/images"   # regenerable; disk is tight
