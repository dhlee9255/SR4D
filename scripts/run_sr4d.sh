#!/usr/bin/env bash
# Train + render + evaluate SR4D.  usage: scripts/run_sr4d.sh <config> <out_dir> [ablation.yaml ...] [-- key=value ...]
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sr4d
export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8
CFG=$1; OUT=$2; shift 2
ABL=(); OVR=()
while [[ $# -gt 0 ]]; do
  if [[ $1 == "--" ]]; then shift; OVR=("$@"); break; fi
  ABL+=("$1"); shift
done
METHOD=${METHOD:-sr4d}
python scripts/train_sr4d.py --config "$CFG" ${ABL[@]:+--ablation "${ABL[@]}"} --out "$OUT" ${OVR[@]:+"${OVR[@]}"}
python -m sr4d.eval.render_cams --ckpt "$OUT/ckpt_final.pt" --out "$OUT"
ROOT=$(python -c "import yaml,sys;print(yaml.safe_load(open('$OUT/config.yaml'))['data']['root'])")
SET=$(python -c "import yaml,sys;print(yaml.safe_load(open('$OUT/config.yaml'))['data']['setting'])")
TMIN=$(python -c "import torch;print(round(torch.load('$OUT/ckpt_final.pt',map_location='cpu',weights_only=False)['train_time']/60,1))")
NG=$(python -c "
import torch;s=torch.load('$OUT/ckpt_final.pt',map_location='cpu',weights_only=False)
print(sum(len(s[k]['params']['opacity']) for k in ('body','free') if s[k] is not None))")
python -m sr4d.eval.metrics --phase "${PHASE:-A}" --data_root "$ROOT" --setting "$SET" --renders "$OUT/renders" --method "$METHOD" \
  --train_min "$TMIN" --n_gauss "$NG" --notes "${NOTES:-}"
# disk is tight (docs/PROGRESS.md): latest == final after training; ablation/variant runs drop stage checkpoints too
rm -f "$OUT/ckpt_latest.pt"
if [[ "$METHOD" == sr4d_* ]]; then rm -f "$OUT"/ckpt_s2.pt "$OUT"/ckpt_s3.pt "$OUT"/ckpt_s4.pt; fi
