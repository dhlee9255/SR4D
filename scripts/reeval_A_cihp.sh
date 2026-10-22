#!/usr/bin/env bash
# Re-render every A-stage (mask_cihp) model at all cameras WITH alpha, then evaluate the held-out cams (JSON only:
# metrics_renders_all.json incl. fg-interior PSNR, tOF, out-of-mask alpha). usage: reeval_A_cihp.sh [settings...]
set -euo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"
source ~/miniconda3/etc/profile.d/conda.sh
for SET in "${@:-real12 real8 real_all}"; do
  scripts/render_allcams.sh $SET
  conda activate sr4d
  for d in outputs/sr4d/387_cihp/$SET outputs/4dgs/387_cihp/$SET outputs/ftgspp_black/387_cihp/$SET outputs/dualgs/387_cihp/$SET; do
    [ -d $d/renders_all ] && python -m sr4d.eval.metrics --data_root data_dev/zju_mocap/387_cihp --setting $SET --renders $d/renders_all --method $(basename $(dirname $(dirname $d))) --no_csv | grep test_
  done
done
