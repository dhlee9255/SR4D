#!/usr/bin/env bash
# D-040: training-view + orbit renders of one layered run. render_layered_figs.sh <gpu> <run> <mode> <train cams "g03 g15"> <orbit dirs...>
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; G=$1; N=$2; M=$3; TC=$4; shift 4
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=.
O=outputs/layered/$N
python -m sr4d.layered.render --run $O --mode $M --out $O/train_$M --cams $TC --frames 0 30 59
for orb in "$@"; do python -m sr4d.layered.render --run $O --mode $M --out $O/$(basename $orb)_$M --orbit $orb; done
echo "FIGDONE $N $M $(date +%H:%M)"
