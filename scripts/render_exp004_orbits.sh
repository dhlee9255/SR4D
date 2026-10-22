#!/usr/bin/env bash
# EXP-004: New-fade (th1 th2) orbit + 35-deg/train-view renders. usage: render_exp004_orbits.sh <gpu> <run> "<th1 th2>" <train cams "g03 g15"> <orbit dirs...>
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; G=$1; N=$2; TH=$3; TC=$4; shift 4; set -- "$@"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=.
O=outputs/layered/$N; T=$(echo $TH | tr ' ' _)
python -m sr4d.layered.render --run $O --mode both --out $O/train_fade$T --cams $TC --frames 0 30 59 --fade $TH
for orb in "$@"; do python -m sr4d.layered.render --run $O --mode both --out $O/$(basename $orb)_fade$T --orbit $orb --fade $TH; done
echo "FIGDONE $N $T $(date +%H:%M)"
