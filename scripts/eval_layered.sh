#!/usr/bin/env bash
# D-040: render + score one layered run. eval_layered.sh <gpu> <run name> <mode both|outer|inner> [sub]; run inside tmux.
# -> outputs/layered/<run>/<sub or renders_<mode>>/<cam>/<frame>.png (+alpha, inner share), metrics_<sub>.json
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; G=$1; N=$2; M=$3; SUB=${4:-renders_$M}
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d; export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=.
O=outputs/layered/$N; R=$(python -c "import json;print(json.load(open('$O/args.json'))['root'])"); S=$(python -c "import json;print(json.load(open('$O/args.json'))['setting'])")
python -m sr4d.layered.render --run $O --mode $M --out $O/$SUB && \
python -m sr4d.eval.metrics --data_root $R --setting $S --renders $O/$SUB --method layered_${N}_$M --phase B --no_csv
echo "EVALDONE $N $M $? $(date +%H:%M)"
