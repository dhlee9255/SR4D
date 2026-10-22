#!/usr/bin/env bash
# D-040 launcher: run_layered.sh <gpu> <name> <data root> <setting> <inner ckpt> [extra train args]; run inside tmux.
# Trains outputs/layered/<name>, log logs/layered_<name>.log, pid logs/layered_<name>.pid
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; G=$1; N=$2; R=$3; S=$4; I=$5; shift 5
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
echo $$ > logs/layered_$N.pid
CUDA_VISIBLE_DEVICES=$G PYTHONPATH=. python -m sr4d.layered.train --root $R --setting $S --inner $I --out outputs/layered/$N "$@" > logs/layered_$N.log 2>&1
echo "EXIT $? $(date +%H:%M)" >> logs/layered_$N.log
