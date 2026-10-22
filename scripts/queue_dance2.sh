#!/usr/bin/env bash
# Dance gen24 (generated-only) SR4D bone variants after the ZJU gen24 job on the same GPU; then jitter diag + orbit.
# usage: queue_dance2.sh <gpu> <tag> <wait_regex>
set -uo pipefail
cd "$(dirname "$0")/.."
G=$1; TAG=$2; W=$3
while pgrep -f "$W" >/dev/null; do sleep 60; done
case $TAG in free) AB="" ;; *) AB=configs/ablations/bones_$TAG.yaml ;; esac
OUT=outputs/sr4d_bones/dance_gen24/$TAG
export CUDA_VISIBLE_DEVICES=$G PHASE=Bdance
METHOD=sr4d_$TAG NOTES="dance gen24 self-eval, bones $TAG" scripts/run_sr4d.sh configs/dance_gen_dev.yaml $OUT ${AB:+$AB} > logs/dance_sr4d_$TAG.log 2>&1
scripts/post_bones.sh $OUT outputs/orbits/dance_gen24 dance_$TAG >> logs/dance_sr4d_$TAG.log 2>&1
