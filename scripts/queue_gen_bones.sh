#!/usr/bin/env bash
# ZJU gen24 (generated-only, D-024) bone variants V0/V1/V2 + current free bones under the same config, then post-process.
# usage: queue_gen_bones.sh <gpu> <tag> [wait_pattern]
set -uo pipefail
cd "$(dirname "$0")/.."
G=$1; TAG=$2; W=${3:-}
[ -n "$W" ] && while pgrep -f "$W" >/dev/null; do sleep 60; done
case $TAG in free) AB="" ;; *) AB=configs/ablations/bones_$TAG.yaml ;; esac
OUT=outputs/sr4d_bones/387_gen24/$TAG
export CUDA_VISIBLE_DEVICES=$G PHASE=Bgen_bones
METHOD=sr4d_gen_$TAG NOTES="ZJU gen24 generated-only self-eval, bones $TAG" scripts/run_sr4d.sh configs/zju_gen_dev.yaml $OUT ${AB:+$AB} > logs/gen_bones_$TAG.log 2>&1
scripts/post_bones.sh $OUT outputs/orbits/387_gen24 gen_$TAG >> logs/gen_bones_$TAG.log 2>&1
