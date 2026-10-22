#!/usr/bin/env bash
# V3 add-on (user): GPU1 = ZJU gen24 V3 after gen V1, then dance V1; GPU2 = dance V3 after dance V2.
cd "$(dirname "$0")/.."
case $1 in
  1) while pgrep -f "[q]ueue_gen_bones.sh 1 v1" >/dev/null; do sleep 60; done
     scripts/queue_gen_bones.sh 1 v3
     scripts/queue_dance2.sh 1 v1 "[q]ueue_gen_bones.sh 1 " ;;
  2) while pgrep -f "[q]ueue_dance2.sh 2 v2" >/dev/null || pgrep -f "[q]ueue_gen_bones.sh 2 v2" >/dev/null; do sleep 60; done
     scripts/queue_dance2.sh 2 v3 "[q]ueue_dance2.sh 2 v2" ;;
esac
