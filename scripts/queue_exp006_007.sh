#!/usr/bin/env bash
# EXP-006 (ZJU near/far) + EXP-007 (dance outer seeds 1, 2), D-042. usage: queue_exp006_007.sh <N> ; run inside tmux.
# 12 runs, 3 sequential per GPU (one layered training per GPU). Each run: scripts/run_exp005.sh with PLAIN=1.
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"; N=$1
ZD=data_dev/zju_mocap_gen/387_gen48_2ring; DD=data_dev/fouranyone_gen/dance5885633_gen48_2ring
ZN=outputs/sr4d_final/zju_gen48_ho8/ckpt_final.pt; ZF=outputs/sr4d_bones/zju_r15h35/v0_posea/ckpt_final.pt
DN=outputs/sr4d_final/dance_gen48_ho8_nol4/ckpt_final.pt; DF=outputs/sr4d_bones/dance_r15h35/v0_posea/ckpt_final.pt
lane() {  # gpu then "name root setting inner minviews seed" ...
  local g=$1; shift
  for spec in "$@"; do set -- $spec; PLAIN=1 bash scripts/run_exp005.sh $g "$@"; done
  echo "LANE $g DONE $(date +%H:%M)" >> logs/queue_exp006_007.log
}
lane 0 "exp006_zju_ho8_n$N $ZD gen48_ho8 $ZN $N 0" "exp007_dance_ho8_n${N}_s1 $DD gen48_ho8 $DN $N 1" "exp007_dance_ho8_drop_s1 $DD gen48_ho8 $DN 0 1" &
lane 1 "exp006_zju_r15h35_n$N $ZD r15h35 $ZF $N 0" "exp007_dance_r15h35_n${N}_s1 $DD r15h35 $DF $N 1" "exp007_dance_r15h35_drop_s1 $DD r15h35 $DF 0 1" &
lane 2 "exp006_zju_ho8_drop $ZD gen48_ho8 $ZN 0 0" "exp007_dance_ho8_n${N}_s2 $DD gen48_ho8 $DN $N 2" "exp007_dance_ho8_drop_s2 $DD gen48_ho8 $DN 0 2" &
lane 3 "exp006_zju_r15h35_drop $ZD r15h35 $ZF 0 0" "exp007_dance_r15h35_n${N}_s2 $DD r15h35 $DF $N 2" "exp007_dance_r15h35_drop_s2 $DD r15h35 $DF 0 2" &
wait; echo "QUEUEDONE $(date +%H:%M)" >> logs/queue_exp006_007.log
