#!/usr/bin/env bash
# After each ZJU bone-variant run frees its GPU, run the dance (4DAnyone gen24, generated-only) jobs there.
set -uo pipefail
cd "$(dirname "$0")/.."
R=data_dev/fouranyone_gen/dance5885633_gen24
export PHASE=Bdance
waitfor() { while pgrep -f "outputs/sr4d_bones/387_cihp_real12/$1" >/dev/null; do sleep 60; done; }
sr() {  # $1 = variant tag, $2 = ablation yaml or ""
  METHOD=sr4d_$1 NOTES="dance gen24 self-eval, bones $1" scripts/run_sr4d.sh configs/dance_gen_dev.yaml outputs/sr4d_bones/dance_gen24/$1 ${2:+$2} > logs/dance_sr4d_$1.log 2>&1
}
case $1 in
  0) waitfor v0; NOTES="dance gen24" scripts/baselines/run_4dgs.sh $R gen24 outputs/4dgs/dance_gen24 > logs/dance_4dgs.log 2>&1; sr v0 configs/ablations/bones_v0.yaml ;;
  1) waitfor v1; BG=black INIT_FIX=0 METHOD=ftgspp_black_official NOTES="dance gen24, legacy: black bg, official init" \
       scripts/baselines/run_ftgspp.sh $R gen24 outputs/ftgspp_black_official/dance_gen24 > logs/dance_ftgspp.log 2>&1; sr v1 configs/ablations/bones_v1.yaml ;;
  2) waitfor v2; sr v2 configs/ablations/bones_v2.yaml; sr free ;;
  3) waitfor v3; sr v3 configs/ablations/bones_v3.yaml ;;
esac
