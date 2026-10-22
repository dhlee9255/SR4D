#!/usr/bin/env bash
# User 2026-10-18 item 3: inner-layer SR4D retrain with the final setting (L4 off = configs/ablations/no_l4_all.yaml) for
# dance + ZJU, generation seed 42 first, then 123; gen48_ho8 and r15h35. One SR4D per GPU, two lanes.
# usage: queue_inner_nol4.sh <gpuA> <gpuB> ; run inside tmux. out outputs/sr4d_final/<k>_<split><sfx>_nol4, log logs/inner_<tag>.log
set -uo pipefail
cd "$(dirname "$0")/.."; source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
one() {  # gpu k split sfx
  local g=$1 K=$2 S=$3 SFX=$4; local TAG=${K}_${S}${SFX}_nol4; local O=outputs/sr4d_final/$TAG L=logs/inner_$TAG.log
  [ -f $O/metrics_renders.json ] && { echo "skip $TAG"; return; }
  echo $BASHPID > logs/inner_$TAG.pid
  CUDA_VISIBLE_DEVICES=$g PHASE=B_g48 METHOD=sr4d_final_nol4 NOTES="$TAG ($S), V0 + pose (a) + L2 + L3, L4 off" \
    scripts/run_sr4d.sh configs/${K}_r15h35$SFX.yaml $O configs/ablations/bones_v0_pose_a.yaml configs/ablations/no_l4_all.yaml -- data.setting=$S > $L 2>&1
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python tools/diag_jitter.py --ckpt $O/ckpt_final.pt --json $O/jitter.json >> $L 2>&1
  [ $S = r15h35 ] && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python -m sr4d.eval.covis_metrics --data_root $(python -c "import yaml;print(yaml.safe_load(open('$O/config.yaml'))['data']['root'])") --setting $S --renders $O/renders >> $L 2>&1
  echo "INNERDONE $TAG $(date +%H:%M)" | tee -a $L >> logs/queue_inner_nol4.log
}
( one $1 dance r15h35 ""; one $1 zju gen48_ho8 ""; one $1 dance gen48_ho8 _s123; one $1 zju r15h35 _s123 ) &
( one $2 zju r15h35 ""; one $2 dance r15h35 _s123; one $2 zju gen48_ho8 _s123 ) &
wait; echo "QUEUEDONE $(date +%H:%M)" >> logs/queue_inner_nol4.log
