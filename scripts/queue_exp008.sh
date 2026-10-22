#!/usr/bin/env bash
# EXP-008 (D-043) transparent-background training, dance near (gen48_ho8) + far (r15h35), two lanes. usage: queue_exp008.sh <gpuA> <gpuB>
set -uo pipefail
cd "$(dirname "$0")/.."
D=data_dev/fouranyone_gen/dance5885633_gen48_2ring
DN=outputs/sr4d_final/dance_gen48_ho8_nol4/ckpt_final.pt; DF=outputs/sr4d_bones/dance_r15h35/v0_posea/ckpt_final.pt
A="--rand_bg --mask_w 0.05"; B="--rand_bg --mask_w 0.05 --max_ratio 10 --vh_prune"
ftt() { echo $BASHPID > logs/lf_ftgspp_transparent_$2.pid; CUDA_VISIBLE_DEVICES=$1 bash scripts/baselines/run_ftgspp_transparent.sh $D $2 outputs/ftgspp_transparent/dance_$2 > logs/lf_ftgspp_transparent_$2.log 2>&1; }
( bash scripts/run_layered_fade.sh $1 exp008a_dance_ho8 $D gen48_ho8 $DN 0 $A; bash scripts/run_layered_fade.sh $1 exp008b_dance_ho8 $D gen48_ho8 $DN 0 $B; ftt $1 gen48_ho8; echo "LANE A DONE $(date +%H:%M)" >> logs/queue_exp008.log ) &
( bash scripts/run_layered_fade.sh $2 exp008a_dance_r15h35 $D r15h35 $DF 0 $A; bash scripts/run_layered_fade.sh $2 exp008b_dance_r15h35 $D r15h35 $DF 0 $B; ftt $2 r15h35; echo "LANE B DONE $(date +%H:%M)" >> logs/queue_exp008.log ) &
wait; echo "QUEUEDONE $(date +%H:%M)" >> logs/queue_exp008.log
