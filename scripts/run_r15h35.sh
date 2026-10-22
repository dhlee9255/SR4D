#!/usr/bin/env bash
# Item 4 held-out ring experiment. usage: [SFX=_s123] run_r15h35.sh <zju|dance> <sr4d_v0|sr4d_pa|sr4d_pb|sr4d_free|sr4d_pa_nol3|sr4d_pa_nol2dx|sr4d_pa_unseen|4dgs|ftgspp|dualgs> <gpu>
# SFX selects a re-generated data copy (item 5, generation seed variance): data <root>$SFX, configs/<k>_r15h35$SFX.yaml, outputs <k>_r15h35$SFX.
set -uo pipefail
cd "$(dirname "$0")/.."
K=$1; M=$2; SFX=${SFX:-}; export CUDA_VISIBLE_DEVICES=$3 PHASE=B_r15h35
case $K in zju) R=data_dev/zju_mocap_gen/387_gen48_2ring; CFG=configs/zju_r15h35.yaml ;; dance) R=data_dev/fouranyone_gen/dance5885633_gen48_2ring; CFG=configs/dance_r15h35.yaml ;;   dualgs) PHASE=B_r15h35 scripts/baselines/run_dualgs.sh $R r15h35 outputs/dualgs/${K}_r15h35$SFX > $L 2>&1 ;;
esac
R=$R$SFX; CFG=${CFG%.yaml}$SFX.yaml
L=logs/r15h35${SFX}_${K}_$M.log
case $M in
  sr4d_v0) METHOD=sr4d_v0 NOTES="$K r15h35$SFX held-out 35deg" scripts/run_sr4d.sh $CFG outputs/sr4d_bones/${K}_r15h35$SFX/v0 configs/ablations/bones_v0.yaml > $L 2>&1 ;;
  sr4d_pa|sr4d_pb) P=${M#sr4d_p}; METHOD=sr4d_v0_pose$P NOTES="$K r15h35$SFX held-out 35deg, D-025 pose ($P)" scripts/run_sr4d.sh $CFG outputs/sr4d_bones/${K}_r15h35$SFX/v0_pose$P configs/ablations/bones_v0_pose_$P.yaml > $L 2>&1
     scripts/post_bones.sh outputs/sr4d_bones/${K}_r15h35$SFX/v0_pose$P outputs/orbits/${K}_r15h35 pose$P$SFX >> $L 2>&1 ;;
  sr4d_pa_nol3) METHOD=sr4d_v0_posea_nol3 NOTES="$K r15h35$SFX held-out 35deg, pose (a), no L3 from scratch" scripts/run_sr4d.sh $CFG outputs/sr4d_bones/${K}_r15h35$SFX/v0_posea_nol3 configs/ablations/bones_v0_pose_a.yaml configs/ablations/no_l3.yaml > $L 2>&1
     scripts/post_bones.sh outputs/sr4d_bones/${K}_r15h35$SFX/v0_posea_nol3 outputs/orbits/${K}_r15h35 posea_nol3$SFX >> $L 2>&1 ;;
  sr4d_pa_nol2dx) METHOD=sr4d_v0_posea_nol2dx NOTES="$K r15h35$SFX held-out 35deg, pose (a), L2 without dx from scratch" scripts/run_sr4d.sh $CFG outputs/sr4d_bones/${K}_r15h35$SFX/v0_posea_nol2dx configs/ablations/bones_v0_pose_a.yaml configs/ablations/l2_no_dx.yaml > $L 2>&1
     scripts/post_bones.sh outputs/sr4d_bones/${K}_r15h35$SFX/v0_posea_nol2dx outputs/orbits/${K}_r15h35 posea_nol2dx$SFX >> $L 2>&1 ;;
  sr4d_pa_unseen) METHOD=sr4d_v0_posea_unseen NOTES="$K r15h35$SFX held-out 35deg, pose (a), prune unseen free Gaussians (D-029)" scripts/run_sr4d.sh $CFG outputs/sr4d_bones/${K}_r15h35$SFX/v0_posea_unseen configs/ablations/bones_v0_pose_a.yaml configs/ablations/prune_unseen.yaml > $L 2>&1
     scripts/post_bones.sh outputs/sr4d_bones/${K}_r15h35$SFX/v0_posea_unseen outputs/orbits/${K}_r15h35 posea_unseen$SFX >> $L 2>&1 ;;
  sr4d_free) METHOD=sr4d_free NOTES="$K r15h35$SFX held-out 35deg" scripts/run_sr4d.sh $CFG outputs/sr4d_bones/${K}_r15h35$SFX/free > $L 2>&1 ;;
  4dgs) NOTES="$K r15h35$SFX held-out 35deg" scripts/baselines/run_4dgs.sh $R r15h35 outputs/4dgs/${K}_r15h35$SFX > $L 2>&1 ;;
  ftgspp) BG=black INIT_FIX=0 METHOD=ftgspp_black_official NOTES="$K r15h35$SFX held-out 35deg, legacy" scripts/baselines/run_ftgspp.sh $R r15h35 outputs/ftgspp_black_official/${K}_r15h35$SFX > $L 2>&1 ;;
  dualgs) PHASE=B_r15h35 scripts/baselines/run_dualgs.sh $R r15h35 outputs/dualgs/${K}_r15h35$SFX > $L 2>&1 ;;
esac
