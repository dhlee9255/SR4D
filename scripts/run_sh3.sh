#!/usr/bin/env bash
# Item 4 of user 2026-10-12: SH degree experiment (max SH 2 -> 3; SH degree is shared by body surfels and free Gaussians
# in the single 3DGS pass, D-004). Final SR4D config (V0 + pose (a), L2/L3 on). usage: [SFX=_s123] run_sh3.sh <zju|dance> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
K=$1; SFX=${SFX:-}; export CUDA_VISIBLE_DEVICES=$2 PHASE=B_r15h35
CFG=configs/${K}_r15h35$SFX.yaml; OUT=outputs/sr4d_bones/${K}_r15h35$SFX/v0_posea_sh3; L=logs/r15h35${SFX}_${K}_sh3.log
METHOD=sr4d_v0_posea_sh3 NOTES="$K r15h35$SFX held-out 35deg, pose (a), max SH 3" scripts/run_sr4d.sh $CFG $OUT configs/ablations/bones_v0_pose_a.yaml -- model.body_surfels.max_sh_degree=3 > $L 2>&1
scripts/post_bones.sh $OUT outputs/orbits/${K}_r15h35 posea_sh3$SFX >> $L 2>&1
echo "done sh3 $K$SFX"
