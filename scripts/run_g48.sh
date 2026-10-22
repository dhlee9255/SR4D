#!/usr/bin/env bash
# Main protocol after the user's 2026-10-12 change (r15h35 frozen as appendix, D-030).
#   gen48_all : 48 generated views (15+35 rings) all train -> orbits / visual comparison / viewer
#   gen48_ho8 : 40 train / 8 held-out generated views -> fg-interior PSNR, LPIPS, tOF, out-of-mask alpha (co-visible aux)
#   gen24_all : original 24-view generation, all train -> GT-free metrics + visual
# SR4D = final config (bones V0, pose (a), L2 + L3). FTGS++ = legacy black bg as in B-stage. Generated views only.
# usage: [SFX=_s123] run_g48.sh <zju|dance> <gen48_all|gen48_ho8|gen24_all> <sr4d|4dgs|ftgspp|dualgs> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
K=$1; S=$2; M=$3; SFX=${SFX:-}; export CUDA_VISIBLE_DEVICES=$4 PHASE=B_g48
case $S in
  gen24_all) [ -n "$SFX" ] && { echo "gen24_all has one generation only"; exit 1; }
    case $K in zju) R=data_dev/zju_mocap_gen/387_gen24_genonly; CFG=configs/zju_gen_dev.yaml ;; dance) R=data_dev/fouranyone_gen/dance5885633_gen24; CFG=configs/dance_gen_dev.yaml ;; esac
    SET=gen24 ;;
  *) case $K in zju) R=data_dev/zju_mocap_gen/387_gen48_2ring$SFX ;; dance) R=data_dev/fouranyone_gen/dance5885633_gen48_2ring$SFX ;; esac
    CFG=configs/${K}_r15h35$SFX.yaml; SET=$S ;;
esac
TAG=${K}_${S}${SFX}; L=logs/g48_${TAG}_$M.log; N="$TAG ($SET)"
case $M in
  sr4d) METHOD=sr4d_final NOTES="$N, V0 + pose (a) + L2 + L3" scripts/run_sr4d.sh $CFG outputs/sr4d_final/$TAG configs/ablations/bones_v0_pose_a.yaml -- data.setting=$SET > $L 2>&1
        source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
        CUDA_VISIBLE_DEVICES=$4 python tools/diag_jitter.py --ckpt outputs/sr4d_final/$TAG/ckpt_final.pt --json outputs/sr4d_final/$TAG/jitter.json >> $L 2>&1 ;;
  4dgs) NOTES="$N" scripts/baselines/run_4dgs.sh $R $SET outputs/4dgs/$TAG > $L 2>&1 ;;
  ftgspp) BG=black INIT_FIX=0 METHOD=ftgspp_black_official NOTES="$N, legacy" scripts/baselines/run_ftgspp.sh $R $SET outputs/ftgspp_black_official/$TAG > $L 2>&1 ;;
  dualgs) NOTES="$N" scripts/baselines/run_dualgs.sh $R $SET outputs/dualgs/$TAG > $L 2>&1
          if [ ! -f outputs/dualgs/$TAG/metrics_renders.json ]; then  # async CUDA failure seen on zju s7 (D-030): retry blocking
            mv $L $L.fail; rm -rf "outputs/dualgs/${TAG:?}"
            NOTES="$N, CUDA_LAUNCH_BLOCKING=1 retry" CUDA_LAUNCH_BLOCKING=1 scripts/baselines/run_dualgs.sh $R $SET outputs/dualgs/$TAG > $L 2>&1
          fi ;;
esac
echo "done $TAG $M rc=$? $(date +%H:%M)"
