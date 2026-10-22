#!/usr/bin/env bash
# SR4D variants on the main protocol (D-030), final config + extra overrides / ablation yamls.
# usage: [SFX=_s123] run_g48_var.sh <zju|dance> <gen48_all|gen48_ho8|gen24_all> <variant_tag> <gpu> [ablation.yaml ...] [-- key=value ...]
set -uo pipefail
cd "$(dirname "$0")/.."
K=$1; S=$2; V=$3; G=$4; shift 4; SFX=${SFX:-}; export CUDA_VISIBLE_DEVICES=$G PHASE=B_g48
case $S in
  gen24_all) case $K in zju) CFG=configs/zju_gen_dev.yaml ;; dance) CFG=configs/dance_gen_dev.yaml ;; esac; SET=gen24 ;;
  *) CFG=configs/${K}_r15h35$SFX.yaml; SET=$S ;;
esac
ABL=(); OVR=()
while [[ $# -gt 0 ]]; do if [[ $1 == "--" ]]; then shift; OVR=("$@"); break; fi; ABL+=("$1"); shift; done
TAG=${K}_${S}${SFX}; OUT=outputs/sr4d_final/${TAG}_$V; L=logs/g48var_${TAG}_$V.log
METHOD=sr4d_final_$V NOTES="$TAG ($SET), final config + $V" scripts/run_sr4d.sh $CFG $OUT configs/ablations/bones_v0_pose_a.yaml ${ABL[@]+"${ABL[@]}"} -- data.setting=$SET ${OVR[@]+"${OVR[@]}"} > $L 2>&1
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
python tools/diag_jitter.py --ckpt $OUT/ckpt_final.pt --json $OUT/jitter.json >> $L 2>&1
echo "done $TAG $V rc=$? $(date +%H:%M)"
