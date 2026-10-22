#!/usr/bin/env bash
# Generation-seed variance (items 5/6): generate the 2-ring (15/35) set with another 4DAnyone seed, convert it,
# then train the given r15h35 methods on it in order. usage: chain_seed.sh <zju|dance> <gpu> <seed> <method...>
set -uo pipefail
cd "$(dirname "$0")/.."
K=$1; G=$2; SEED=$3; shift 3
FA=/data/AI-Graphics/Applications/4Danyone
case $K in
  zju) VID=$FA/gt_compare/zju_387_camB5_121f.mp4; ORIG=$FA/runs/zju_24view_run1; CLIP=zju_387_camB5_121f; D=data_dev/zju_mocap_gen/387_gen48_2ring; FPS=50 ;;
  dance) VID=$FA/4DAnyone/data/source/pexels/5885633-hd_1080_1920_25fps.mp4; ORIG=$FA/runs/dance5885633_24view_run1; CLIP=5885633-hd_1080_1920_25fps; D=data_dev/fouranyone_gen/dance5885633_gen48_2ring; FPS=25 ;;
esac
RUN=outputs/fouranyone_runs/${K}_2ring_s$SEED
[ -d ${D}_s$SEED/images ] || {
  bash scripts/gen_4danyone_2ring.sh $VID $ORIG $RUN $G $SEED > logs/gen2ring_${K}_s$SEED.log 2>&1 || { echo gen failed; exit 1; }
  bash scripts/prep_2ring.sh $RUN $CLIP ${D}_s$SEED $FPS $G > logs/prep2ring_${K}_s$SEED.log 2>&1 || { echo prep failed; exit 1; }
}
sed "s|_gen48_2ring\$|_gen48_2ring_s$SEED|; s|^# Item 4:|# Generation seed $SEED copy of the item-4 data:|" configs/${K}_r15h35.yaml > configs/${K}_r15h35_s$SEED.yaml
for M in "$@"; do SFX=_s$SEED bash scripts/run_r15h35.sh $K $M $G; echo "done $M $(date)"; done
echo chain_seed done $K $SEED
