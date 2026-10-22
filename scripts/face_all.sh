#!/usr/bin/env bash
# Face-region metrics (D-033) for main-protocol runs, CPU only (GPUs are training). Skips runs already scored.
cd "$(dirname "$0")/.."; source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=. OMP_NUM_THREADS=8
for D in outputs/*/{zju,dance}_gen48_ho8* outputs/*/{zju,dance}_gen48_all outputs/*/{zju,dance}_gen24_all; do
  [ -f $D/metrics_renders.json ] || continue; [ -f $D/metrics_face_renders.json ] && continue
  case $D in *zju_gen24*) R=data_dev/zju_mocap_gen/387_gen24_genonly; S=gen24 ;; *dance_gen24*) R=data_dev/fouranyone_gen/dance5885633_gen24; S=gen24 ;;
    *zju_*) R=data_dev/zju_mocap_gen/387_gen48_2ring ;; *dance_*) R=data_dev/fouranyone_gen/dance5885633_gen48_2ring ;; esac
  case $D in *gen48_ho8*) S=gen48_ho8 ;; *gen48_all*) S=gen48_all ;; esac
  echo "== $D"; python tools/face_metrics.py --data_root $R --setting $S --renders $D/renders 2>&1 | grep -v -i warn | tail -1
done
