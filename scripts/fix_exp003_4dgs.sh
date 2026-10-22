#!/usr/bin/env bash
# regenerate outputs/4dgs/dance_r15h35/data/ims (deleted for disk) via a temp export, then render the 4DGS EXP-003 orbits.
set -uo pipefail
SR4D=$(cd "$(dirname "$0")/.." && pwd); cd "$SR4D"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate sr4d
T=$SR4D/outputs/_tmp_4dgs_r15h35
cd tools && python export_4dgs.py --root $SR4D/data_dev/fouranyone_gen/dance5885633_gen48_2ring --setting r15h35 --out $T && cd $SR4D
cmp <(python -c "import json;print(json.load(open('$T/data/train_meta.json'))['fn'][:3])") <(python -c "import json;print(json.load(open('outputs/4dgs/dance_r15h35/data/train_meta.json'))['fn'][:3])") || echo "WARN meta differs"
mv $T/data/ims outputs/4dgs/dance_r15h35/data/ims && rm -rf $T
scripts/render_exp003_orbits.sh 2
rm -rf outputs/4dgs/dance_r15h35/data/ims
echo "FIXDONE"
