"""D-029 check before training with the rule: in a TRAINED model, which free Gaussians are never inside any training
camera's image (deformed centre, all frames)? Remove them post hoc (opacity -> 0), re-render the held-out cams and
write <run>/diag_unseen/renders (+ stats.json) so the out-of-mask alpha map / metrics can be compared with the full model.
usage: PYTHONPATH=. python tools/diag_unseen.py --ckpt <run>/ckpt_final.pt
"""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from sr4d.data.dataset import MultiViewSequence
from sr4d.eval.render_cams import render_model
from sr4d.train.densify import unseen_free_mask
from sr4d.train.trainer import load_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    ck = Path(a.ckpt)
    st = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(st["cfg"])
    data = MultiViewSequence(cfg.data.root, cfg.data.setting, list(range(*cfg.data.frames)), load_images=False)
    m = load_model(ck, data, cfg).eval()
    m.use_l2, m.use_l3, m.use_l4 = m.l2 is not None, m.l3 is not None, m.l4 is not None
    m.body_adhere = False
    un = unseen_free_mask(m, data)
    op = torch.sigmoid(m.free["opacity"][:, 0])
    stats = dict(n_free=int(len(un)), n_unseen=int(un.sum()), frac=float(un.float().mean()),
                 unseen_opacity_sum=float(op[un].sum()), free_opacity_sum=float(op.sum()))
    print(json.dumps(stats))
    m.free.params["opacity"].data[un] = -20.0
    out = ck.parent / "diag_unseen"
    render_model(m, data, data.split["test_extrap_cams"], range(data.num_frames), out / "renders")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
