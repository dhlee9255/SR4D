"""L2 as a 2D CNN on canonical front/back orthographic maps (D-035, Animatable Gaussians CVPR 2024 style).
Optional alternative to the MLP in pose_nonrigid.py, selected with model.l2.type = cnn (default stays mlp).

Map layout: every canonical Gaussian (body surfels AND free Gaussians, so hair / skirt are covered — SMPL-X UV would not
cover them) is orthographically projected along canonical z onto a pixel (u = x, v = -y) of a fixed canonical frame.
Per pixel the Gaussians are split into a FRONT and a BACK layer at the mid-depth between the pixel's nearest and farthest
Gaussian. Input = the Gaussians' current-pose LBS positions, expressed in the root (pelvis) frame, splatted (mean) into the
two layers + occupancy -> 8 channels. A small U-Net maps them to 2 x 13 channels (dx 3, dq 3 (+1), dlog_s 3, dcolor 3 + 1
spare); each Gaussian samples its own layer at its pixel (bilinear). Output dict and scaling are identical to the MLP.
forward(x, pose_feat, x_posed=None, G0=None): without x_posed the maps of the last full call are re-used (regularisers).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin=8, cout=26, ch=(32, 64, 128, 128)):
        super().__init__()
        self.down = nn.ModuleList()
        c = cin
        for o in ch:
            self.down.append(_blk(c, o)); c = o
        self.up = nn.ModuleList()
        for o in reversed(ch[:-1]):
            self.up.append(_blk(c + o, o)); c = o
        self.out = nn.Conv2d(c, cout, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = []
        for i, b in enumerate(self.down):
            x = b(x)
            if i < len(self.down) - 1:
                skips.append(x); x = F.avg_pool2d(x, 2)
        for b in self.up:
            s = skips.pop()
            x = b(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.out(x)


class PoseNonRigidCNN(nn.Module):
    def __init__(self, bbox_min, bbox_max, pose_in: int, cfg):
        super().__init__()
        bmin, bmax = torch.as_tensor(bbox_min, dtype=torch.float32), torch.as_tensor(bbox_max, dtype=torch.float32)
        self.register_buffer("bmin", bmin); self.register_buffer("bmax", bmax)
        H = int(cfg.get("cnn_res", 256))
        W = max(16, int(round(H * float(bmax[0] - bmin[0]) / float(bmax[1] - bmin[1]) / 16)) * 16)
        self.H, self.W = H, W
        self.net = UNet(8, 26, tuple(cfg.get("cnn_ch", [32, 64, 128, 128])))
        self._maps = None  # (1, 26, H, W) output of the last full call
        self._layer = None

    def _pix(self, x):
        u = (x[:, 0] - self.bmin[0]) / (self.bmax[0] - self.bmin[0])          # 0..1 left->right
        v = (self.bmax[1] - x[:, 1]) / (self.bmax[1] - self.bmin[1])          # 0..1 top->bottom
        return u.clamp(0, 1), v.clamp(0, 1)

    def _layers(self, x, iu, iv):
        """front (True) / back per Gaussian: split each pixel at the mid depth of its Gaussians."""
        idx = iv * self.W + iu
        n = self.H * self.W
        zmax = torch.full((n,), -1e9, device=x.device).scatter_reduce(0, idx, x[:, 2], "amax")
        zmin = torch.full((n,), 1e9, device=x.device).scatter_reduce(0, idx, x[:, 2], "amin")
        return x[:, 2] >= 0.5 * (zmax[idx] + zmin[idx])

    def forward(self, x: torch.Tensor, pose_feat: torch.Tensor, x_posed: torch.Tensor | None = None, G0: torch.Tensor | None = None):
        u, v = self._pix(x)
        if x_posed is not None:
            iu = (u * (self.W - 1)).round().long(); iv = (v * (self.H - 1)).round().long()
            front = self._layers(x.detach(), iu, iv)
            p = x_posed
            if G0 is not None:  # root-relative: remove global rotation / translation
                p = (p - G0[:3, 3]) @ G0[:3, :3]
            n = self.H * self.W
            idx = iv * self.W + iu + (~front).long() * n                      # 2 layers stacked
            s = torch.zeros(2 * n, 3, device=x.device).index_add(0, idx, p)
            c = torch.zeros(2 * n, device=x.device).index_add(0, idx, torch.ones_like(p[:, 0]))
            m = s / c.clamp_min(1)[:, None]
            occ = (c > 0).float()
            inp = torch.cat([m[:n].T.reshape(3, self.H, self.W), occ[:n].reshape(1, self.H, self.W),
                             m[n:].T.reshape(3, self.H, self.W), occ[n:].reshape(1, self.H, self.W)], 0)[None]
            self._maps = self.net(inp)
            self._layer = front
        assert self._maps is not None, "first call of the CNN L2 needs x_posed"
        g = torch.stack([u * 2 - 1, v * 2 - 1], -1)[None, :, None]             # (1,N,1,2)
        # regulariser calls (no x_posed) read the cached maps WITHOUT graph: their graph belongs to an earlier sample whose
        # backward may already have run; the CNN output is spatially smooth by construction, so the MLP's smoothness term
        # is a no-op here (D-035)
        maps = self._maps if x_posed is not None else self._maps.detach()
        of = F.grid_sample(maps[:, :13], g, mode="bilinear", align_corners=True)[0, :, :, 0].T
        ob = F.grid_sample(maps[:, 13:], g, mode="bilinear", align_corners=True)[0, :, :, 0].T
        front = self._layer if (self._layer is not None and len(self._layer) == len(x) and x_posed is not None) else torch.ones(len(x), dtype=torch.bool, device=x.device)
        if x_posed is None:  # regulariser call on a subset: pick the layer by mid-depth again
            iu = (u * (self.W - 1)).round().long(); iv = (v * (self.H - 1)).round().long()
            front = self._layers(x.detach(), iu, iv)
        o = torch.where(front[:, None], of, ob)
        dq = torch.cat([torch.ones_like(o[:, :1]), o[:, 3:6]], -1)
        return {"dx": o[:, 0:3] * 0.05, "dq": dq, "dlog_s": o[:, 6:9] * 0.1, "dcolor": o[:, 9:12] * 0.1}
