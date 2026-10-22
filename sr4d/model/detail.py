"""D-037: per-frame detail layer (stage s7). The shared SR4D model is frozen; every frame t gets its own small offsets
for every Gaussian: dx[t] (position, m) and dc[t] (colour, added to SH0 after /C0). Zero-initialised; regularised to 0
and to the neighbouring frames (train.loss.detail_mag / detail_smooth). Gaussian count is fixed during s7 (no densify)."""
import torch
import torch.nn as nn


class FrameDetail(nn.Module):
    def __init__(self, T: int, N: int):
        super().__init__()
        self.dx = nn.Parameter(torch.zeros(T, N, 3))
        self.dc = nn.Parameter(torch.zeros(T, N, 3))

    def reg(self, ts):
        mag = sum((self.dx[t] ** 2).sum(-1).mean() * 1e4 + (self.dc[t] ** 2).sum(-1).mean() for t in ts) / len(ts)  # dx in cm^2
        sm = 0.0
        for t in ts:
            for u in (t - 1, t + 1):
                if 0 <= u < self.dx.shape[0]:
                    sm = sm + ((self.dx[t] - self.dx[u]) ** 2).sum(-1).mean() * 1e4 + ((self.dc[t] - self.dc[u]) ** 2).sum(-1).mean()
        return mag, sm / len(ts)
