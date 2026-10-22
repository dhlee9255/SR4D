"""Stage schedule (CLAUDE.md 7.5).

S1: L0 scaffold refinement (silhouette)          — run before Gaussians exist
S2: L1 (body surfels + free Gaussians) + free bones + L4 colour affine
S3: + L2 pose non-rigid
S4: + L3 instant residual (dropout p)
S5/S6: optional, not implemented yet (docs/DECISIONS.md D-010)
`train.joint=true` (ablation e): L2/L3 enabled from the start of S2, same total iterations.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stage:
    name: str
    start: int
    end: int


def stage_table(cfg) -> list[Stage]:
    st = cfg.train.stages
    out, it = [], 0
    for name in ("s2", "s3", "s4", "s7"):  # s7 = per-frame detail layer (D-037), only if configured
        n = int(st.get(name, 0) or 0)
        if n > 0:
            out.append(Stage(name, it, it + n))
            it += n
    return out


def stage_at(table: list[Stage], it: int) -> str:
    for s in table:
        if s.start <= it < s.end:
            return s.name
    return table[-1].name if table else "s2"


def layer_flags(cfg, stage: str) -> dict:
    joint = bool(cfg.train.get("joint", False))
    order = {"s2": 0, "s3": 1, "s4": 2, "s7": 3}[stage]
    return {
        "detail": stage == "s7",
        "l2": bool(cfg.model.l2.enabled) and (joint or order >= 1),
        "l3": bool(cfg.model.l3.enabled) and (joint or order >= 2),
        "l4": bool(cfg.model.l4.enabled),
    }
