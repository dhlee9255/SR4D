"""Config loading: base.yaml <- dataset config <- ablation configs <- CLI dotlist overrides."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def load_config(cfg_paths: list[str], overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(ROOT / "configs" / "base.yaml")
    for p in cfg_paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    if cfg.data.root is None:
        paths = OmegaConf.load(ROOT / "configs" / "paths.yaml")
        cfg.data.root = str(Path(paths.storage.work_data) / cfg.data.dataset / str(cfg.data.seq))
    if not Path(cfg.data.root).is_absolute():
        cfg.data.root = str(ROOT / cfg.data.root)
    return cfg
