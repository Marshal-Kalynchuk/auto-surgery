"""Inverse Dynamics Model (IDM) prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IDMConfig:
    obs_dim: int
    act_dim: int
    hidden_dim: int = 256


def build_idm_mlp(cfg: IDMConfig) -> Any:
    """Create a tiny MLP IDM for Stage-0 sanity runs."""
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError("Install the `train` extra: uv sync --extra train") from e

    return nn.Sequential(
        nn.Linear(cfg.obs_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, cfg.act_dim),
    )

