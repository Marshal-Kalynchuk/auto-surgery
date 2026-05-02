"""Bootstrap substrate MLP for imitation sanity runs."""

from __future__ import annotations

from typing import Any


def build_tiny_substrate_mlp(obs_dim: int, action_dim: int) -> Any:
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError("Install the `train` extra: uv sync --extra train") from e
    return nn.Sequential(
        nn.Linear(obs_dim, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim),
    )
