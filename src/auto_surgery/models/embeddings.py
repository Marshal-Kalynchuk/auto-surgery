"""Bootstrap joint embedding MLP (until a VLP/VLA backbone is wired)."""

from __future__ import annotations

from typing import Any


def build_tiny_embedding_mlp(in_dim: int, out_dim: int) -> Any:
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError("Install the `train` extra: uv sync --extra train") from e
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )
