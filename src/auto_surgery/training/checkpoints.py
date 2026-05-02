"""Atomic checkpoint writes with manifests."""

from __future__ import annotations

import io
from typing import Any

import fsspec

from auto_surgery.schemas.manifests import CheckpointManifest


def save_torch_checkpoint_atomic(
    checkpoint_uri: str,
    state: dict[str, Any],
    manifest: CheckpointManifest,
    *,
    fs_kwargs: dict[str, Any] | None = None,
) -> None:
    """Write ``torch.save`` payload atomically via rename-on-same-filesystem pattern."""

    try:
        import torch
    except ImportError as e:
        raise ImportError("Saving checkpoints requires torch (train extra).") from e
    fs, path = fsspec.url_to_fs(checkpoint_uri, **(fs_kwargs or {}))
    buf = io.BytesIO()
    torch.save(state, buf)
    data = buf.getvalue()
    partial = f"{path}.partial"
    fs.pipe_file(partial, data)
    if fs.exists(path):
        fs.rm(path)
    fs.mv(partial, path)
    man_path = f"{path}.manifest.json"
    fs.pipe_file(man_path, manifest.model_dump_json(indent=2).encode("utf-8"))


def load_torch_checkpoint(
    checkpoint_uri: str, *, fs_kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as e:
        raise ImportError("Loading checkpoints requires torch (train extra).") from e
    fs, path = fsspec.url_to_fs(checkpoint_uri, **(fs_kwargs or {}))
    with fs.open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)
