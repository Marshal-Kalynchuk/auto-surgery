"""Minimal LoRA adapter promotion registry (architecture §8.2)."""

from __future__ import annotations

import hashlib
from typing import Any

import fsspec
from pydantic import BaseModel

from auto_surgery.logging import storage


class AdapterRecord(BaseModel):
    model_config = {"extra": "forbid"}

    adapter_hash: str
    base_model_hash: str
    eval_score: float | None = None
    promoted: bool = False
    notes: str | None = None


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LoRARegistry:
    """Append-only JSONL registry of adapter candidates and promotion decisions."""

    def __init__(self, storage_root: str, fs_kwargs: dict[str, Any] | None = None) -> None:
        self._fs, self._root = fsspec.url_to_fs(storage_root, **(fs_kwargs or {}))
        self._path = f"{self._root}/{storage.adapter_registry_path()}"
        self._fs.makedirs(self._path.rsplit("/", 1)[0], exist_ok=True)

    def append(self, record: AdapterRecord) -> None:
        line = record.model_dump_json() + "\n"
        with self._fs.open(self._path, "ab") as f:
            f.write(line.encode("utf-8"))

    def promote_if_eval_ok(self, record: AdapterRecord, *, min_score: float) -> AdapterRecord:
        promoted = record.eval_score is not None and record.eval_score >= min_score
        updated = record.model_copy(update={"promoted": promoted})
        self.append(updated)
        return updated
