"""Append-only case/session catalog (operational artifact, architecture §8.1.3)."""

from __future__ import annotations

import json
from typing import Any

import fsspec

from auto_surgery.logging import storage
from auto_surgery.schemas.manifests import SessionManifest


class CaseCatalog:
    """JSONL catalog of finalized sessions for audit and training curation."""

    def __init__(self, storage_root: str, fs_kwargs: dict[str, Any] | None = None) -> None:
        self._fs, self._root = fsspec.url_to_fs(storage_root, **(fs_kwargs or {}))
        self._catalog_path = f"{self._root}/{storage.catalog_sessions_path()}"
        self._fs.makedirs(self._catalog_path.rsplit("/", 1)[0], exist_ok=True)

    def append(self, manifest: SessionManifest, *, manifest_relative_path: str) -> None:
        entry = {
            "case_id": manifest.case_id,
            "session_id": manifest.session_id,
            "manifest_relative_path": manifest_relative_path,
            "data_classification": manifest.data_classification.value,
            "retention_tier": manifest.retention_tier.value,
            "software_git_sha": manifest.software_git_sha,
        }
        line = json.dumps(entry, sort_keys=True) + "\n"
        with self._fs.open(self._catalog_path, "ab") as f:
            f.write(line.encode("utf-8"))
