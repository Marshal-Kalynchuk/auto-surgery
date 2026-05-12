"""fsspec-backed path helpers for session layout."""

from __future__ import annotations


def session_prefix(case_id: str, session_id: str) -> str:
    return f"cases/{case_id}/sessions/{session_id}"


def segments_dir(case_id: str, session_id: str) -> str:
    return f"{session_prefix(case_id, session_id)}/segments"


def blobs_dir(case_id: str, session_id: str) -> str:
    return f"{session_prefix(case_id, session_id)}/blobs"


def session_manifest_path(case_id: str, session_id: str) -> str:
    return f"{session_prefix(case_id, session_id)}/session_manifest.json"


def run_metadata_path(case_id: str, session_id: str) -> str:
    return f"{session_prefix(case_id, session_id)}/run_metadata.json"


def segment_path(case_id: str, session_id: str, segment_index: int) -> str:
    return f"{segments_dir(case_id, session_id)}/segment_{segment_index:05d}.parquet"


def catalog_sessions_path() -> str:
    """Relative path under storage root for session catalog JSONL."""
    return "catalog/sessions.jsonl"


def dataset_registry_path() -> str:
    return "catalog/datasets.jsonl"


def adapter_registry_path() -> str:
    return "catalog/adapters.jsonl"
