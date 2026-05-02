"""Helpers for resolving dataset entries over fsspec URIs."""

from __future__ import annotations

from urllib.parse import urlparse


def split_session_manifest_uri(session_manifest_uri: str) -> tuple[str, str, str]:
    """Derive storage root, case_id, session_id from a manifest URI.

    Expected suffix: ``.../cases/<case_id>/sessions/<session_id>/session_manifest.json``
    """

    parsed = urlparse(session_manifest_uri)
    path = parsed.path
    marker = "/cases/"
    if marker not in path:
        raise ValueError(f"Manifest URI must contain '/cases/': {session_manifest_uri}")
    root_path, rest = path.split(marker, 1)
    parts = rest.strip("/").split("/")
    if len(parts) != 4 or parts[1] != "sessions" or parts[3] != "session_manifest.json":
        raise ValueError(f"Unexpected manifest path layout: {path}")
    case_id = parts[0]
    session_id = parts[2]
    scheme = parsed.scheme or "file"
    netloc = parsed.netloc
    if scheme == "file":
        storage_root_uri = f"file://{root_path.rstrip('/')}/"
    else:
        storage_root_uri = f"{scheme}://{netloc}{root_path.rstrip('/')}/"
    return storage_root_uri, case_id, session_id
