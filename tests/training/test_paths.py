from __future__ import annotations

from auto_surgery.training.paths import split_session_manifest_uri


def test_split_session_manifest_uri_file_scheme() -> None:
    uri = "file:///data/store/cases/c1/sessions/s1/session_manifest.json"
    root, case_id, sid = split_session_manifest_uri(uri)
    assert case_id == "c1"
    assert sid == "s1"
    assert root.startswith("file:///data/store/")
