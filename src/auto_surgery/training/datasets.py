"""Dataset iterators — manifest-driven only."""

from __future__ import annotations

from collections.abc import Iterator

from auto_surgery.logging.writer import (
    count_segments,
    load_segment_frames,
    load_session_manifest,
)
from auto_surgery.schemas.logging import LoggedFrame
from auto_surgery.schemas.manifests import DataClassification, DatasetManifest
from auto_surgery.training.paths import split_session_manifest_uri


class PhiDatasetBlockedError(RuntimeError):
    """Training on PHI without explicit gate clearance."""


def assert_training_gate(manifest: DatasetManifest, *, phi_training_allowed: bool = False) -> None:
    """Block accidental PHI training in CI/dev unless explicitly allowed."""

    if (
        manifest.data_classification == DataClassification.PHI_RESTRICTED
        and not phi_training_allowed
    ):
        raise PhiDatasetBlockedError(
            "Dataset is PHI_RESTRICTED: set phi_training_allowed=True only after "
            "RBAC/encryption/audit controls from docs/DATA_GOVERNANCE.md are satisfied."
        )


def iter_logged_frames(
    manifest: DatasetManifest,
    *,
    phi_training_allowed: bool = False,
    fs_kwargs: dict | None = None,
) -> Iterator[LoggedFrame]:
    """Yield all frames for all sessions referenced by the dataset manifest."""

    assert_training_gate(manifest, phi_training_allowed=phi_training_allowed)
    fs_kw = fs_kwargs or {}
    for uri in manifest.session_manifest_paths:
        storage_root, case_id, session_id = split_session_manifest_uri(uri)
        _ = load_session_manifest(storage_root, case_id, session_id, **fs_kw)
        n_seg = count_segments(storage_root, case_id, session_id, **fs_kw)
        for seg_idx in range(n_seg):
            frames = load_segment_frames(storage_root, case_id, session_id, seg_idx, **fs_kw)
            yield from frames


def frame_count_estimate(
    manifest: DatasetManifest,
    *,
    phi_training_allowed: bool = False,
    fs_kwargs: dict | None = None,
) -> int:
    """Count frames by scanning segments (small datasets / tests only)."""

    return sum(
        1
        for _ in iter_logged_frames(
            manifest,
            phi_training_allowed=phi_training_allowed,
            fs_kwargs=fs_kwargs,
        )
    )
