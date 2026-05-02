from __future__ import annotations

import pytest

from auto_surgery.schemas.manifests import DataClassification, DatasetManifest
from auto_surgery.training.datasets import PhiDatasetBlockedError, assert_training_gate


def test_phi_blocked_by_default() -> None:
    manifest = DatasetManifest(
        dataset_id="d1",
        session_manifest_paths=[],
        data_classification=DataClassification.PHI_RESTRICTED,
    )
    with pytest.raises(PhiDatasetBlockedError):
        assert_training_gate(manifest)


def test_phi_allowed_when_explicit() -> None:
    manifest = DatasetManifest(
        dataset_id="d1",
        session_manifest_paths=[],
        data_classification=DataClassification.PHI_RESTRICTED,
    )
    assert_training_gate(manifest, phi_training_allowed=True)


def test_simulation_never_blocked() -> None:
    manifest = DatasetManifest(
        dataset_id="d1",
        session_manifest_paths=[],
        data_classification=DataClassification.SIMULATION,
    )
    assert_training_gate(manifest)
