"""Session, dataset, and checkpoint manifests (provenance + governance)."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DataClassification(StrEnum):
    """Storage handling tier for governance (plan §9)."""

    SYNTHETIC = "synthetic"
    SIMULATION = "simulation"
    LAB_NON_PHI = "lab_non_phi"
    PHI_RESTRICTED = "phi_restricted"
    DEIDENTIFIED_DERIVED = "deidentified_derived"


class RetentionTier(StrEnum):
    """Retention policy label — actual durations are operational."""

    RAW_SENSORS = "raw_sensors"
    DERIVED_FEATURES = "derived_features"
    CURATED_TRAINING = "curated_training"
    CHECKPOINTS = "checkpoints"
    AUDIT_ONLY = "audit_only"


SESSION_MANIFEST_SCHEMA_VERSION = "session_manifest_v1"
DATASET_MANIFEST_SCHEMA_VERSION = "dataset_manifest_v1"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "checkpoint_manifest_v1"


class EnvConfig(BaseModel):
    """Simulator / environment reset configuration."""

    model_config = {"extra": "forbid"}

    scenario_id: str = "default"
    seed: int = 0
    domain_randomization: dict[str, Any] = Field(default_factory=dict)


class SessionManifest(BaseModel):
    """Written once per capture session (plan §3 data pipeline)."""

    model_config = {"extra": "forbid"}

    schema_version: str = Field(default=SESSION_MANIFEST_SCHEMA_VERSION)
    session_id: str
    case_id: str
    capture_rig_id: str
    clock_source: str
    software_git_sha: str
    logged_frame_schema_version: str
    sensor_list: list[str] = Field(default_factory=list)
    data_classification: DataClassification
    retention_tier: RetentionTier = RetentionTier.RAW_SENSORS
    deidentification_method: str | None = Field(
        default=None,
        description="Required for DEIDENTIFIED_DERIVED; N/A for synthetic/sim.",
    )
    access_control_decision_ref: str | None = Field(
        default=None,
        description="Link to ticket/policy id for PHI exports.",
    )
    partition_checksums: dict[str, str] = Field(
        default_factory=dict,
        description="Relative path -> sha256.",
    )
    notes: str | None = None


class DatasetManifest(BaseModel):
    """Training/eval dataset slice — consumed only via manifest paths."""

    model_config = {"extra": "forbid"}

    schema_version: str = Field(default=DATASET_MANIFEST_SCHEMA_VERSION)
    dataset_id: str
    session_manifest_paths: list[str] = Field(
        ...,
        description="fsspec URIs to session manifests or session roots.",
    )
    frame_filters: dict[str, Any] = Field(default_factory=dict)
    data_classification: DataClassification
    retention_tier: RetentionTier = RetentionTier.CURATED_TRAINING


class CheckpointManifest(BaseModel):
    """Checkpoint provenance for reproducibility."""

    model_config = {"extra": "forbid"}

    schema_version: str = Field(default=CHECKPOINT_MANIFEST_SCHEMA_VERSION)
    checkpoint_id: str
    base_model_hash: str | None = None
    adapter_hash: str | None = None
    dataset_manifest_path: str
    training_config_path: str
    git_sha: str
    mlflow_run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
