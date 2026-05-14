"""Session, dataset, and checkpoint manifests (provenance + governance)."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from auto_surgery.schemas.scene import SceneConfig

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
RUN_METADATA_SCHEMA_VERSION = "run_metadata_v1"
_DEFAULT_SCENE_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "scenes" / "dejavu_brain.yaml"


def _default_scene_config() -> SceneConfig:
    from auto_surgery.config import load_scene_config

    return load_scene_config(_DEFAULT_SCENE_CONFIG_PATH)


class EnvConfig(BaseModel):
    """Simulator / environment reset configuration."""

    model_config = {"extra": "forbid"}

    scenario_id: str = "default"
    seed: int = 0
    scene: SceneConfig = Field(default_factory=_default_scene_config)
    control_rate_hz: float = Field(
        default=250.0,
        ge=1.0,
        description="Duration of each control tick (controls dt = 1/control_rate_hz).",
    )
    frame_rate_hz: float = Field(
        default=30.0,
        ge=1.0,
        description="Rendering cadence; used to compute capture ticks.",
    )
    episode_max_ticks: int | None = Field(
        default=None,
        description="Caller hint for max ticks; not enforced by env.",
    )


class RunMetadata(BaseModel):
    """Per-rollout provenance snapshot (written next to the session manifest)."""

    model_config = {"extra": "forbid"}

    schema_version: str = Field(default=RUN_METADATA_SCHEMA_VERSION)
    software_git_sha: str
    steps_requested: int
    fallback_to_stub: bool
    run_id: str | None = None
    sofa_scene_path: str | None = None
    sofa_scene_id: str | None = None
    sofa_tool_id: str | None = None
    action_generator_config: dict[str, Any] = Field(default_factory=dict)
    capture_modalities: list[str] = Field(default_factory=list)
    run_manifest_path: str | None = None
    session_manifest_path: str | None = None
    trace_artifact_paths: dict[str, str] = Field(default_factory=dict)


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
    artifact_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Session-local artifact paths keyed by purpose (e.g. forceps trace parquet).",
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
