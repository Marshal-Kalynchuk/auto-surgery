"""Logged frame and safety session records (architecture §8.1.2)."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from auto_surgery.schemas.commands import Pose, RobotCommand, Twist, Vec3
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import Contact, SensorBundle

LOGGED_FRAME_SCHEMA_VERSION = "logged_frame_v1"


class SafetyDecision(BaseModel):
    """Outcome of sync command gate / safety evaluator stub."""

    model_config = {"extra": "forbid"}

    ok: bool = Field(..., description="True if command may proceed (sync gate pass).")
    gate_action: Literal["pass", "project", "veto"] = "pass"
    reason_codes: list[str] = Field(default_factory=list)


class TeleopInput(BaseModel):
    """Surgeon teleoperation sample."""

    model_config = {"extra": "forbid"}

    timestamp_ns: int
    raw_channels: dict[str, Any] = Field(default_factory=dict)


class ForcepsContractTrace(BaseModel):
    """Step-aligned forceps telemetry for command/state audit trails."""

    model_config = {"extra": "forbid"}

    frame_index: int
    timestamp_ns: int
    step_sim_index: int
    command_cycle_id: int
    command_enable: bool
    command_twist: Twist | None
    tool_pose: Pose
    tool_twist: Twist
    tool_wrench: Vec3
    tool_jaw: float
    tool_in_contact: bool
    contact_count: int
    contacts: list[Contact] = Field(default_factory=list)
    safety_blocked: bool
    safety_block_reason: str | None = None


class LoggedFrame(BaseModel):
    """One synchronized logged timestep."""

    model_config = {"extra": "forbid"}

    schema_version: str = Field(default=LOGGED_FRAME_SCHEMA_VERSION)
    frame_index: int
    timestamp_ns: int
    sensor_payload: SensorBundle
    scene_snapshot: SceneGraph | None = None
    commanded_action: RobotCommand | None = None
    executed_action: RobotCommand | None = None
    forceps_trace: ForcepsContractTrace | None = None
    safety_decision: SafetyDecision | None = None
    entity_state: dict[str, Any] | None = Field(
        default=None,
        description="Entity-centric state snapshot for downstream consumers.",
    )
    skill_state: dict[str, Any] | None = Field(
        default=None,
        description="Deprecated alias for entity_state in archived Stage-0 logs.",
    )
    surgeon_input: TeleopInput | None = None
    outcome_label: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_skill_state(cls, values: dict[str, Any]) -> dict[str, Any]:
        if (
            isinstance(values, dict)
            and values.get("entity_state") is None
            and values.get("skill_state") is not None
        ):
            values["entity_state"] = values["skill_state"]
        return values
