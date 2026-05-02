"""Logged frame and safety session records (architecture §8.1.2)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import SensorBundle

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


class LoggedFrame(BaseModel):
    """One synchronized logged timestep."""

    model_config = {"extra": "forbid"}

    schema_version: str = Field(default=LOGGED_FRAME_SCHEMA_VERSION)
    frame_index: int
    timestamp_ns: int
    sensor_payload: SensorBundle
    scene_snapshot: SceneGraph
    commanded_action: RobotCommand | None = None
    executed_action: RobotCommand | None = None
    safety_decision: SafetyDecision | None = None
    skill_state: dict[str, Any] | None = Field(
        default=None,
        description="Legacy/audit field; None for substrate-driven operation.",
    )
    surgeon_input: TeleopInput | None = None
    outcome_label: dict[str, Any] | None = None
