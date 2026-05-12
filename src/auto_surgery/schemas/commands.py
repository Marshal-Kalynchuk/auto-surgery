"""Robot command types (action space boundary)."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Vec3(BaseModel):
    """Right-handed 3D vector (meters / meters-per-second)."""

    model_config = {"extra": "forbid"}

    x: float
    y: float
    z: float


class Quaternion(BaseModel):
    """Rotation quaternion in Hamilton convention."""

    model_config = {"extra": "forbid"}

    w: float
    x: float
    y: float
    z: float

    @model_validator(mode="after")
    def _ensure_unit_norm(self) -> Quaternion:
        norm = math.sqrt(
            self.w ** 2
            + self.x ** 2
            + self.y ** 2
            + self.z ** 2
        )
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("Quaternion must be normalized within 1e-6.")
        return self


class Pose(BaseModel):
    """Rigid pose in a declared frame."""

    model_config = {"extra": "forbid"}

    position: Vec3
    rotation: Quaternion


class Twist(BaseModel):
    """Instantaneous velocity: linear (m/s) + angular (rad/s axis-rate)."""

    model_config = {"extra": "forbid"}

    linear: Vec3
    angular: Vec3


class ControlFrame(StrEnum):
    CAMERA = "camera"
    TOOL_TIP = "tool_tip"
    SCENE = "scene"
    ROBOT_BASE = "robot_base"


class ControlMode(StrEnum):
    CARTESIAN_TWIST = "cartesian_twist"
    CARTESIAN_POSE = "cartesian_pose"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_POSITION = "joint_position"


_MODE_PAYLOAD_FIELD = {
    ControlMode.CARTESIAN_TWIST: "cartesian_twist",
    ControlMode.CARTESIAN_POSE: "cartesian_pose_target",
    ControlMode.JOINT_VELOCITY: "joint_velocities",
    ControlMode.JOINT_POSITION: "joint_positions",
}


class RobotCommand(BaseModel):
    """Command issued toward the robot after safety gating."""

    model_config = {"extra": "forbid"}

    timestamp_ns: int
    cycle_id: int = Field(..., ge=0)
    control_mode: ControlMode = ControlMode.CARTESIAN_TWIST

    cartesian_twist: Twist | None = None
    cartesian_pose_target: Pose | None = None
    joint_velocities: dict[str, float] | None = None
    joint_positions: dict[str, float] | None = None

    frame: ControlFrame = ControlFrame.CAMERA

    tool_jaw_target: float | None = Field(default=None, ge=0.0, le=1.0)

    enable: bool = False
    source: str = Field(
        default="sim",
        description="Origin of the command: sim | policy | tele_op | scripted.",
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> RobotCommand:
        target_field = _MODE_PAYLOAD_FIELD[self.control_mode]
        target_value = getattr(self, target_field)
        if target_value is None:
            raise ValueError(
                f"{self.control_mode.value} requires populating {target_field}."
            )
        for field_name in _MODE_PAYLOAD_FIELD.values():
            if field_name == target_field:
                continue
            if getattr(self, field_name) is not None:
                raise ValueError(
                    "Only the payload matching control_mode may be populated."
                )
        return self
