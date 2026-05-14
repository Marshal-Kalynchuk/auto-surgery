"""Sensor bundle — typed per spec §5.4."""

from __future__ import annotations

from pydantic import BaseModel, Field

from auto_surgery.schemas.commands import Pose, Twist, Vec3


class CameraIntrinsics(BaseModel):
    """Immutable camera intrinsics for a captured frame."""

    model_config = {"extra": "forbid"}

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class ToolState(BaseModel):
    """Tool tip telemetry expressed in scene frame.

    The shaft ``MechanicalObject`` (Rigid3d) is integrated by SOFA in scene
    coordinates and is read back unchanged by the forceps observer, so both
    ``pose`` and ``twist`` are in scene frame. Cartesian actions use
    ``ControlMode.CARTESIAN_POSE`` with ``frame=ControlFrame.SCENE`` (tip target
    pose); the applier servoes the shaft toward that tip pose.
    """

    model_config = {"extra": "forbid"}

    pose: Pose
    twist: Twist
    jaw: float = Field(ge=0.0, le=1.0)
    wrench: Vec3
    in_contact: bool


class CameraView(BaseModel):
    """Appearance payload for a single camera."""

    model_config = {"extra": "forbid"}

    camera_id: str
    timestamp_ns: int
    extrinsics: Pose
    intrinsics: CameraIntrinsics
    frame_rgb: bytes | None = None


class SafetyStatus(BaseModel):
    """Current safety gate decision and cycle echo."""

    model_config = {"extra": "forbid"}

    motion_enabled: bool
    command_blocked: bool
    block_reason: str | None = None
    cycle_id_echo: int


class SensorBundle(BaseModel):
    """Synchronized sensor snapshot for a single tick."""

    model_config = {"extra": "forbid"}

    timestamp_ns: int
    sim_time_s: float
    tool: ToolState
    cameras: list[CameraView]
    safety: SafetyStatus


class JointState(BaseModel):
    """Off-path joint telemetry for bridges/debug."""

    model_config = {"extra": "forbid"}

    positions: dict[str, float]
    velocities: dict[str, float]
    torques: dict[str, float] | None = None


class Contact(BaseModel):
    """Single contact point between tool and tissue in camera frame."""

    model_config = {"extra": "forbid"}

    point: Vec3
    normal: Vec3
    force_magnitude: float
    body_id: str
    penetration_depth: float
