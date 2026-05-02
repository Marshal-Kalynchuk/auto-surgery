"""Robot command types (action space boundary)."""

from typing import Any

from pydantic import BaseModel, Field


class RobotCommand(BaseModel):
    """Command issued toward the robot after safety gating (architecture §8.3.2)."""

    model_config = {"extra": "forbid"}

    timestamp_ns: int = Field(..., description="Monotonic or synced wall time in nanoseconds.")
    representation: str = Field(default="mixed", description="joint | cartesian | mixed")
    joint_positions: dict[str, float] | None = None
    cartesian_pose: dict[str, Any] | None = None
    gripper: float | None = Field(default=None, ge=0.0, le=1.0)
    mode_flags: dict[str, Any] = Field(default_factory=dict)
