"""Tests for forceps applier magnitude and accel limiting (pose-servo path)."""

from __future__ import annotations

import pytest

from auto_surgery.env.sofa_tools import build_forceps_velocity_applier
from auto_surgery.schemas.commands import ControlFrame, ControlMode, Pose, Quaternion, RobotCommand, Vec3
from auto_surgery.schemas.motion import MotionShaping


class _Data:
    """Minimal SOFA-like container exposing a mutable value field."""

    def __init__(self, value: list[float] | tuple[float, ...]) -> None:
        self.value = list(value)


class _ForcepsDof:
    def __init__(self, initial: list[float] | tuple[float, ...]) -> None:
        self.position = _Data(list(initial))
        self.v = _Data([0.0, 0.0, 0.0])
        self.w = _Data([0.0, 0.0, 0.0])
        self.force = _Data([0.0, 0.0, 0.0])

    def getObject(self, name: str) -> _Data | None:
        if name == "v":
            return self.v
        if name == "w":
            return self.w
        if name == "force":
            return self.force
        if name == "position":
            return self.position
        return None


def _pose_mm(*, x: float, y: float = 0.0, z: float = 0.0) -> Pose:
    return Pose(
        position=Vec3(x=x, y=y, z=z),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )


def _scene_pose_cmd(
    *,
    timestamp_ns: int,
    cycle_id: int,
    tip: Pose,
    shaping: MotionShaping,
) -> RobotCommand:
    return RobotCommand(
        timestamp_ns=timestamp_ns,
        cycle_id=cycle_id,
        control_mode=ControlMode.CARTESIAN_POSE,
        frame=ControlFrame.SCENE,
        cartesian_pose_target=tip,
        enable=True,
        source="test",
        motion_shaping=shaping,
        motion_shaping_enabled=True,
    )


def test_forceps_velocity_applier_limits_linear_and_angular_accel() -> None:
    shaping = MotionShaping(
        max_linear_mm_s=2.0,
        max_angular_rad_s=2.0,
        max_linear_accel_mm_s2=0.5,
        max_angular_accel_rad_s2=0.2,
        bias_gain_max=0.0,
        bias_ramp_distance_mm=1.0,
        orientation_bias_gain=0.0,
        orientation_deadband_rad=0.0,
    )
    shaping_relaxed = shaping.model_copy(update={"max_linear_accel_mm_s2": 100.0})
    dof = _ForcepsDof((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    applier = build_forceps_velocity_applier(
        forceps_dof=dof,
        motion_shaping=shaping,
        tool_tip_offset_local=(0.0, 0.0, 0.0),
    )
    dof_key = str(id(dof))

    applier(None, _scene_pose_cmd(timestamp_ns=0, cycle_id=0, tip=_pose_mm(x=0.0), shaping=shaping))
    applier(None, _scene_pose_cmd(timestamp_ns=1_000_000_000, cycle_id=1, tip=_pose_mm(x=0.0), shaping=shaping))

    applier(
        None,
        _scene_pose_cmd(timestamp_ns=2_000_000_000, cycle_id=2, tip=_pose_mm(x=2.0), shaping=shaping_relaxed),
    )
    assert dof.v.value == pytest.approx([2.0, 0.0, 0.0])
    assert dof.w.value == pytest.approx([0.0, 0.0, 0.0])
    first_feedback = applier.safety_feedback[dof_key]
    assert first_feedback["clamped_linear"] is False
    assert first_feedback["clamped_angular"] is False
    assert first_feedback["scaled_linear"] is None
    assert first_feedback["scaled_angular"] is None

    applier(
        None,
        _scene_pose_cmd(timestamp_ns=3_000_000_000, cycle_id=3, tip=_pose_mm(x=0.0), shaping=shaping),
    )
    assert dof.v.value == pytest.approx([1.5, 0.0, 0.0])
    assert dof.w.value == pytest.approx([0.0, 0.0, 0.0])
    second_feedback = applier.safety_feedback[dof_key]
    assert second_feedback["clamped_linear"] is True
    assert second_feedback["clamped_angular"] is False
    assert second_feedback["scaled_linear"] is not None
    assert float(second_feedback["scaled_linear"]) < 1.0
    assert second_feedback["scaled_angular"] is None
