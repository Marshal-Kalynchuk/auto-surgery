"""Tests for forceps applier magnitude and accel limiting."""

from __future__ import annotations

import pytest

from auto_surgery.env.sofa_tools import build_forceps_velocity_applier
from auto_surgery.schemas.commands import ControlMode, RobotCommand, Twist, Vec3
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


def test_forceps_velocity_applier_limits_linear_and_angular_accel() -> None:
    shaping = MotionShaping(
        max_linear_m_s=2.0,
        max_angular_rad_s=2.0,
        max_linear_accel_m_s2=0.5,
        max_angular_accel_rad_s2=0.2,
        bias_gain_max=0.0,
        bias_ramp_distance_m=1.0,
        orientation_bias_gain=0.0,
        orientation_deadband_rad=0.0,
    )
    dof = _ForcepsDof((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    applier = build_forceps_velocity_applier(
        forceps_dof=dof,
        force_scale=1.0,
        motion_shaping=shaping,
    )
    dof_key = str(id(dof))

    first_cmd = RobotCommand(
        timestamp_ns=0,
        cycle_id=0,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=1.0, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=1.0),
        ),
        enable=True,
        source="test",
    )
    applier(None, first_cmd)
    assert dof.v.value == pytest.approx([1.0, 0.0, 0.0])
    assert dof.w.value == pytest.approx([0.0, 0.0, 1.0])
    first_feedback = applier.safety_feedback[dof_key]
    assert first_feedback == {
        "clamped_linear": False,
        "clamped_angular": False,
        "scaled_linear": None,
        "scaled_angular": None,
    }

    second_cmd = RobotCommand(
        timestamp_ns=1_000_000_000,
        cycle_id=1,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=0.0, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
        source="test",
    )
    applier(None, second_cmd)
    assert dof.v.value == pytest.approx([0.5, 0.0, 0.0])
    assert dof.w.value == pytest.approx([0.0, 0.0, 0.8])
    second_feedback = applier.safety_feedback[dof_key]
    assert second_feedback["clamped_linear"] is True
    assert second_feedback["clamped_angular"] is True
    assert second_feedback["scaled_linear"] == pytest.approx(0.5)
    assert second_feedback["scaled_angular"] == pytest.approx(0.2)

