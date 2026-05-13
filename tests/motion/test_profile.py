from __future__ import annotations

from math import isfinite

import pytest

from auto_surgery.motion.profile import (
    min_jerk_position_scalar,
    min_jerk_retract_duration,
    min_jerk_velocity_scalar,
)


def test_min_jerk_position_scalar_edge_cases() -> None:
    assert min_jerk_position_scalar(-1.0) == 0.0
    assert min_jerk_position_scalar(0.0) == 0.0
    assert min_jerk_position_scalar(1.0) == 1.0
    assert min_jerk_position_scalar(2.0) == 1.0


def test_min_jerk_position_scalar_midpoint() -> None:
    midpoint = min_jerk_position_scalar(0.5)
    assert midpoint == pytest.approx(0.5)


def test_min_jerk_velocity_scalar_edge_cases() -> None:
    assert min_jerk_velocity_scalar(-0.1, 1.0) == 0.0
    assert min_jerk_velocity_scalar(0.0, 1.0) == 0.0
    assert min_jerk_velocity_scalar(1.0, 1.0) == 0.0
    assert min_jerk_velocity_scalar(0.5, 0.0) == 0.0
    assert isfinite(min_jerk_velocity_scalar(0.5, 2.0))


def test_min_jerk_velocity_scalar_scaling() -> None:
    assert min_jerk_velocity_scalar(0.5, 0.25) == pytest.approx(1.875 / 0.25)
    assert min_jerk_velocity_scalar(0.5, 0.5) == pytest.approx(1.875 / 0.5)


def test_min_jerk_retract_duration_returns_zero_for_invalid_inputs() -> None:
    assert min_jerk_retract_duration(distance_m=0.0, peak_retract_speed_m_per_s=0.0) == 0.0
    assert min_jerk_retract_duration(distance_m=0.1, peak_retract_speed_m_per_s=0.05) == pytest.approx(3.75)
    assert min_jerk_retract_duration(distance_m=0.1, peak_retract_speed_m_per_s=-0.05) == pytest.approx(3.75)


def test_min_jerk_retract_duration_formula() -> None:
    distance = 0.01
    speed = 0.02
    expected = 1.875 * distance / speed
    assert min_jerk_retract_duration(distance_m=distance, peak_retract_speed_m_per_s=speed) == pytest.approx(expected)
