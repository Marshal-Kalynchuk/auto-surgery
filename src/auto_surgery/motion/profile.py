"""Minimum-jerk profile helpers for piece-3 motion primitives."""

from __future__ import annotations

import math


# x6 minimum-jerk peak speed occurs at 0.8*T where velocity is (30*0.8^2 - 60*0.8^3 + 30*0.8^4) / T
# = 1.875 / T. Therefore T = 1.875 * distance / desired_peak.
_MIN_RETRACT_PEAK_FACTOR = 1.875


def min_jerk_position_scalar(tau: float) -> float:
    """Return minimum-jerk position fraction at normalized phase ``tau``.

    The scalar is in ``[0, 1]`` with zero slope at both ends.
    """
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return 1.0

    value = float(tau)
    return (10.0 * value**3) - (15.0 * value**4) + (6.0 * value**5)


def min_jerk_velocity_scalar(tau: float, duration_s: float) -> float:
    """Return minimum-jerk velocity multiplier at normalized phase ``tau``.

    The returned value is already divided by ``duration_s``.
    """
    if duration_s <= 0.0 or not math.isfinite(float(duration_s)):
        return 0.0
    if tau <= 0.0 or tau >= 1.0:
        return 0.0

    value = float(tau)
    return (
        (30.0 * value**2 - 60.0 * value**3 + 30.0 * value**4)
        / float(duration_s)
    )


def min_jerk_retract_duration(*, distance_m: float, peak_retract_speed_m_per_s: float) -> float:
    """Return the minimum jerk duration needed to retract ``distance_m``.

    Returns 0.0 when the duration is not physically meaningful.
    """
    distance = abs(float(distance_m))
    speed = abs(float(peak_retract_speed_m_per_s))
    if distance <= 0.0 or speed <= 0.0 or not math.isfinite(distance) or not math.isfinite(speed):
        return 0.0
    return _MIN_RETRACT_PEAK_FACTOR * distance / speed
