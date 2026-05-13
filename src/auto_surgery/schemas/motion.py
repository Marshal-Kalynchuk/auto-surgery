"""Motion generator configuration contract."""

from __future__ import annotations

import math
from collections.abc import Mapping

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class MotionShaping(BaseModel):
    """Configuration for motion shaping (bounds, scaling, biasing)."""

    model_config = {"extra": "forbid"}

    max_linear_m_s: float = Field(gt=0.0, description="Maximum linear speed in m/s")
    max_angular_rad_s: float = Field(gt=0.0, description="Maximum angular speed in rad/s")
    max_linear_accel_m_s2: float = Field(gt=0.0, description="Maximum linear acceleration in m/s^2")
    max_angular_accel_rad_s2: float = Field(gt=0.0, description="Maximum angular acceleration in rad/s^2")
    bias_gain_max: float = Field(ge=0.0, le=1.0, description="Maximum bias blending gain [0,1]")
    bias_ramp_distance_m: float = Field(gt=0.0, description="Distance over which bias ramps up")
    orientation_bias_gain: float = Field(ge=0.0, le=1.0, description="Orientation bias gain [0,1]")
    orientation_deadband_rad: float = Field(ge=0.0, description="Angular deadband in radians")


_RANGE_CONSTRAINTS: Mapping[str, tuple[float | None, float | None]] = {
    "approach_duration_range_s": (0.0, None),
    "dwell_duration_range_s": (0.0, None),
    "retract_duration_range_s": (0.0, None),
    "retract_distance_range_m": (0.0, None),
    "sweep_duration_range_s": (0.0, None),
    "sweep_arc_range_rad": (0.0, None),
    "rotate_duration_range_s": (0.0, None),
    "rotate_angle_range_rad": (0.0, None),
    "probe_duration_range_s": (0.0, None),
    "probe_hold_range_s": (0.0, None),
    "jaw_value_range": (0.0, 1.0),
}


class MotionGeneratorConfig(BaseModel):
    """Per-episode motion randomisation knobs (piece 3 surface)."""

    model_config = {"extra": "forbid"}

    seed: int = 0
    motion_shaping_enabled: bool = False

    # Sequence shape.
    primitive_count_min: int = Field(default=8, ge=0)
    primitive_count_max: int = Field(default=20, ge=0)

    # Primitive selection weights (un-normalised).
    weight_approach: float = Field(default=1.0, ge=0.0)
    weight_dwell: float = Field(default=0.5, ge=0.0)
    weight_retract: float = Field(default=0.7, ge=0.0)
    weight_sweep: float = Field(default=0.6, ge=0.0)
    weight_rotate: float = Field(default=0.4, ge=0.0)
    weight_probe: float = Field(default=0.8, ge=0.0)
    # New, plan-aligned primitive weights. Keep defaults chosen to preserve
    # backward compatibility with the legacy per-primitive defaults above.
    weight_reach: float = Field(default=1.0, ge=0.0)
    weight_hold: float = Field(default=0.5, ge=0.0)
    weight_contact_reach: float = Field(default=0.7, ge=0.0)
    weight_grip: float = Field(default=0.8, ge=0.0)
    weight_drag: float = Field(default=0.6, ge=0.0)
    weight_brush: float = Field(default=0.4, ge=0.0)

    # Per-primitive parameter ranges (SI units).
    approach_duration_range_s: tuple[float, float] = (0.6, 1.5)
    dwell_duration_range_s: tuple[float, float] = (0.3, 0.8)
    retract_duration_range_s: tuple[float, float] = (0.4, 0.9)
    retract_distance_range_m: tuple[float, float] = (0.003, 0.012)
    sweep_duration_range_s: tuple[float, float] = (0.6, 1.4)
    sweep_arc_range_rad: tuple[float, float] = (0.15, 0.6)
    rotate_duration_range_s: tuple[float, float] = (0.5, 1.2)
    rotate_angle_range_rad: tuple[float, float] = (0.2, 0.8)
    probe_duration_range_s: tuple[float, float] = (0.6, 1.4)
    probe_hold_range_s: tuple[float, float] = (0.15, 0.45)

    # Orientation perturbation applied to Approach / Probe targets.
    target_orientation_jitter_rad: float = Field(default=0.26, ge=0.0, le=math.pi)

    # Jaw randomisation.
    jaw_value_range: tuple[float, float] = (0.0, 1.0)
    jaw_change_probability: float = Field(default=0.4, ge=0.0, le=1.0)

    # Legacy sequencer compatibility fields.
    probe_retract_peak_speed_m_per_s: float = Field(default=0.03, gt=0.0)
    probe_duration_safety_margin_s: float = Field(default=0.05, ge=0.0)
    sweep_axis_bias_scale: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator(*_RANGE_CONSTRAINTS)
    @classmethod
    def _validate_numeric_ranges(
        cls,
        value: tuple[float, float],
        info: ValidationInfo,
    ) -> tuple[float, float]:
        field_name = info.field_name or "range"
        minimum, maximum = _RANGE_CONSTRAINTS[field_name]
        return cls._validate_range(
            field_name,
            value,
            minimum=minimum,
            maximum=maximum,
            label=f"{field_name} range"
        )

    @model_validator(mode="after")
    def _validate_domain_constraints(self) -> "MotionGeneratorConfig":
        if self.primitive_count_max < self.primitive_count_min:
            raise ValueError(
                "primitive_count_max must be greater than or equal to primitive_count_min."
            )
        return self

    @staticmethod
    def _validate_range(
        name: str,
        values: tuple[float, float],
        *,
        minimum: float = 0.0,
        maximum: float | None = None,
        label: str = "Range",
    ) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"{label} '{name}' must contain exactly two values.")

        minimum_value = float(values[0])
        maximum_value = float(values[1])
        if not math.isfinite(minimum_value) or not math.isfinite(maximum_value):
            raise ValueError(f"{label} '{name}' values must be finite numbers.")

        if minimum_value < minimum or maximum_value < minimum:
            raise ValueError(f"{label} '{name}' values must be >= {minimum}.")
        if maximum is not None and (minimum_value > maximum or maximum_value > maximum):
            raise ValueError(f"{label} '{name}' values must be <= {maximum}.")

        if minimum_value > maximum_value:
            raise ValueError(f"{label} '{name}' minimum must be <= maximum.")

        return (minimum_value, maximum_value)
