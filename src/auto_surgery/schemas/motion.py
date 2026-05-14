"""Motion generator configuration contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class MotionShaping(BaseModel):
    """Configuration for motion shaping (bounds, scaling, biasing).

    Speed and acceleration caps apply to the pose-servo applier (scene-frame
    Cartesian pose commands). Fields ``bias_gain_max``, ``bias_ramp_distance_mm``,
    ``orientation_bias_gain``, and ``orientation_deadband_rad`` are retained for
    backward-compatible YAML only; they are not consumed by the pose-servo path.
    """

    model_config = {"extra": "forbid"}

    max_linear_mm_s: float = Field(gt=0.0, description="Maximum linear speed in mm/s (scene units)")
    max_angular_rad_s: float = Field(gt=0.0, description="Maximum angular speed in rad/s")
    max_linear_accel_mm_s2: float = Field(gt=0.0, description="Maximum linear acceleration in mm/s^2")
    max_angular_accel_rad_s2: float = Field(gt=0.0, description="Maximum angular acceleration in rad/s^2")
    bias_gain_max: float = Field(
        ge=0.0,
        le=1.0,
        description="Legacy: not used by pose-servo applier. Maximum bias blending gain [0,1].",
    )
    bias_ramp_distance_mm: float = Field(
        gt=0.0,
        description="Legacy: not used by pose-servo applier. Distance in mm over which bias ramps up.",
    )
    orientation_bias_gain: float = Field(
        ge=0.0,
        le=1.0,
        description="Legacy: not used by pose-servo applier. Orientation bias gain [0,1].",
    )
    orientation_deadband_rad: float = Field(
        ge=0.0,
        description="Legacy: not used by pose-servo applier. Angular deadband in radians.",
    )
    frustum_margin_mm: float | None = Field(
        default=None,
        description="Reserved for future FOV-aware target sampling. Not consumed by current code.",
    )


_LEGACY_WEIGHT_TO_CANONICAL: tuple[tuple[str, str], ...] = (
    ("weight_approach", "weight_reach"),
    ("weight_dwell", "weight_hold"),
    ("weight_retract", "weight_drag"),
    ("weight_sweep", "weight_brush"),
    ("weight_rotate", "weight_grip"),
    ("weight_probe", "weight_contact_reach"),
)

_LEGACY_DURATION_TO_CANONICAL: tuple[tuple[str, str], ...] = (
    ("approach_duration_range_s", "reach_duration_range_s"),
    ("dwell_duration_range_s", "hold_duration_range_s"),
    ("retract_duration_range_s", "drag_duration_range_s"),
    ("sweep_duration_range_s", "brush_duration_range_s"),
    ("sweep_arc_range_rad", "brush_arc_range_rad"),
)


_RANGE_CONSTRAINTS: Mapping[str, tuple[float | None, float | None]] = {
    "reach_duration_range_s": (0.0, None),
    "hold_duration_range_s": (0.0, None),
    "drag_duration_range_s": (0.0, None),
    "drag_distance_range_mm": (0.0, None),
    "brush_duration_range_s": (0.0, None),
    "brush_arc_range_rad": (0.0, None),
    "rotate_duration_range_s": (0.0, None),
    "rotate_angle_range_rad": (0.0, None),
    "probe_duration_range_s": (0.0, None),
    "probe_hold_range_s": (0.0, None),
    "jaw_value_range": (0.0, 1.0),
}


class MotionGeneratorConfig(BaseModel):
    """Per-episode motion randomisation knobs (piece 3 surface).

    YAML uses canonical ``weight_*`` and ``*_duration_range_s`` / ``*_range_*`` names.
    Legacy keys (for example ``weight_approach``, ``approach_duration_range_s``) are
    accepted at load time: they are copied onto the canonical field when the canonical
    key is absent, then removed so ``extra=forbid`` validation stays strict.
    """

    model_config = {"extra": "forbid"}

    seed: int = 0
    motion_shaping_enabled: bool = False
    motion_shaping: MotionShaping | None = None

    # Sequence shape.
    primitive_count_min: int = Field(default=8, ge=0)
    primitive_count_max: int = Field(default=20, ge=0)

    # Primitive selection weights (un-normalised), canonical names aligned with
    # ``PrimitiveKind`` in ``motion/primitives.py``.
    weight_reach: float = Field(default=1.0, ge=0.0)
    weight_hold: float = Field(default=0.5, ge=0.0)
    weight_contact_reach: float = Field(default=0.7, ge=0.0)
    weight_grip: float = Field(default=0.8, ge=0.0)
    weight_drag: float = Field(default=0.6, ge=0.0)
    weight_brush: float = Field(default=0.4, ge=0.0)

    # Per-primitive parameter ranges: time/angle in SI seconds and radians;
    # drag distance in scene millimetres (DejaVu brain meshes).
    reach_duration_range_s: tuple[float, float] = (0.6, 1.5)
    hold_duration_range_s: tuple[float, float] = (0.3, 0.8)
    drag_duration_range_s: tuple[float, float] = (0.4, 0.9)
    drag_distance_range_mm: tuple[float, float] = (3.0, 12.0)
    brush_duration_range_s: tuple[float, float] = (0.6, 1.4)
    brush_arc_range_rad: tuple[float, float] = (0.15, 0.6)

    # Reserved for future motion variants (not read by the current sequencer).
    rotate_duration_range_s: tuple[float, float] = (0.5, 1.2)
    rotate_angle_range_rad: tuple[float, float] = (0.2, 0.8)
    probe_duration_range_s: tuple[float, float] = (0.6, 1.4)
    probe_hold_range_s: tuple[float, float] = (0.15, 0.45)

    # Orientation perturbation applied to Reach / ContactReach target rotations.
    target_orientation_jitter_rad: float = Field(default=0.26, ge=0.0, le=math.pi)

    # Jaw randomisation.
    jaw_value_range: tuple[float, float] = (0.0, 1.0)
    jaw_change_probability: float = Field(default=0.4, ge=0.0, le=1.0)

    probe_retract_peak_speed_mm_per_s: float = Field(default=30.0, gt=0.0)
    probe_duration_safety_margin_s: float = Field(default=0.05, ge=0.0)
    sweep_axis_bias_scale: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_motion_yaml(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for legacy, canonical in _LEGACY_WEIGHT_TO_CANONICAL:
            if canonical not in out and legacy in out:
                out[canonical] = out[legacy]
            if legacy in out:
                del out[legacy]
        for legacy, canonical in _LEGACY_DURATION_TO_CANONICAL:
            if canonical not in out and legacy in out:
                out[canonical] = out[legacy]
            if legacy in out:
                del out[legacy]
        # Interim / mistaken YAML key from an earlier mm rename (same semantics as drag).
        if "drag_distance_range_mm" not in out and "retract_distance_range_mm" in out:
            out["drag_distance_range_mm"] = out["retract_distance_range_mm"]
        if "retract_distance_range_mm" in out:
            del out["retract_distance_range_mm"]
        return out

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
            label=f"{field_name} range",
        )

    @model_validator(mode="after")
    def _validate_domain_constraints(self) -> "MotionGeneratorConfig":
        if self.primitive_count_max < self.primitive_count_min:
            raise ValueError(
                "primitive_count_max must be greater than or equal to primitive_count_min.",
            )
        return self

    def with_default_motion_shaping(self, scene_id: str) -> "MotionGeneratorConfig":
        """Return a copy with scene-default ``MotionShaping`` when none is set in YAML.

        When ``motion_shaping`` is already populated, returns ``self`` unchanged so
        explicit YAML overrides win over ``motion_shaping_defaults.yaml``.
        """

        if self.motion_shaping is not None:
            return self
        from auto_surgery.config import load_scene_motion_shaping

        return self.model_copy(
            update={
                "motion_shaping": load_scene_motion_shaping(scene_id),
                "motion_shaping_enabled": True,
            },
        )

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
