"""Environment step results."""

from __future__ import annotations

from pydantic import BaseModel

from auto_surgery.schemas.sensors import SensorBundle


class StepResult(BaseModel):
    """Deterministic outcome of a single env step."""

    model_config = {"extra": "forbid"}

    sensors: SensorBundle
    dt: float
    sim_step_index: int
    is_capture_tick: bool
