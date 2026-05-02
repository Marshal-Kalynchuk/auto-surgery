"""Environment step results (architecture §8.3.2)."""

from typing import Any

from pydantic import BaseModel, Field

from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import SensorBundle


class StepResult(BaseModel):
    """Return type of Environment.step."""

    model_config = {"extra": "forbid"}

    next_scene: SceneGraph
    sensor_observation: SensorBundle
    info: dict[str, Any] = Field(default_factory=dict)
