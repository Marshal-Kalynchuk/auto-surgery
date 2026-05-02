"""Sensor bundle — multimodal payloads keyed by modality (architecture §5.1)."""

from typing import Any

from pydantic import BaseModel, Field


class SensorBundle(BaseModel):
    """All synchronized sensor streams for one sampling instant."""

    model_config = {"extra": "forbid"}

    timestamp_ns: int
    clock_source: str = Field(
        ...,
        description="ptp | ntp | monotonic — documented per deployment.",
    )
    modalities: dict[str, Any] = Field(
        default_factory=dict,
        description="Keys e.g. stereo_rgb, depth, kinematics, wrench, audio — values or blob refs.",
    )
