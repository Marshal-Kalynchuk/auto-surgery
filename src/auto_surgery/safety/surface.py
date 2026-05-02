"""SafetySurface envelope (v0 minimal fields)."""

from typing import Any

from pydantic import BaseModel, Field


class SafetySurface(BaseModel):
    model_config = {"extra": "forbid"}

    schema_version: str = "safety_surface_v0"
    entity_constraints: dict[str, Any] = Field(default_factory=dict)
