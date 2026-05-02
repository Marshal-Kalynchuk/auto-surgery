"""Scene graph / entity-knowledge-store snapshot (embedding-first slots, architecture §3.1)."""

from typing import Any

from pydantic import BaseModel, Field


class SlotRecord(BaseModel):
    """One persistent slot instance."""

    model_config = {"extra": "forbid"}

    slot_id: str
    pose: dict[str, float] | None = None
    geometry_ref: str | None = Field(default=None, description="Blob URI for mesh/pointcloud.")
    embedding_ref: str | None = Field(default=None, description="Pointer to embedding tensor blob.")
    derived_labels: dict[str, Any] = Field(default_factory=dict)
    surgeon_tags: list[dict[str, Any]] = Field(default_factory=list)


class SceneGraph(BaseModel):
    """Snapshot of slots and lightweight scene metadata."""

    model_config = {"extra": "forbid"}

    schema_version: str = "scene_v1"
    frame_index: int = 0
    slots: list[SlotRecord] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
