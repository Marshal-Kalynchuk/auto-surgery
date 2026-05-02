"""Shared lightweight types for model code (no I/O)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TensorSpec(BaseModel):
    """Document expected tensor shapes/dtypes at module boundaries."""

    model_config = {"extra": "forbid"}

    shape: tuple[int, ...]
    dtype: str = Field(default="float32", description="torch dtype name")
