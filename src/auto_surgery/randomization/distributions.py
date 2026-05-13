"""Typed distribution primitives for piece-4 randomization.

These classes remain pure, numpy-only, and intentionally lightweight so they can be
reused by deterministic samplers and schema-driven YAML loaders.
"""

from __future__ import annotations

import math
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel, Field, model_validator

from auto_surgery.schemas.commands import Vec3

T = TypeVar("T")


class Range(BaseModel):
    """Inclusive scalar interval."""

    model_config = {"extra": "forbid"}

    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> "Range":
        if not self.low <= self.high:
            raise ValueError(f"Range requires low <= high; got [{self.low}, {self.high}].")
        return self

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(float(self.low), float(self.high)))


class LogRange(BaseModel):
    """Uniform in log10-space, samples are exp(log_uniform(log(low), log(high)))."""

    model_config = {"extra": "forbid"}

    low: float = Field(gt=0.0)
    high: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> "LogRange":
        if not self.low <= self.high:
            raise ValueError(
                f"LogRange requires low <= high; got [{self.low}, {self.high}]."
            )
        return self

    def sample(self, rng: np.random.Generator) -> float:
        return float(10 ** rng.uniform(math.log10(float(self.low)), math.log10(float(self.high))))


class Vec3Range(BaseModel):
    """Component-wise independent ranges."""

    model_config = {"extra": "forbid"}

    x: Range
    y: Range
    z: Range

    def sample(self, rng: np.random.Generator) -> Vec3:
        return Vec3(
            x=float(self.x.sample(rng)),
            y=float(self.y.sample(rng)),
            z=float(self.z.sample(rng)),
        )


class Choice(BaseModel, Generic[T]):
    """Discrete categorical with optional weights (uniform if absent)."""

    model_config = {"extra": "forbid"}

    options: list[T] = Field(min_length=1)
    weights: list[float] | None = None

    @model_validator(mode="after")
    def _validate_weights(self) -> "Choice[T]":
        if self.weights is not None:
            if len(self.weights) != len(self.options):
                raise ValueError("Choice.weights length must match Choice.options.")
            if any(w < 0 for w in self.weights):
                raise ValueError("Choice.weights must be non-negative.")
            if sum(self.weights) == 0:
                raise ValueError("Choice.weights must have a positive sum.")
        return self

    def sample(self, rng: np.random.Generator) -> T:
        if len(self.options) == 1:
            return self.options[0]
        if self.weights is None:
            index = int(rng.integers(0, len(self.options)))
            return self.options[index]
        normalized = np.asarray(self.weights, dtype=float)
        normalized /= float(np.sum(normalized))
        return self.options[int(rng.choice(len(self.options), p=normalized))]
