"""Pluggable synthetic command generators for simulation rollouts."""

from __future__ import annotations

import math
import random
from typing import Any, Protocol

from auto_surgery.schemas.commands import RobotCommand


class ActionGenerator(Protocol):
    """Produces the next `RobotCommand` for a rollout step."""

    def next_command(self, *, step_index: int, timestamp_ns: int) -> RobotCommand: ...


class SineJointGenerator:
    """Deterministic single-joint sinusoid (legacy Stage-0 smoke)."""

    def __init__(
        self,
        *,
        joint_key: str = "j0",
        amplitude: float = 0.05,
        phase_scale: float = 0.1,
    ) -> None:
        self._joint_key = joint_key
        self._amplitude = amplitude
        self._phase_scale = phase_scale

    def next_command(self, *, step_index: int, timestamp_ns: int) -> RobotCommand:
        t = step_index * self._phase_scale
        joint = self._amplitude * math.sin(t)
        return RobotCommand(
            timestamp_ns=timestamp_ns,
            joint_positions={self._joint_key: float(joint)},
            representation="joint",
        )


class CoherentRandomWalkGenerator:
    """Bounded random walk with momentum on a single joint (POC coherent motion)."""

    def __init__(
        self,
        *,
        joint_key: str = "j0",
        rng: random.Random | None = None,
        step_sigma: float = 0.02,
        max_abs: float = 0.25,
        momentum: float = 0.85,
    ) -> None:
        self._joint_key = joint_key
        self._rng = rng or random.Random()
        self._step_sigma = float(step_sigma)
        self._max_abs = float(max_abs)
        self._momentum = float(momentum)
        self._velocity = 0.0

    def next_command(self, *, step_index: int, timestamp_ns: int) -> RobotCommand:
        _ = step_index
        noise = self._rng.gauss(0.0, self._step_sigma)
        self._velocity = self._momentum * self._velocity + (1.0 - self._momentum) * noise
        raw = self._velocity * 5.0
        clamped = max(-self._max_abs, min(self._max_abs, raw))
        return RobotCommand(
            timestamp_ns=timestamp_ns,
            joint_positions={self._joint_key: float(clamped)},
            representation="joint",
        )


def build_default_action_generator(config: dict[str, Any] | None) -> ActionGenerator:
    """Factory used by rollouts: `kind` selects implementation."""

    cfg = dict(config or {})
    kind = str(cfg.get("kind", "sine")).lower().strip()
    if kind in ("sine", "sin", "lite"):
        return SineJointGenerator(
            joint_key=str(cfg.get("joint_key", "j0")),
            amplitude=float(cfg.get("amplitude", 0.05)),
            phase_scale=float(cfg.get("phase_scale", 0.1)),
        )
    if kind in ("random_walk", "walk", "coherent_random_walk"):
        seed = cfg.get("seed")
        rng = random.Random(int(seed)) if seed is not None else None
        return CoherentRandomWalkGenerator(
            joint_key=str(cfg.get("joint_key", "j0")),
            rng=rng,
            step_sigma=float(cfg.get("step_sigma", 0.02)),
            max_abs=float(cfg.get("max_abs", 0.25)),
            momentum=float(cfg.get("momentum", 0.85)),
        )
    raise ValueError(f"Unknown action generator kind: {kind!r}")
