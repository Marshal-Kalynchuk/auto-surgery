"""Pluggable synthetic command generators for simulation rollouts."""

from __future__ import annotations

import math
import random
from typing import Any, Protocol

from auto_surgery.schemas.commands import ControlMode, RobotCommand, Twist, Vec3


class ActionGenerator(Protocol):
    """Produces the next `RobotCommand` for a rollout step."""

    def next_command(self, *, step_index: int, timestamp_ns: int) -> RobotCommand: ...


def build_sine_joint_command(
    step_index: int,
    *,
    base_ns: int = 1_000_000,
    amplitude: float = 0.05,
    phase_scale: float = 0.1,
    joint_start: float = 0.0,
    joint_step: float = 0.0,
    joint_key: str = "j0",
) -> RobotCommand:
    """Build a deterministic one-joint sine command."""

    joint = build_sine_joint_position(
        step_index,
        joint_start=joint_start,
        joint_step=joint_step,
        amplitude=amplitude,
        phase_scale=phase_scale,
    )
    return RobotCommand(
        timestamp_ns=base_ns + step_index,
        cycle_id=step_index,
        control_mode=ControlMode.JOINT_POSITION,
        enable=True,
        source="scripted",
        joint_positions={joint_key: float(joint)},
    )


def build_sine_twist_command(
    step_index: int,
    *,
    base_ns: int = 1_000_000,
    linear_amplitude: float = 0.05,
    phase_scale: float = 0.1,
    linear_axis: str = "x",
    scale_axis: float = 1.0,
) -> RobotCommand:
    """Build a deterministic single-axis sinusoidal twist command."""

    linear = build_sine_joint_position(
        step_index,
        amplitude=linear_amplitude,
        phase_scale=phase_scale,
    )
    linear_axes = {
        "x": (float(linear) * scale_axis, 0.0, 0.0),
        "y": (0.0, float(linear) * scale_axis, 0.0),
        "z": (0.0, 0.0, float(linear) * scale_axis),
    }
    vx, vy, vz = linear_axes.get(linear_axis, (float(linear) * scale_axis, 0.0, 0.0))
    return RobotCommand(
        timestamp_ns=base_ns + step_index,
        cycle_id=step_index,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=vx, y=vy, z=vz),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
        source="scripted",
    )


def build_sine_joint_position(
    step_index: int,
    *,
    joint_start: float = 0.0,
    joint_step: float = 0.0,
    amplitude: float = 0.05,
    phase_scale: float = 0.1,
) -> float:
    """Build a one-dimensional sine+offset joint trajectory sample."""

    t = step_index * phase_scale
    return float(joint_start + joint_step * step_index + amplitude * math.sin(t))


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
        return build_sine_joint_command(
            step_index,
            base_ns=timestamp_ns - step_index,
            amplitude=self._amplitude,
            phase_scale=self._phase_scale,
            joint_key=self._joint_key,
            joint_start=0.0,
            joint_step=0.0,
        )


class SineTwistGenerator:
    """Deterministic single-axis sinusoid for primary Stage-1 path."""

    def __init__(
        self,
        *,
        linear_amplitude: float = 0.05,
        phase_scale: float = 0.1,
        linear_axis: str = "x",
    ) -> None:
        self._linear_amplitude = linear_amplitude
        self._phase_scale = phase_scale
        self._linear_axis = linear_axis

    def next_command(self, *, step_index: int, timestamp_ns: int) -> RobotCommand:
        return build_sine_twist_command(
            step_index,
            base_ns=timestamp_ns - step_index,
            linear_amplitude=self._linear_amplitude,
            phase_scale=self._phase_scale,
            linear_axis=self._linear_axis,
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
            cycle_id=step_index,
            control_mode=ControlMode.JOINT_POSITION,
            joint_positions={self._joint_key: float(clamped)},
            enable=True,
            source="scripted",
        )


class CoherentRandomWalkTwistGenerator:
    """Bounded one-axis random walk mapped to primary twist path."""

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        linear_scale: float = 0.02,
        max_abs: float = 0.25,
        momentum: float = 0.85,
        linear_axis: str = "x",
    ) -> None:
        self._rng = rng or random.Random()
        self._linear_scale = float(linear_scale)
        self._max_abs = float(max_abs)
        self._momentum = float(momentum)
        self._velocity = 0.0
        self._linear_axis = linear_axis

    def next_command(self, *, step_index: int, timestamp_ns: int) -> RobotCommand:
        _ = step_index
        noise = self._rng.gauss(0.0, self._linear_scale)
        self._velocity = self._momentum * self._velocity + (1.0 - self._momentum) * noise
        clamped = max(-self._max_abs, min(self._max_abs, self._velocity * 5.0))
        linear_axes = {
            "x": (float(clamped), 0.0, 0.0),
            "y": (0.0, float(clamped), 0.0),
            "z": (0.0, 0.0, float(clamped)),
        }
        vx, vy, vz = linear_axes.get(self._linear_axis, (float(clamped), 0.0, 0.0))
        return RobotCommand(
            timestamp_ns=timestamp_ns,
            cycle_id=step_index,
            control_mode=ControlMode.CARTESIAN_TWIST,
            cartesian_twist=Twist(
                linear=Vec3(x=vx, y=vy, z=vz),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            enable=True,
            source="scripted",
        )


def build_default_action_generator(config: dict[str, Any] | None) -> ActionGenerator:
    """Factory used by rollouts: `kind` selects implementation."""

    cfg = dict(config or {})
    kind = str(cfg.get("kind", "sine")).lower().strip()
    if kind in ("sine", "sin", "lite"):
        return SineTwistGenerator(
            linear_amplitude=float(cfg.get("amplitude", 0.05)),
            phase_scale=float(cfg.get("phase_scale", 0.1)),
            linear_axis=str(cfg.get("linear_axis", "x")),
        )
    if kind in ("random_walk", "walk", "coherent_random_walk"):
        seed = cfg.get("seed")
        rng = random.Random(int(seed)) if seed is not None else None
        return CoherentRandomWalkTwistGenerator(
            rng=rng,
            linear_scale=float(cfg.get("linear_scale", 0.02)),
            max_abs=float(cfg.get("max_abs", 0.25)),
            momentum=float(cfg.get("momentum", 0.85)),
            linear_axis=str(cfg.get("linear_axis", "x")),
        )
    if kind in ("sine_joint", "legacy_sine_joint"):
        return SineJointGenerator(
            joint_key=str(cfg.get("joint_key", "j0")),
            amplitude=float(cfg.get("amplitude", 0.05)),
            phase_scale=float(cfg.get("phase_scale", 0.1)),
        )
    if kind in ("random_walk_joint", "walk_joint", "legacy_random_walk"):
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
