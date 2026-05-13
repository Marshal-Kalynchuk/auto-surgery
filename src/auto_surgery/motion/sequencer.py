"""Primitive sequencing and stochastic parameter sampling for motion generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from auto_surgery.motion.profile import min_jerk_retract_duration
from auto_surgery.motion.primitives import (
    Approach,
    Dwell,
    Probe,
    Primitive,
    PrimitiveKind,
    Retract,
    Rotate,
    Sweep,
)
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.motion import MotionGeneratorConfig


_EPS = 1.0e-12


@dataclass(frozen=True)
class _InternalTargetVolume:
    label: str
    center_scene: Vec3
    half_extents_scene: Vec3
    shape: str = "sphere"


@dataclass(frozen=True)
class Sequencer:
    """Sample primitives lazily and deterministically per episode."""

    motion_config: MotionGeneratorConfig
    scene_config: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rng", np.random.default_rng(self.motion_config.seed))
        object.__setattr__(self, "_planned_count", 0)
        object.__setattr__(self, "_issued", 0)
        object.__setattr__(self, "_initial_step", None)

    def reset(self, initial_step: StepResult) -> None:
        """Reset the generator state for a new episode."""
        object.__setattr__(self, "_rng", np.random.default_rng(self.motion_config.seed))
        min_count = int(self.motion_config.primitive_count_min)
        max_count = int(self.motion_config.primitive_count_max)
        if max_count < min_count:
            max_count = min_count
        planned_count = int(self._rng.integers(low=min_count, high=max_count + 1))
        object.__setattr__(self, "_planned_count", planned_count)
        object.__setattr__(self, "_issued", 0)
        object.__setattr__(self, "_initial_step", initial_step)

    def next_primitive(self, last_step: StepResult, last_jaw: float) -> Primitive | None:
        del last_jaw
        if self._issued >= self._planned_count:
            return None

        if self._issued == 0 and self._planned_count > 0:
            primitive = self._build_first_approach(last_step)
        else:
            kind = self._sample_kind()
            match kind:
                case PrimitiveKind.APPROACH:
                    primitive = self._build_approach(last_step)
                case PrimitiveKind.DWELL:
                    primitive = self._build_dwell()
                case PrimitiveKind.RETRACT:
                    primitive = self._build_retract()
                case PrimitiveKind.SWEEP:
                    primitive = self._build_sweep()
                case PrimitiveKind.ROTATE:
                    primitive = self._build_rotate()
                case PrimitiveKind.PROBE:
                    primitive = self._build_probe(last_step)
                case _:
                    primitive = self._build_dwell()

        object.__setattr__(self, "_issued", self._issued + 1)
        return primitive

    def _sample_kind(self) -> PrimitiveKind:
        candidates = (
            PrimitiveKind.APPROACH,
            PrimitiveKind.DWELL,
            PrimitiveKind.RETRACT,
            PrimitiveKind.SWEEP,
            PrimitiveKind.ROTATE,
            PrimitiveKind.PROBE,
        )
        weights = (
            float(self.motion_config.weight_approach),
            float(self.motion_config.weight_dwell),
            float(self.motion_config.weight_retract),
            float(self.motion_config.weight_sweep),
            float(self.motion_config.weight_rotate),
            float(self.motion_config.weight_probe),
        )
        total = sum(max(0.0, weight) for weight in weights)
        if total <= 0.0:
            return PrimitiveKind.APPROACH

        needle = float(self._rng.uniform(0.0, total))
        cumulative = 0.0
        for candidate, weight in zip(candidates, weights):
            cumulative += max(0.0, float(weight))
            if needle <= cumulative:
                return candidate
        return candidates[-1]

    def _build_first_approach(self, last_step: StepResult) -> Approach:
        volumes = self._sample_target_volumes(last_step, fallback_to_current=True)
        if volumes:
            target_center = volumes[0].center_scene
        elif last_step is not None:
            target_center = self._tool_pose_scene(last_step).position
        else:
            target_center = Vec3(x=0.0, y=0.0, z=0.0)
        start_pose_scene = self._tool_pose_scene(last_step)
        jaw_start, jaw_end = self._sample_jaw_targets()

        return Approach(
            target_pose_scene=Pose(
                position=target_center,
                rotation=_compose_with_jitter(
                    base=start_pose_scene.rotation,
                    jitter_rad=0.0,
                    rng=self._rng,
                ),
            ),
            duration_s=0.5 * sum(self.motion_config.approach_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
            end_on_contact=True,
        )

    def _build_approach(self, last_step: StepResult) -> Approach:
        start_pose_scene = self._tool_pose_scene(last_step)
        target_pose = self._sample_target_pose(last_step, start_pose_scene)
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Approach(
            target_pose_scene=target_pose,
            duration_s=self._sample_range(self.motion_config.approach_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
            end_on_contact=True,
        )

    def _build_dwell(self) -> Dwell:
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Dwell(
            duration_s=self._sample_range(self.motion_config.dwell_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _build_retract(self) -> Retract:
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Retract(
            distance_m=self._sample_range(self.motion_config.retract_distance_range_m),
            duration_s=self._sample_range(self.motion_config.retract_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _build_sweep(self) -> Sweep:
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Sweep(
            axis_camera=_vector_to_vec3(self._sample_sweep_axis()),
            arc_radians=self._sample_signed_range(self.motion_config.sweep_arc_range_rad),
            duration_s=self._sample_range(self.motion_config.sweep_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _build_rotate(self) -> Rotate:
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Rotate(
            axis_camera=_vector_to_vec3(_normalize(self._rng.normal(size=3))),
            angle_radians=self._sample_signed_range(self.motion_config.rotate_angle_range_rad),
            duration_s=self._sample_range(self.motion_config.rotate_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _build_probe(self, last_step: StepResult) -> Probe:
        start_pose_scene = self._tool_pose_scene(last_step)
        target_pose = self._sample_target_pose(last_step, start_pose_scene)
        approach_time = self._sample_range(self.motion_config.probe_duration_range_s)
        hold_after_contact = self._sample_range(self.motion_config.probe_hold_range_s)
        retract_distance = self._sample_range(self.motion_config.retract_distance_range_m)
        retract_time = min_jerk_retract_duration(
            distance_m=float(retract_distance),
            peak_retract_speed_m_per_s=float(self.motion_config.probe_retract_peak_speed_m_per_s),
        )
        duration = (
            approach_time
            + hold_after_contact
            + retract_time
            + float(self.motion_config.probe_duration_safety_margin_s)
        )

        jaw_start, jaw_end = self._sample_jaw_targets()
        return Probe(
            target_pose_scene=target_pose,
            hold_after_contact_s=hold_after_contact,
            retract_distance_m=retract_distance,
            retract_peak_speed_m_per_s=float(self.motion_config.probe_retract_peak_speed_m_per_s),
            duration_s=duration,
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _sample_jaw_targets(self) -> tuple[float | None, float | None]:
        if self._rng.random() >= float(self.motion_config.jaw_change_probability):
            return None, None

        low, high = self.motion_config.jaw_value_range
        if float(low) == float(high):
            return None, float(low)

        return None, float(self._rng.uniform(min(low, high), max(low, high)))

    def _sample_target_pose(self, last_step: StepResult, start_pose_scene: Pose) -> Pose:
        volume = self._sample_target_volume(last_step)
        center_m = _vec_to_array(volume.center_scene)
        half_extents_m = _vec_to_array(volume.half_extents_scene)
        position = _vector_to_vec3(
            _sample_point_in_volume(
                rng=self._rng,
                shape=volume.shape,
                center=center_m,
                half_extents=half_extents_m,
            )
        )
        rotation = _compose_with_jitter(
            base=start_pose_scene.rotation,
            jitter_rad=float(self.motion_config.target_orientation_jitter_rad),
            rng=self._rng,
        )
        return Pose(position=position, rotation=rotation)

    def _sample_target_volume(
        self,
        last_step: StepResult,
        fallback_to_current: bool = False,
    ) -> _InternalTargetVolume:
        volumes = self._sample_target_volumes(last_step, fallback_to_current=fallback_to_current)
        if not volumes:
            return _InternalTargetVolume(
                label="general",
                center_scene=Vec3(x=0.0, y=0.0, z=0.0),
                half_extents_scene=Vec3(x=0.0, y=0.0, z=0.0),
                shape="sphere",
            )
        return volumes[int(self._rng.integers(0, len(volumes)))]

    def _sample_target_volumes(
        self,
        last_step: StepResult,
        fallback_to_current: bool = False,
    ) -> tuple[_InternalTargetVolume, ...]:
        raw_volumes = getattr(self.scene_config, "target_volumes", ())
        if raw_volumes:
            return tuple(_to_internal_target_volume(volume) for volume in raw_volumes)

        if not fallback_to_current:
            return ()

        if last_step is None and self._initial_step is not None:
            last_step = self._initial_step
        if last_step is not None:
            pose = self._tool_pose_scene(last_step)
            return (
                _InternalTargetVolume(
                    label="general",
                    center_scene=pose.position,
                    half_extents_scene=Vec3(x=0.0, y=0.0, z=0.0),
                    shape="sphere",
                ),
            )

        return ()

    def _sample_sweep_axis(self) -> np.ndarray:
        xy = self._rng.normal(size=2)
        if float(np.linalg.norm(xy)) <= _EPS:
            xy = np.array([1.0, 0.0], dtype=float)
        xy = _normalize(xy)
        z = float(
            self._rng.uniform(
                -abs(float(self.motion_config.sweep_axis_bias_scale)),
                abs(float(self.motion_config.sweep_axis_bias_scale)),
            )
        )
        return _normalize(np.array([xy[0], xy[1], z], dtype=float))

    def _sample_range(self, values: tuple[float, float]) -> float:
        low = float(values[0])
        high = float(values[1])
        return float(self._rng.uniform(min(low, high), max(low, high)))

    def _sample_signed_range(self, values: tuple[float, float]) -> float:
        low = float(values[0])
        high = float(values[1])
        magnitude = max(abs(low), abs(high))
        return float(self._rng.uniform(-magnitude, magnitude))

    def _tool_pose_scene(self, step: StepResult) -> Pose:
        tool_pose = step.sensors.tool.pose
        if not step.sensors.cameras:
            return tool_pose

        extrinsics = step.sensors.cameras[0].extrinsics
        rot = _quat_to_matrix(extrinsics.rotation)
        position = _apply_rotation(rot, _vec_to_array(tool_pose.position)) + _vec_to_array(extrinsics.position)
        orientation = _matrix_to_quat(rot @ _quat_to_matrix(tool_pose.rotation))
        return Pose(position=_vector_to_vec3(position), rotation=orientation)


# Compatibility alias for existing callers.
_Sequencer = Sequencer


def _to_internal_target_volume(entry: object) -> _InternalTargetVolume:
    return _InternalTargetVolume(
        label=getattr(entry, "label", "general"),
        center_scene=getattr(entry, "center_scene", Vec3(x=0.0, y=0.0, z=0.0)),
        half_extents_scene=getattr(
            entry,
            "half_extents_scene",
            Vec3(x=0.0, y=0.0, z=0.0),
        ),
        shape=getattr(entry, "shape", "sphere"),
    )


def _sample_point_in_volume(
    *,
    rng: np.random.Generator,
    shape: str,
    center: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    half_extents = np.abs(_as_array(half_extents))
    if str(shape).lower() == "bbox":
        return center + rng.uniform(low=-half_extents, high=half_extents)

    radius = float(half_extents[0]) if half_extents.size else 0.0
    if radius <= _EPS:
        return center.copy()

    direction = _normalize(rng.normal(size=3))
    distance = radius * float(rng.uniform(0.0, 1.0)) ** (1.0 / 3.0)
    return center + distance * direction


def _compose_with_jitter(
    *,
    base: Quaternion,
    jitter_rad: float,
    rng: np.random.Generator,
) -> Quaternion:
    jitter = float(jitter_rad)
    if jitter <= 0.0 or not np.isfinite(jitter):
        return base

    half = abs(jitter)
    roll = float(rng.uniform(-half, half))
    pitch = float(rng.uniform(-half, half))
    yaw = float(rng.uniform(-half, half))
    delta = _quat_from_euler(roll, pitch, yaw)
    return _quat_to_pose(_quat_multiply(delta, _quat_to_tuple(base)))


def _quat_to_tuple(quaternion: Quaternion) -> tuple[float, float, float, float]:
    return (float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w))


def _quat_to_pose(quat: tuple[float, float, float, float]) -> Quaternion:
    x, y, z, w = quat
    return Quaternion(x=x, y=y, z=z, w=w)


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_to_matrix(quaternion: Quaternion) -> np.ndarray:
    x, y, z, w = _quat_to_tuple(quaternion)
    norm = math.sqrt(max(_EPS, x * x + y * y + z * z + w * w))
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _matrix_to_quat(matrix: np.ndarray) -> Quaternion:
    m = np.asarray(matrix, dtype=float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))


def _normalize(vector: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= _EPS:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return arr / norm


def _as_array(values: np.ndarray | Vec3) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return np.array([float(values.x), float(values.y), float(values.z)], dtype=float)


def _vec_to_array(vector: Vec3) -> np.ndarray:
    return np.array([float(vector.x), float(vector.y), float(vector.z)], dtype=float)


def _vector_to_vec3(values: np.ndarray) -> Vec3:
    return Vec3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _apply_rotation(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float) @ np.asarray(vector, dtype=float)
