"""Primitive sequencing and stochastic parameter sampling for motion generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from auto_surgery.env.scene_geometry import SceneGeometry, build_scene_geometry
from auto_surgery.motion.profile import min_jerk_retract_duration
from auto_surgery.motion.primitives import (
    Reach,
    Hold,
    ContactReach,
    Grip,
    Drag,
    Brush,
    Primitive,
    PrimitiveKind,
)
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.scene import OrientationBias, WorkspaceEnvelope
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
    """Sample primitives lazily and deterministically per episode.

    Determinism is achieved by seeding a shared `SeedSequence` from
    ``motion_config.seed`` and splitting it into independent, named substreams:
    ``_rng_targets``, ``_rng_noise``, and ``_rng_blend``.

    Reusing the same seed recreates the same trajectory sequence, because each
    substream is consumed in the same order by the same code path.
    """

    motion_config: MotionGeneratorConfig
    scene_config: object
    scene_geometry: SceneGeometry | None = None
    workspace_envelope: WorkspaceEnvelope | None = None

    def __post_init__(self) -> None:
        tool_spec = getattr(self.scene_config, "tool", None)
        envelope = self.workspace_envelope
        if envelope is None and tool_spec is not None:
            envelope = getattr(tool_spec, "workspace_envelope", None)

        geometry = self.scene_geometry
        if geometry is None:
            geometry = _build_geometry_if_available(self.scene_config)

        object.__setattr__(self, "_workspace_envelope", envelope)
        object.__setattr__(self, "_scene_geometry", geometry)
        object.__setattr__(
            self,
            "_orientation_bias",
            getattr(tool_spec, "orientation_bias", OrientationBias()),
        )
        object.__setattr__(self, "_rng_seed", int(self.motion_config.seed))
        self._reset_rngs(int(self.motion_config.seed))
        object.__setattr__(self, "_planned_count", 0)
        object.__setattr__(self, "_issued", 0)
        object.__setattr__(self, "_initial_step", None)

    def _reset_rngs(self, seed: int) -> None:
        """Reset named RNG streams from one shared deterministic seed."""
        seed_seq = np.random.SeedSequence(int(seed))
        branches = seed_seq.spawn(4)
        object.__setattr__(self, "_rng", np.random.default_rng(branches[0]))
        object.__setattr__(self, "_rng_targets", np.random.default_rng(branches[1]))
        object.__setattr__(self, "_rng_noise", np.random.default_rng(branches[2]))
        object.__setattr__(self, "_rng_blend", np.random.default_rng(branches[3]))

    def _sample_weight(self, primary: str, fallback: str | None = None) -> float:
        primary_name = f"weight_{primary}"
        if hasattr(self.motion_config, primary_name):
            return float(getattr(self.motion_config, primary_name))
        if fallback is None:
            return 0.0
        fallback_name = f"weight_{fallback}"
        if hasattr(self.motion_config, fallback_name):
            return float(getattr(self.motion_config, fallback_name))
        return 0.0

    def _has_weight(self, name: str) -> bool:
        return hasattr(self.motion_config, f"weight_{name}")

    def reset(self, initial_step: StepResult) -> None:
        """Reset the generator state for a new episode."""
        self._reset_rngs(int(self._rng_seed))
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
                    primitive = self._build_reach(last_step)
                case PrimitiveKind.DWELL:
                    primitive = self._build_hold(last_step)
                case PrimitiveKind.RETRACT:
                    primitive = self._build_drag()
                case PrimitiveKind.SWEEP:
                    primitive = self._build_brush(last_step)
                case PrimitiveKind.ROTATE:
                    primitive = self._build_grip(last_step)
                case PrimitiveKind.PROBE:
                    if self._has_weight("contact_reach"):
                        primitive = self._build_contact_reach(last_step)
                    else:
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
            self._sample_weight("reach", "approach"),
            self._sample_weight("hold", "dwell"),
            self._sample_weight("drag", "retract"),
            self._sample_weight("brush", "sweep"),
            self._sample_weight("grip", "rotate"),
            self._sample_weight("contact_reach", "probe"),
        )
        total = sum(max(0.0, weight) for weight in weights)
        if total <= 0.0:
            return PrimitiveKind.APPROACH

        needle = float(self._rng_blend.uniform(0.0, total))
        cumulative = 0.0
        for candidate, weight in zip(candidates, weights):
            cumulative += max(0.0, float(weight))
            if needle <= cumulative:
                return candidate
        return candidates[-1]

    def _build_reach(self, last_step: StepResult) -> Reach:
        if last_step is None:
            raise ValueError("last_step is required for reach sampling")

        start_pose_scene = self._tool_pose_scene(last_step)
        target_pose = self._sample_target_pose(last_step, start_pose_scene)
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Reach(
            target_pose_scene=target_pose,
            duration_s=self._sample_range(self.motion_config.approach_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
            end_on_contact=True,
        )

    def _build_hold(self, last_step: StepResult | None = None) -> Hold:
        if last_step is None and self._initial_step is not None:
            last_step = self._initial_step
        if last_step is not None:
            start_pose_scene = self._tool_pose_scene(last_step)
            _ = self._sample_target_pose(last_step, start_pose_scene)
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Hold(
            duration_s=self._sample_range(self.motion_config.dwell_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _build_contact_reach(self, last_step: StepResult) -> ContactReach:
        if last_step is None:
            raise ValueError("last_step is required for contact reach sampling")
        jaw_start, jaw_end = self._sample_jaw_targets()
        return ContactReach(
            direction_hint_scene=None,
            max_search_m=0.1,
            peak_speed_m_per_s=0.05,
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _build_grip(self, last_step: StepResult) -> Grip:
        if last_step is None:
            raise ValueError("last_step is required for grip sampling")

        start_pose_scene = self._tool_pose_scene(last_step)
        _, surface_axis = self._sample_target_position_and_surface(last_step, start_pose_scene)
        jaw_start, jaw_end = self._sample_jaw_targets()
        
        approach = ContactReach(
            direction_hint_scene=_vector_to_vec3(_normalize(surface_axis)),
            max_search_m=0.1,
            peak_speed_m_per_s=0.05,
            jaw_target_start=jaw_start,
            jaw_target_end=None,
        )
        
        return Grip(
            approach=approach,
            lift_distance_m=0.01,
            lift_duration_s=0.5,
            release_after_s=0.5,
            jaw_close_duration_s=0.3,
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _build_drag(self) -> Drag:
        if self._initial_step is not None:
            initial_pose = self._tool_pose_scene(self._initial_step)
            _ = self._sample_target_position_and_surface(
                last_step=self._initial_step,
                start_pose_scene=initial_pose,
            )
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Drag(
            direction_hint_scene=None,
            distance_m=self._sample_range(self.motion_config.retract_distance_range_m),
            normal_force_target=0.1,
            duration_s=self._sample_range(self.motion_config.retract_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _build_brush(self, last_step: StepResult) -> Brush:
        if last_step is None:
            raise ValueError("last_step is required for brush sampling")

        start_pose_scene = self._tool_pose_scene(last_step)
        _, surface_axis = self._sample_target_position_and_surface(last_step, start_pose_scene)
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Brush(
            amplitude_m=0.005,
            frequency_hz=2.0,
            duration_s=self._sample_range(self.motion_config.sweep_duration_range_s),
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
        )

    def _build_first_approach(self, last_step: StepResult) -> Reach:
        volumes = self._sample_target_volumes(last_step, fallback_to_current=True)
        if volumes:
            target_center = volumes[0].center_scene
        elif last_step is not None:
            target_center = self._tool_pose_scene(last_step).position
        else:
            target_center = Vec3(x=0.0, y=0.0, z=0.0)
        start_pose_scene = self._tool_pose_scene(last_step)
        jaw_start, jaw_end = self._sample_jaw_targets()

        return Reach(
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

    def _build_approach(self, last_step: StepResult) -> Reach:
        return self._build_reach(last_step)

    def _build_dwell(self) -> Hold:
        return self._build_hold()

    def _build_retract(self) -> Drag:
        return self._build_drag()

    def _build_sweep(self) -> Brush:
        if self._initial_step is None:
            raise ValueError("initial_step is required for legacy sweep sampling")
        return self._build_brush(self._initial_step)

    def _build_rotate(self) -> Grip:
        if self._initial_step is None:
            raise ValueError("initial_step is required for legacy rotate sampling")
        return self._build_grip(self._initial_step)

    def _build_probe(self, last_step: StepResult) -> Reach:
        start_pose_scene = self._tool_pose_scene(last_step)
        target_pose = self._sample_target_pose(last_step, start_pose_scene)
        approach_time = self._sample_range(self.motion_config.probe_duration_range_s)
        jaw_start, jaw_end = self._sample_jaw_targets()
        return Reach(
            target_pose_scene=target_pose,
            duration_s=approach_time,
            jaw_target_start=jaw_start,
            jaw_target_end=jaw_end,
            end_on_contact=False,
        )

    def _sample_jaw_targets(self) -> tuple[float | None, float | None]:
        if self._rng.random() >= float(self.motion_config.jaw_change_probability):
            return None, None

        low, high = self.motion_config.jaw_value_range
        if float(low) == float(high):
            return None, float(low)

        return None, float(self._rng.uniform(min(low, high), max(low, high)))

    def _sample_target_pose(self, last_step: StepResult, start_pose_scene: Pose) -> Pose:
        target_position, surface_normal = self._sample_target_position_and_surface(
            last_step=last_step,
            start_pose_scene=start_pose_scene,
        )
        start_forward = _apply_rotation(_quat_to_matrix(start_pose_scene.rotation), _vec_to_array(self._orientation_bias.forward_axis_local))
        scene_center_dir = self._scene_center_direction(
            target_position=target_position,
            last_step=last_step,
            start_pose_scene=start_pose_scene,
        )
        blend = float(self._orientation_bias.surface_normal_blend)
        desired_forward = _slerp_direction(
            start=scene_center_dir,
            end=surface_normal,
            blend=blend,
        )
        base_to_desired = _quat_from_vector_to_vector(start_forward, desired_forward)
        base_rotation = _quat_multiply(base_to_desired, _quat_to_tuple(start_pose_scene.rotation))
        rotation = _add_jitter_about_axis(
            quaternion=base_rotation,
            axis=desired_forward,
            jitter_rad=float(self.motion_config.target_orientation_jitter_rad),
            rng=self._rng_noise,
        )
        return Pose(position=_vector_to_vec3(target_position), rotation=_quat_to_pose(rotation))

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
        return volumes[int(self._rng_targets.integers(0, len(volumes)))]

    def _sample_target_position_and_surface(
        self,
        last_step: StepResult,
        start_pose_scene: Pose,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = self._sample_target_position(last_step)
        start_position = _vec_to_array(start_pose_scene.position)
        if np.linalg.norm(position - start_position) <= _EPS:
            position = start_position
        if self._scene_geometry is None:
            target_direction = _normalize(position - start_position)
            return _safe_vector(position), _safe_vector(target_direction)

        try:
            surface = self._scene_geometry.closest_surface_point(_vector_to_vec3(position))
            return _vec_to_array(position), _vec_to_array(surface.normal)
        except Exception:
            target_direction = _normalize(position - start_position)
            return _safe_vector(position), _safe_vector(target_direction)

    def _sample_target_position(self, last_step: StepResult) -> np.ndarray:
        volumes = self._sample_target_volumes(last_step, fallback_to_current=True)
        if not volumes:
            if last_step is not None:
                return _safe_vector(_vec_to_array(last_step.sensors.tool.pose.position))
            if self._initial_step is not None:
                return _safe_vector(_vec_to_array(self._initial_step.sensors.tool.pose.position))
            return _safe_vector(np.zeros(3, dtype=float))

        fallback_position = _vec_to_array(volumes[0].center_scene)
        for _ in range(64):
            volume = volumes[int(self._rng_targets.integers(0, len(volumes)))]
            point = _sample_point_in_volume(
                rng=self._rng_targets,
                shape=volume.shape,
                center=_vec_to_array(volume.center_scene),
                half_extents=_vec_to_array(volume.half_extents_scene),
            )
            if self._is_inside_inner_envelope(point):
                return _safe_vector(point)
        return _safe_vector(fallback_position)

    def _is_inside_inner_envelope(self, point: np.ndarray) -> bool:
        envelope = self._workspace_envelope
        if envelope is None:
            return True
        inner_margin = float(getattr(envelope, "inner_margin_m", 0.0))
        if inner_margin <= 0.0:
            return True
        distance = float(envelope.signed_distance_to_envelope(_vector_to_vec3(point)))
        return distance >= inner_margin

    def _scene_center_direction(
        self,
        *,
        target_position: np.ndarray,
        last_step: StepResult,
        start_pose_scene: Pose,
    ) -> np.ndarray:
        scene_center = self._scene_center(last_step=last_step, start_pose_scene=start_pose_scene)
        direction = scene_center - target_position
        if float(np.linalg.norm(direction)) <= _EPS:
            direction = _vec_to_array(start_pose_scene.position) - target_position
        return _normalize(direction)

    def _scene_center(self, last_step: StepResult, start_pose_scene: Pose) -> np.ndarray:
        geometry = self._scene_geometry
        if geometry is not None:
            try:
                minimum, maximum = geometry.bounds()
                return 0.5 * (_vec_to_array(minimum) + _vec_to_array(maximum))
            except Exception:
                pass
        if last_step is None:
            last_step = self._initial_step
        if last_step is not None:
            return _vec_to_array(self._tool_pose_scene(last_step).position)
        return _vec_to_array(start_pose_scene.position)

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


def _build_geometry_if_available(scene_config: object) -> SceneGeometry | None:
    if scene_config is None:
        return None
    geometry = getattr(scene_config, "scene_geometry", None)
    if geometry is not None:
        return geometry
    try:
        return build_scene_geometry(scene_config)
    except Exception:
        return None


def _safe_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        arr = arr.reshape(-1)
    if arr.size != 3:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return arr.astype(float)


def _quat_from_vector_to_vector(
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[float, float, float, float]:
    start_n = _normalize(start)
    end_n = _normalize(end)
    dot = float(np.clip(np.dot(start_n, end_n), -1.0, 1.0))
    if dot > 1.0 - _EPS:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -1.0 + _EPS:
        axis = _normalize(np.cross(start_n, np.array([1.0, 0.0, 0.0], dtype=float)))
        if np.linalg.norm(axis) <= _EPS:
            axis = _normalize(np.cross(start_n, np.array([0.0, 1.0, 0.0], dtype=float)))
        return tuple(np.asarray([*axis * math.sin(math.pi / 2.0), math.cos(math.pi / 2.0)]))
    axis = np.cross(start_n, end_n)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= _EPS:
        return (0.0, 0.0, 0.0, 1.0)
    axis = axis / axis_norm
    angle = math.acos(dot)
    half = angle / 2.0
    sin_half = math.sin(half)
    return (float(axis[0] * sin_half), float(axis[1] * sin_half), float(axis[2] * sin_half), float(math.cos(half)))


def _slerp_direction(start: np.ndarray, end: np.ndarray, blend: float) -> np.ndarray:
    start_n = _normalize(start)
    end_n = _normalize(end)
    if blend <= 0.0:
        return start_n
    if blend >= 1.0:
        return end_n

    dot = float(np.clip(np.dot(start_n, end_n), -1.0, 1.0))
    if dot > 1.0 - _EPS:
        return start_n
    if dot < -1.0 + _EPS:
        # choose arbitrary axis for antipodal vectors
        axis = _normalize(_safe_vector(np.cross(start_n, np.array([0.0, 0.0, 1.0], dtype=float))))
        if np.linalg.norm(axis) <= _EPS:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        angle = math.pi * blend
        return _normalize(start_n * math.cos(angle) + axis * math.sin(angle))

    omega = math.acos(dot)
    sin_omega = math.sin(omega)
    if sin_omega <= _EPS:
        return start_n
    t0 = math.sin((1.0 - blend) * omega) / sin_omega
    t1 = math.sin(blend * omega) / sin_omega
    return _normalize(t0 * start_n + t1 * end_n)


def _add_jitter_about_axis(
    quaternion: tuple[float, float, float, float],
    axis: np.ndarray,
    jitter_rad: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    jitter = float(jitter_rad)
    if jitter <= 0.0 or not np.isfinite(jitter):
        return quaternion
    angle = float(rng.uniform(-abs(jitter), abs(jitter)))
    unit_axis = _normalize(axis)
    half = angle * 0.5
    sx, sy, sz = unit_axis * math.sin(half)
    cw = math.cos(half)
    twist = (float(sx), float(sy), float(sz), float(cw))
    return _quat_multiply(twist, quaternion)


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
