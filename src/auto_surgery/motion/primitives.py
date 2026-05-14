# Collapsed motion primitives.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3
from auto_surgery.schemas.results import StepResult


_CONTACT_TOLERANCE_MM = 1.0
_DEFAULT_CONTACT_SEARCH_MM = 12.0
_DEFAULT_CONTACT_SPEED_MM_PER_S = 20.0
_SMALL_STANDOFF_MM = 1.0
_CONTACT_STEP_MIN_DT = 1.0e-3
_EPS = 1.0e-12


class PrimitiveKind(StrEnum):
    """Primitive discriminants for the collapsed motion set."""

    REACH = "reach"
    HOLD = "hold"
    CONTACT_REACH = "contact_reach"
    GRIP = "grip"
    DRAG = "drag"
    BRUSH = "brush"


@dataclass(frozen=True)
class _CommonParams:
    jaw_target_start: float | None
    jaw_target_end: float | None


@dataclass(frozen=True)
class _CommonTimedParams(_CommonParams):
    duration_s: float


@dataclass(frozen=True)
class Reach(_CommonTimedParams):
    target_pose_scene: Pose
    end_on_contact: bool = False


@dataclass(frozen=True)
class Hold(_CommonTimedParams):
    """Explicit stationary intent."""


@dataclass(frozen=True)
class ContactReach(_CommonTimedParams):
    direction_hint_scene: Vec3 | None = None
    max_search_mm: float = _DEFAULT_CONTACT_SEARCH_MM
    peak_speed_mm_per_s: float = _DEFAULT_CONTACT_SPEED_MM_PER_S


@dataclass(frozen=True)
class Grip:
    approach: ContactReach
    lift_distance_mm: float = 10.0
    lift_duration_s: float = 0.5
    release_after_s: float = 0.5
    jaw_close_duration_s: float = 0.3
    duration_s: float = 2.0
    jaw_target_start: float | None = None
    jaw_target_end: float | None = None


@dataclass(frozen=True)
class Drag(_CommonTimedParams):
    direction_hint_scene: Vec3 | None = None
    distance_mm: float = 8.0
    normal_force_target: float = 0.1


@dataclass(frozen=True)
class Brush(_CommonTimedParams):
    amplitude_mm: float = 4.0
    frequency_hz: float = 2.0


Primitive = Reach | Hold | ContactReach | Grip | Drag | Brush


@dataclass(frozen=True)
class PrimitiveOutput:
    twist_camera: Twist
    jaw_target: float
    is_finished: bool


@dataclass(frozen=True)
class ActivePrimitive:
    primitive: Primitive
    started_at_pose_scene: Pose
    started_at_jaw: float
    duration_s: float
    elapsed_s: float
    contact_was_in: bool = False
    in_post_contact_phase: bool = False
    post_contact_started_at_s: float = 0.0
    phase_index: int = 0
    phase_started_at_s: float = 0.0


def evaluate(active: ActivePrimitive, last_step: StepResult) -> PrimitiveOutput:
    return _evaluate(active, last_step)


def _evaluate(active: ActivePrimitive, last_step: StepResult) -> PrimitiveOutput:
    camera_matrix, camera_position = _camera_basis(last_step)
    to_camera = camera_matrix.T
    elapsed_s = max(0.0, float(getattr(active, "elapsed_s", 0.0)))
    duration_s = max(0.0, float(getattr(active, "duration_s", 0.0)))
    tau = _time_to_fraction(elapsed_s, duration_s)
    primitive = getattr(active, "primitive")

    if isinstance(primitive, Reach):
        return PrimitiveOutput(
            twist_camera=_evaluate_reach(
                active=active,
                primitive=primitive,
                camera_matrix=camera_matrix,
                camera_position=camera_position,
                to_camera=to_camera,
                duration_s=duration_s,
                tau=tau,
            ),
            jaw_target=_jaw_target(primitive=primitive, active=active, tau=tau),
            is_finished=_reach_finished(last_step, elapsed_s, duration_s, primitive.end_on_contact),
        )

    if isinstance(primitive, Hold):
        return PrimitiveOutput(
            twist_camera=_zero_twist(),
            jaw_target=_jaw_target(primitive=primitive, active=active, tau=tau),
            is_finished=elapsed_s >= duration_s,
        )

    if isinstance(primitive, ContactReach):
        return _evaluate_contact_reach(
            active=active,
            primitive=primitive,
            last_step=last_step,
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            elapsed_s=elapsed_s,
            duration_s=duration_s,
        )

    if isinstance(primitive, Grip):
        return _evaluate_grip(
            active=active,
            primitive=primitive,
            last_step=last_step,
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            elapsed_s=elapsed_s,
            duration_s=duration_s,
        )

    if isinstance(primitive, Drag):
        return _evaluate_drag(
            active=active,
            primitive=primitive,
            last_step=last_step,
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            elapsed_s=elapsed_s,
            duration_s=duration_s,
            tau=tau,
        )

    if isinstance(primitive, Brush):
        return _evaluate_brush(
            active=active,
            primitive=primitive,
            last_step=last_step,
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            elapsed_s=elapsed_s,
            duration_s=duration_s,
            tau=tau,
        )

    return PrimitiveOutput(
        twist_camera=_zero_twist(),
        jaw_target=_jaw_target(primitive=primitive, active=active, tau=tau),
        is_finished=elapsed_s >= duration_s,
    )


def _evaluate_reach(
    *,
    active: ActivePrimitive,
    primitive: Reach,
    camera_matrix: np.ndarray,
    camera_position: np.ndarray,
    to_camera: np.ndarray,
    duration_s: float,
    tau: float,
) -> Twist:
    if duration_s <= 0.0:
        return _zero_twist()

    start_pose_scene = getattr(active, "started_at_pose_scene", primitive.target_pose_scene)
    delta_position = _vec_to_array(primitive.target_pose_scene.position) - _vec_to_array(start_pose_scene.position)
    delta_rotation = _pose_rotation_delta_vec(start_pose_scene, primitive.target_pose_scene)
    velocity = min_jerk_velocity_scalar(tau=tau, duration_s=duration_s)
    return Twist(
        linear=_vector_to_vec3(_apply_rotation(to_camera, delta_position * velocity)),
        angular=_vector_to_vec3(_apply_rotation(to_camera, delta_rotation * velocity)),
    )


def _evaluate_contact_reach(
    *,
    active: ActivePrimitive,
    primitive: ContactReach,
    last_step: StepResult,
    camera_matrix: np.ndarray,
    camera_position: np.ndarray,
    to_camera: np.ndarray,
    elapsed_s: float,
    duration_s: float,
) -> PrimitiveOutput:
    if duration_s <= 0.0:
        return PrimitiveOutput(
            twist_camera=_zero_twist(),
            jaw_target=_jaw_target(primitive=primitive, active=active, tau=1.0),
            is_finished=True,
        )

    tip_scene = last_step.sensors.tool.pose
    search_dir = _preferred_direction(primitive.direction_hint_scene, tip_scene)
    if np.linalg.norm(search_dir) <= _EPS:
        search_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    search_dir = _axis_unit(search_dir)
    target_point = _build_search_pose(tip_scene, search_dir, primitive.max_search_mm)
    target_pos = _vec_to_array(target_point.position) - search_dir * _SMALL_STANDOFF_MM

    reached = bool(getattr(last_step.sensors.tool, "in_contact", False))
    touch_distance = _contact_distance_hint(last_step)
    if touch_distance is None:
        touch_distance = _vector_norm(_vec_to_array(target_point.position) - _vec_to_array(tip_scene.position))

    remaining = max(duration_s - elapsed_s, 0.0)
    local_dt = max(_CONTACT_STEP_MIN_DT, min(float(getattr(last_step, "dt", _CONTACT_STEP_MIN_DT)), max(remaining, _CONTACT_STEP_MIN_DT)))
    is_finished = reached or remaining <= _CONTACT_STEP_MIN_DT or touch_distance < _CONTACT_TOLERANCE_MM

    return PrimitiveOutput(
        twist_camera=_evaluate_reach(
            active=active,
            primitive=Reach(
                target_pose_scene=Pose(position=_vector_to_vec3(target_pos), rotation=tip_scene.rotation),
                duration_s=local_dt,
                end_on_contact=False,
                jaw_target_start=primitive.jaw_target_start,
                jaw_target_end=primitive.jaw_target_end,
            ),
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            duration_s=local_dt,
            tau=0.5,
        ),
        jaw_target=_jaw_target(
            primitive=primitive,
            active=active,
            tau=_time_to_fraction(elapsed_s, duration_s),
        ),
        is_finished=is_finished,
    )


def _evaluate_grip(
    *,
    active: ActivePrimitive,
    primitive: Grip,
    last_step: StepResult,
    camera_matrix: np.ndarray,
    camera_position: np.ndarray,
    to_camera: np.ndarray,
    elapsed_s: float,
    duration_s: float,
) -> PrimitiveOutput:
    if duration_s <= 0.0:
        return PrimitiveOutput(
            twist_camera=_zero_twist(),
            jaw_target=_jaw_target(primitive=_legacy_grip_primitive(primitive), active=active, tau=1.0),
            is_finished=True,
        )

    phase1 = max(0.0, duration_s - (primitive.jaw_close_duration_s + primitive.lift_duration_s + primitive.release_after_s + primitive.lift_duration_s))
    phase2 = primitive.jaw_close_duration_s
    phase3 = primitive.lift_duration_s
    phase4 = primitive.release_after_s
    phase5 = max(0.0, duration_s - (phase1 + phase2 + phase3 + phase4), 0.0)

    p2 = phase1
    p3 = phase1 + phase2
    p4 = phase1 + phase2 + phase3
    p5 = phase1 + phase2 + phase3 + phase4

    tip_scene = last_step.sensors.tool.pose
    normal = np.array([0.0, 0.0, 1.0], dtype=float)
    jaw_start = active.started_at_jaw if primitive.jaw_target_start is None else primitive.jaw_target_start
    jaw_end = jaw_start if primitive.jaw_target_end is None else primitive.jaw_target_end

    if elapsed_s < p2:
        return _evaluate_contact_reach(
            active=active,
            primitive=primitive.approach,
            last_step=last_step,
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            elapsed_s=elapsed_s,
            duration_s=phase1,
        )

    if elapsed_s < p3:
        close_t = _time_to_fraction(elapsed_s - p2, max(phase2, _CONTACT_STEP_MIN_DT))
        return PrimitiveOutput(
            twist_camera=_zero_twist(),
            jaw_target=_linear_interpolate(jaw_start, 1.0, close_t),
            is_finished=elapsed_s >= p3,
        )

    if elapsed_s < p4:
        lift_t = _time_to_fraction(elapsed_s - p3, max(phase3, _CONTACT_STEP_MIN_DT))
        target_pos = _vec_to_array(tip_scene.position) + normal * primitive.lift_distance_mm * lift_t
        target_pose = Pose(position=_vector_to_vec3(target_pos), rotation=tip_scene.rotation)
        return PrimitiveOutput(
            twist_camera=_evaluate_reach(
                active=active,
                primitive=Reach(
                    target_pose_scene=target_pose,
                    duration_s=max(phase3, _CONTACT_STEP_MIN_DT),
                    end_on_contact=False,
                    jaw_target_start=1.0,
                    jaw_target_end=1.0,
                ),
                camera_matrix=camera_matrix,
                camera_position=camera_position,
                to_camera=to_camera,
                duration_s=max(phase3, _CONTACT_STEP_MIN_DT),
                tau=0.5,
            ),
            jaw_target=1.0,
            is_finished=elapsed_s >= p4,
        )

    if elapsed_s < p5:
        open_t = _time_to_fraction(elapsed_s - p4, max(phase4, _CONTACT_STEP_MIN_DT))
        return PrimitiveOutput(
            twist_camera=_zero_twist(),
            jaw_target=_linear_interpolate(1.0, jaw_end, open_t),
            is_finished=elapsed_s >= p5,
        )

    remaining = max(duration_s - elapsed_s, 0.0)
    target_pose = Pose(position=_vector_to_vec3(_vec_to_array(tip_scene.position) + normal * _SMALL_STANDOFF_MM), rotation=tip_scene.rotation)
    return PrimitiveOutput(
        twist_camera=_evaluate_reach(
            active=active,
            primitive=Reach(
                target_pose_scene=target_pose,
                duration_s=max(remaining, _CONTACT_STEP_MIN_DT),
                end_on_contact=False,
                jaw_target_start=jaw_end,
                jaw_target_end=jaw_end,
            ),
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            duration_s=max(remaining, _CONTACT_STEP_MIN_DT),
            tau=0.5,
        ),
        jaw_target=jaw_end,
        is_finished=remaining <= 0.0,
    )


def _evaluate_drag(
    *,
    active: ActivePrimitive,
    primitive: Drag,
    last_step: StepResult,
    camera_matrix: np.ndarray,
    camera_position: np.ndarray,
    to_camera: np.ndarray,
    elapsed_s: float,
    duration_s: float,
    tau: float,
) -> PrimitiveOutput:
    if duration_s <= 0.0:
        return PrimitiveOutput(_zero_twist(), _jaw_target(primitive=primitive, active=active, tau=tau), True)

    tip_scene = last_step.sensors.tool.pose
    direction = _preferred_direction(primitive.direction_hint_scene, tip_scene)
    normal = np.array([0.0, 0.0, 1.0], dtype=float)
    tangent = direction - normal * float(np.dot(direction, normal))
    if _vector_norm(tangent) <= _EPS:
        tangent = np.array([1.0, 0.0, 0.0], dtype=float)
    tangent = _axis_unit(tangent)

    local_dt = max(_CONTACT_STEP_MIN_DT, float(getattr(last_step, "dt", _CONTACT_STEP_MIN_DT)))
    speed = primitive.distance_mm / max(duration_s, local_dt)
    target_pos = _vec_to_array(tip_scene.position) + tangent * speed * local_dt
    wrench = getattr(last_step.sensors.tool, "wrench", None)
    if wrench is not None:
        normal_force_error = primitive.normal_force_target - float(getattr(wrench, "z", 0.0))
        target_pos = target_pos + normal * 0.05 * normal_force_error * local_dt

    target_pose = Pose(position=_vector_to_vec3(target_pos), rotation=tip_scene.rotation)
    return PrimitiveOutput(
        twist_camera=_evaluate_reach(
            active=active,
            primitive=Reach(
                target_pose_scene=target_pose,
                duration_s=local_dt,
                end_on_contact=False,
                jaw_target_start=primitive.jaw_target_start,
                jaw_target_end=primitive.jaw_target_end,
            ),
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            duration_s=local_dt,
            tau=0.5,
        ),
        jaw_target=_jaw_target(primitive=primitive, active=active, tau=_time_to_fraction(elapsed_s, duration_s)),
        is_finished=elapsed_s >= duration_s,
    )


def _evaluate_brush(
    *,
    active: ActivePrimitive,
    primitive: Brush,
    last_step: StepResult,
    camera_matrix: np.ndarray,
    camera_position: np.ndarray,
    to_camera: np.ndarray,
    elapsed_s: float,
    duration_s: float,
    tau: float,
) -> PrimitiveOutput:
    if duration_s <= 0.0:
        return PrimitiveOutput(_zero_twist(), _jaw_target(primitive=primitive, active=active, tau=tau), True)

    tip_scene = last_step.sensors.tool.pose
    direction = _axis_unit(_preferred_direction(None, tip_scene))
    omega = 2.0 * math.pi * primitive.frequency_hz
    speed = abs(omega * primitive.amplitude_mm * math.cos(omega * elapsed_s))
    local_dt = max(_CONTACT_STEP_MIN_DT, float(getattr(last_step, "dt", _CONTACT_STEP_MIN_DT)))
    target_pos = _vec_to_array(tip_scene.position) + direction * speed * local_dt
    target_pose = Pose(position=_vector_to_vec3(target_pos), rotation=tip_scene.rotation)

    return PrimitiveOutput(
        twist_camera=_evaluate_reach(
            active=active,
            primitive=Reach(
                target_pose_scene=target_pose,
                duration_s=local_dt,
                end_on_contact=False,
                jaw_target_start=primitive.jaw_target_start,
                jaw_target_end=primitive.jaw_target_end,
            ),
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            duration_s=local_dt,
            tau=0.5,
        ),
        jaw_target=_jaw_target(primitive=primitive, active=active, tau=_time_to_fraction(elapsed_s, duration_s)),
        is_finished=elapsed_s >= duration_s,
    )


def _reach_finished(last_step: StepResult, elapsed_s: float, duration_s: float, is_touchable: bool) -> bool:
    if elapsed_s >= duration_s:
        return True
    return bool(getattr(last_step.sensors.tool, "in_contact", False)) if is_touchable else False


def _jaw_target(*, primitive: Primitive, active: ActivePrimitive, tau: float) -> float:
    jaw_start = float(primitive.jaw_target_start) if primitive.jaw_target_start is not None else float(active.started_at_jaw)
    jaw_end = float(primitive.jaw_target_end) if primitive.jaw_target_end is not None else jaw_start
    if jaw_end == jaw_start:
        return jaw_start
    return jaw_start + (jaw_end - jaw_start) * min_jerk_position_scalar(_time_to_fraction(tau, 1.0))


def _contact_distance_hint(last_step: StepResult) -> float | None:
    geometry = getattr(last_step.sensors, "scene_geometry", None)
    if geometry is None:
        return None
    try:
        tip = last_step.sensors.tool.pose.position
        closest = geometry.closest_surface_point(tip)
        if isinstance(closest, dict) and "distance" in closest:
            return float(closest["distance"])
    except Exception:
        pass
    try:
        signed_distance = getattr(geometry, "signed_distance", None)
        if callable(signed_distance):
            return float(signed_distance(last_step.sensors.tool.pose.position))
    except Exception:
        pass
    return None


def _camera_basis(last_step: StepResult) -> tuple[np.ndarray, np.ndarray]:
    cameras = last_step.sensors.cameras
    if not cameras:
        return np.eye(3, dtype=float), np.zeros(3, dtype=float)
    extrinsics = cameras[0].extrinsics
    return _quat_to_matrix(extrinsics.rotation), _vec_to_array(extrinsics.position)


def _build_search_pose(tool_pose: Pose, direction: np.ndarray, max_search_mm: float) -> Pose:
    search_position = _vec_to_array(tool_pose.position) + _axis_unit(direction) * float(max_search_mm)
    return Pose(position=_vector_to_vec3(search_position), rotation=tool_pose.rotation)


def _preferred_direction(direction_hint_scene: Vec3 | None, tip_pose: Pose) -> np.ndarray:
    if direction_hint_scene is not None:
        return _vec_to_array(direction_hint_scene)
    return _quat_to_matrix(tip_pose.rotation)[:, 2]


def _pose_rotation_delta_vec(start: Pose, target: Pose) -> np.ndarray:
    delta = _quat_multiply(_quat_tuple(start.rotation), _quat_inverse(_quat_tuple(target.rotation)))
    return _quat_to_rotvec(delta)


def _quat_to_rotvec(quat: tuple[float, float, float, float]) -> np.ndarray:
    qx, qy, qz, qw = _normalize_quat(quat)
    angle = 2.0 * math.atan2(math.sqrt(qx * qx + qy * qy + qz * qz), qw)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    if abs(angle) <= _EPS:
        return np.zeros(3, dtype=float)
    s = math.sin(angle * 0.5)
    if abs(s) <= _EPS:
        return np.zeros(3, dtype=float)
    axis = np.array([qx, qy, qz], dtype=float) / s
    return axis * angle


def _quat_to_matrix(quat: Quaternion) -> np.ndarray:
    x, y, z, w = _quat_tuple(quat)
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    return np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), ww - xx + yy - zz, 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), ww - xx - yy + zz],
        ],
        dtype=float,
    )


def _matrix_to_quat(matrix: np.ndarray) -> Quaternion:
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
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


def _quat_tuple(value: Quaternion) -> tuple[float, float, float, float]:
    return (float(value.x), float(value.y), float(value.z), float(value.w))


def _normalize_quat(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in quat))
    if norm <= _EPS:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / norm
    x, y, z, w = quat
    return (x * inv, y * inv, z * inv, w * inv)


def _quat_inverse(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = _normalize_quat(quat)
    return (-x, -y, -z, w)


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lx * rw + lw * rx + ly * rz - lz * ry,
        ly * rw + lw * ry + lz * rx - lx * rz,
        lz * rw + lw * rz + lx * ry - ly * rx,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _apply_rotation(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return rotation @ np.asarray(vector, dtype=float)


def _vec_to_array(value: Vec3) -> np.ndarray:
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=float)


def _vector_to_vec3(values: np.ndarray) -> Vec3:
    return Vec3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _axis_unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vector / norm


def _vector_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _time_to_fraction(elapsed_s: float, duration_s: float) -> float:
    if duration_s <= 0.0:
        return 1.0
    return max(0.0, min(elapsed_s / duration_s, 1.0))


def _linear_interpolate(start: float, end: float, ratio: float) -> float:
    t = min(1.0, max(0.0, ratio))
    return float(start + (end - start) * t)


def _zero_twist() -> Twist:
    return Twist(
        linear=Vec3(x=0.0, y=0.0, z=0.0),
        angular=Vec3(x=0.0, y=0.0, z=0.0),
    )


def _legacy_grip_primitive(primitive: Grip) -> Hold:
    return Hold(
        duration_s=max(_CONTACT_STEP_MIN_DT, primitive.lift_duration_s),
        jaw_target_start=primitive.jaw_target_start,
        jaw_target_end=primitive.jaw_target_end,
    )

def min_jerk_position_scalar(tau: float) -> float:
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return 1.0
    value = float(tau)
    return (10.0 * value**3) - (15.0 * value**4) + (6.0 * value**5)


def min_jerk_velocity_scalar(*, tau: float, duration_s: float) -> float:
    if duration_s <= 0.0 or not math.isfinite(float(duration_s)):
        return 0.0
    if tau <= 0.0 or tau >= 1.0:
        return 0.0
    value = float(tau)
    return (30.0 * value**2 - 60.0 * value**3 + 30.0 * value**4) / float(duration_s)