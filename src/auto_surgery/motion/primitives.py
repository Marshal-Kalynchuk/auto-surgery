"""Motion primitive definitions and per-tick evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from auto_surgery.motion.profile import (
    min_jerk_position_scalar,
    min_jerk_retract_duration,
    min_jerk_velocity_scalar,
)
from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3
from auto_surgery.schemas.results import StepResult


_DEFAULT_PROBE_RETRACT_PEAK_SPEED_M_PER_S = 0.03


class PrimitiveKind(StrEnum):
    """Primitive discriminants for stochastic selection."""

    APPROACH = "approach"
    DWELL = "dwell"
    RETRACT = "retract"
    SWEEP = "sweep"
    ROTATE = "rotate"
    PROBE = "probe"


@dataclass(frozen=True)
class _CommonParams:
    duration_s: float
    jaw_target_start: float | None
    jaw_target_end: float | None


@dataclass(frozen=True)
class Approach(_CommonParams):
    """Translate to a target scene pose."""

    target_pose_scene: Pose
    end_on_contact: bool = True


@dataclass(frozen=True)
class Dwell(_CommonParams):
    """Hold stationary for a fixed duration."""


@dataclass(frozen=True)
class Retract(_CommonParams):
    """Move away from tissue along camera -z in scene frame."""

    distance_m: float


@dataclass(frozen=True)
class Sweep(_CommonParams):
    """Rotate around a camera-frame axis."""

    axis_camera: Vec3
    arc_radians: float


@dataclass(frozen=True)
class Rotate(_CommonParams):
    """Rotate around a fixed camera-frame axis."""

    axis_camera: Vec3
    angle_radians: float


@dataclass(frozen=True)
class Probe(_CommonParams):
    """Approach target, hold on contact, then retract."""

    target_pose_scene: Pose
    hold_after_contact_s: float = 0.3
    retract_distance_m: float = 0.005
    retract_peak_speed_m_per_s: float = _DEFAULT_PROBE_RETRACT_PEAK_SPEED_M_PER_S


Primitive = Approach | Dwell | Retract | Sweep | Rotate | Probe
 

@dataclass(frozen=True)
class PrimitiveOutput:
    """Per-tick motion output from primitive evaluation."""

    twist_camera: Twist
    jaw_target: float
    is_finished: bool


@dataclass(frozen=True)
class ActivePrimitive:
    """Runtime representation fed to :func:`evaluate`."""

    primitive: Primitive
    started_at_pose_scene: Pose
    started_at_jaw: float
    duration_s: float
    elapsed_s: float
    contact_was_in: bool = False
    in_post_contact_phase: bool = False
    post_contact_started_at_s: float = 0.0


def evaluate(active: ActivePrimitive, last_step: StepResult) -> PrimitiveOutput:
    """Evaluate one tick for the currently active primitive."""
    return _evaluate_primitive(active, last_step)


def _evaluate_primitive(active: ActivePrimitive, last_step: StepResult) -> PrimitiveOutput:
    """Internal evaluator with public alias kept for the legacy runtime."""
    camera_matrix, camera_position = _camera_basis(last_step)
    to_camera = camera_matrix.T
    elapsed_s = max(0.0, float(getattr(active, "elapsed_s", 0.0)))
    duration_s = max(0.0, float(getattr(active, "duration_s", 0.0)))
    tau = _time_to_fraction(elapsed_s, duration_s)
    jaw_target = _jaw_target(
        primitive=getattr(active, "primitive", None),
        active=active,
        tau=tau,
    )

    primitive = getattr(active, "primitive")
    if isinstance(primitive, Approach):
        twist = _evaluate_approach(
            active=active,
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            to_camera=to_camera,
            duration_s=duration_s,
            tau=tau,
        )
        is_finished = _approach_finished(active, last_step, elapsed_s, duration_s)
        return PrimitiveOutput(twist_camera=twist, jaw_target=jaw_target, is_finished=is_finished)

    if isinstance(primitive, Dwell):
        is_finished = elapsed_s >= duration_s
        return PrimitiveOutput(twist_camera=_zero_twist(), jaw_target=jaw_target, is_finished=is_finished)

    if isinstance(primitive, Retract):
        twist = _evaluate_retract(
            active=active,
            camera_matrix=camera_matrix,
            to_camera=to_camera,
            duration_s=duration_s,
            tau=tau,
        )
        return PrimitiveOutput(twist_camera=twist, jaw_target=jaw_target, is_finished=elapsed_s >= duration_s)

    if isinstance(primitive, Sweep):
        twist = _evaluate_rotation_about_axis(
            active=active,
            camera_matrix=camera_matrix,
            to_camera=to_camera,
            duration_s=duration_s,
            tau=tau,
            angle_scalar=primitive.arc_radians,
        )
        return PrimitiveOutput(twist_camera=twist, jaw_target=jaw_target, is_finished=elapsed_s >= duration_s)

    if isinstance(primitive, Rotate):
        twist = _evaluate_rotation_about_axis(
            active=active,
            camera_matrix=camera_matrix,
            to_camera=to_camera,
            duration_s=duration_s,
            tau=tau,
            angle_scalar=primitive.angle_radians,
        )
        return PrimitiveOutput(twist_camera=twist, jaw_target=jaw_target, is_finished=elapsed_s >= duration_s)

    if isinstance(primitive, Probe):
        return _evaluate_probe(
            active=active,
            last_step=last_step,
            camera_matrix=camera_matrix,
            to_camera=to_camera,
            tau=tau,
        )

    return PrimitiveOutput(twist_camera=_zero_twist(), jaw_target=jaw_target, is_finished=elapsed_s >= duration_s)


def _evaluate_probe(*, active: ActivePrimitive, last_step: StepResult, camera_matrix: np.ndarray, to_camera: np.ndarray, tau: float) -> PrimitiveOutput:
    primitive = active.primitive
    assert isinstance(primitive, Probe)
    elapsed_s = max(0.0, float(active.elapsed_s))
    duration_s = max(0.0, float(active.duration_s))
    # In the initial phase, perform approach interpolation toward target.
    if not getattr(active, "in_post_contact_phase", False):
        twist = _evaluate_approach(
            active=active,
            camera_matrix=camera_matrix,
            camera_position=_camera_extrinsics_position(last_step),
            to_camera=to_camera,
            duration_s=duration_s,
            tau=tau,
        )
        return PrimitiveOutput(
            twist_camera=twist,
            jaw_target=_jaw_target(
                primitive=primitive,
                active=active,
                tau=tau,
            ),
            is_finished=elapsed_s >= duration_s,
        )

    # Post-contact: hold for hold_after_contact_s, then retract.
    post_started = max(0.0, float(active.post_contact_started_at_s))
    hold_elapsed = elapsed_s - post_started
    if hold_elapsed < primitive.hold_after_contact_s:
        return PrimitiveOutput(
            twist_camera=_zero_twist(),
            jaw_target=_jaw_target(primitive=primitive, active=active, tau=tau),
            is_finished=False,
        )

    retract_elapsed = hold_elapsed - primitive.hold_after_contact_s
    retract_time = _probe_retract_time(
        distance_m=primitive.retract_distance_m,
        peak_retract_speed_m_per_s=float(primitive.retract_peak_speed_m_per_s),
    )
    retract_tau = _time_to_fraction(retract_elapsed, retract_time)
    retract_speed = min_jerk_velocity_scalar(
        tau=retract_tau,
        duration_s=retract_time,
    )
    retract_axis_scene = _apply_rotation(
        camera_matrix,
        _axis_unit(np.array([0.0, 0.0, -1.0], dtype=float)),
    )
    linear_scene = retract_axis_scene * float(primitive.retract_distance_m) * retract_speed
    linear_camera = _apply_rotation(to_camera, linear_scene)
    is_finished = retract_elapsed >= retract_time
    return PrimitiveOutput(
        twist_camera=Twist(
            linear=_vector_to_vec3(linear_camera),
            angular=_ZERO_VEC3,
        ),
        jaw_target=_jaw_target(primitive=primitive, active=active, tau=tau),
        is_finished=is_finished,
    )


def _evaluate_approach(
    *,
    active: ActivePrimitive,
    camera_matrix: np.ndarray,
    camera_position: np.ndarray,
    to_camera: np.ndarray,
    duration_s: float,
    tau: float,
) -> Twist:
    # Probe reuses the approach kinematics during its pre-contact phase, so the
    # active primitive may be either Approach or Probe; both expose
    # ``target_pose_scene``.
    approach = active.primitive
    if not isinstance(approach, (Approach, Probe)):
        raise TypeError(
            f"_evaluate_approach requires Approach or Probe, got {type(approach).__name__}"
        )
    if duration_s <= 0.0:
        return _zero_twist()

    velocity = min_jerk_velocity_scalar(tau=tau, duration_s=duration_s)
    start_pose_scene = _camera_tool_to_scene(
        tool_pose=getattr(active, "started_at_pose_scene", approach.target_pose_scene),
        camera_matrix=camera_matrix,
        camera_position=camera_position,
    )
    delta_position = _vec_to_array(approach.target_pose_scene.position) - _vec_to_array(start_pose_scene.position)
    delta_rotation = _pose_rotation_delta_vec(start_pose_scene, approach.target_pose_scene)
    linear_scene = delta_position * velocity
    angular_scene = delta_rotation * velocity
    return Twist(
        linear=_vector_to_vec3(_apply_rotation(to_camera, linear_scene)),
        angular=_vector_to_vec3(_apply_rotation(to_camera, angular_scene)),
    )


def _evaluate_retract(
    *,
    active: ActivePrimitive,
    camera_matrix: np.ndarray,
    to_camera: np.ndarray,
    duration_s: float,
    tau: float,
) -> Twist:
    retract = active.primitive
    assert isinstance(retract, Retract)
    if duration_s <= 0.0:
        return _zero_twist()

    velocity = min_jerk_velocity_scalar(tau=tau, duration_s=duration_s)
    axis_scene = _apply_rotation(
        camera_matrix,
        _axis_unit(np.array([0.0, 0.0, -1.0], dtype=float)),
    )
    linear_scene = axis_scene * float(retract.distance_m) * velocity
    return Twist(
        linear=_vector_to_vec3(_apply_rotation(to_camera, linear_scene)),
        angular=_ZERO_VEC3,
    )


def _evaluate_rotation_about_axis(
    *,
    active: ActivePrimitive,
    camera_matrix: np.ndarray,
    to_camera: np.ndarray,
    duration_s: float,
    tau: float,
    angle_scalar: float,
) -> Twist:
    primitive = active.primitive
    if duration_s <= 0.0:
        return _zero_twist()
    axis_camera = _vec_to_array(primitive.axis_camera if isinstance(primitive, (Sweep, Rotate)) else np.array([0.0, 0.0, 1.0], dtype=float))
    axis_scene = _apply_rotation(camera_matrix, _axis_unit(axis_camera))
    angular_speed = float(angle_scalar) * min_jerk_velocity_scalar(
        tau=tau,
        duration_s=duration_s,
    )
    angular_scene = axis_scene * angular_speed
    return Twist(
        linear=_ZERO_VEC3,
        angular=_vector_to_vec3(_apply_rotation(to_camera, angular_scene)),
    )


def _approach_finished(active: ActivePrimitive, last_step: StepResult, elapsed_s: float, duration_s: float) -> bool:
    approach = active.primitive
    if not isinstance(approach, Approach):
        return elapsed_s >= duration_s
    if not approach.end_on_contact:
        return elapsed_s >= duration_s
    in_contact = bool(last_step.sensors.tool.in_contact)
    return in_contact or elapsed_s >= duration_s


def _jaw_target(*, primitive: Primitive, active: ActivePrimitive, tau: float) -> float:
    jaw_start = float(primitive.jaw_target_start) if primitive.jaw_target_start is not None else float(active.started_at_jaw)
    jaw_end = float(primitive.jaw_target_end) if primitive.jaw_target_end is not None else jaw_start
    if jaw_end == jaw_start:
        return jaw_start
    return jaw_start + (jaw_end - jaw_start) * min_jerk_position_scalar(tau)


def _probe_retract_time(
    distance_m: float,
    peak_retract_speed_m_per_s: float = _DEFAULT_PROBE_RETRACT_PEAK_SPEED_M_PER_S,
) -> float:
    return min_jerk_retract_duration(
        distance_m=distance_m,
        peak_retract_speed_m_per_s=peak_retract_speed_m_per_s,
    )


def _camera_basis(last_step: StepResult) -> tuple[np.ndarray, np.ndarray]:
    cameras = last_step.sensors.cameras
    if not cameras:
        return np.eye(3, dtype=float), np.zeros(3, dtype=float)
    extrinsics = cameras[0].extrinsics
    return _quat_to_matrix(extrinsics.rotation), _vec_to_array(extrinsics.position)


def _camera_tool_to_scene(*, tool_pose: Pose, camera_matrix: np.ndarray, camera_position: np.ndarray) -> Pose:
    pose_array = _vec_to_array(tool_pose.position)
    rotation = _quat_to_matrix(tool_pose.rotation)
    pos_scene = camera_matrix @ pose_array + camera_position
    rot_scene = _matrix_to_quat(camera_matrix @ rotation)
    return Pose(position=_vector_to_vec3(pos_scene), rotation=rot_scene)


def _pose_rotation_delta_vec(start: Pose, target: Pose) -> np.ndarray:
    delta = _quat_multiply(_quat_tuple(target.rotation), _quat_inverse(_quat_tuple(start.rotation)))
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


def _quat_tuple(value: Quaternion) -> tuple[float, float, float, float]:
    return (float(value.x), float(value.y), float(value.z), float(value.w))


def _normalize_quat(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= _EPS:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / norm
    return (x * inv, y * inv, z * inv, w * inv)


def _quat_inverse(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = quat
    return (-x, -y, -z, w)


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
    x, y, z, w = _normalize_quat(_quat_tuple(quaternion))
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


def _axis_unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vector / norm


def _apply_rotation(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float) @ np.asarray(vector, dtype=float)


def _time_to_fraction(elapsed_s: float, duration_s: float) -> float:
    if duration_s <= 0.0:
        return 1.0
    return max(0.0, min(elapsed_s / duration_s, 1.0))


def _camera_extrinsics_position(last_step: StepResult) -> np.ndarray:
    if not last_step.sensors.cameras:
        return np.zeros(3, dtype=float)
    return _vec_to_array(last_step.sensors.cameras[0].extrinsics.position)


def _vec_to_array(value: Vec3) -> np.ndarray:
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=float)


def _vector_to_vec3(values: np.ndarray) -> Vec3:
    return Vec3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


_ZERO_VEC3 = Vec3(x=0.0, y=0.0, z=0.0)
_EPS = 1.0e-12


def _zero_twist() -> Twist:
    return Twist(linear=_ZERO_VEC3, angular=_ZERO_VEC3)


# Public compatibility symbol used by the legacy generator.
_evaluate = evaluate

