"""Rigid pose helpers for scene/camera observation and pose-servo paths.

Quaternion storage on :class:`~auto_surgery.schemas.commands.Quaternion` is
Hamilton scalar-first ``(w, x, y, z)``. Internal multiplies use ``(x, y, z, w)``
vector-first tuples to match common Hamilton product formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3


@dataclass(frozen=True)
class _TwistFrame:
    _twist: Twist

    def data(self) -> Twist:
        return self._twist


class TwistCamera(_TwistFrame):
    """Velocity expressed in the camera frame at the tool tip (legacy / tests)."""


class TwistSceneTip(_TwistFrame):
    """Velocity expressed at the tool tip in scene coordinates (legacy / tests)."""


class TwistSceneShaft(_TwistFrame):
    """Velocity expressed at the shaft origin in scene coordinates (legacy / tests)."""


def to_scene_tip(twist_camera: TwistCamera, camera_pose: Pose) -> TwistSceneTip:
    """Legacy placeholder: returns camera twist unchanged (tests only)."""

    _ = camera_pose
    return TwistSceneTip(twist_camera.data())


def to_scene_shaft(
    twist_scene_tip: TwistSceneTip,
    dof_pose: Pose,
    tip_offset_local: Vec3,
) -> TwistSceneShaft:
    """Legacy placeholder: returns tip twist unchanged (tests only)."""

    _ = dof_pose
    _ = tip_offset_local
    return TwistSceneShaft(twist_scene_tip.data())


def _quat_to_xyzw(q: Quaternion) -> tuple[float, float, float, float]:
    return (float(q.x), float(q.y), float(q.z), float(q.w))


def _quat_from_xyzw(x: float, y: float, z: float, w: float) -> Quaternion:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-15:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    inv = 1.0 / norm
    return Quaternion(w=w * inv, x=x * inv, y=y * inv, z=z * inv)


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    ax, ay, az, aw = _quat_to_xyzw(a)
    bx, by, bz, bw = _quat_to_xyzw(b)
    return _quat_from_xyzw(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_inverse(q: Quaternion) -> Quaternion:
    x, y, z, w = _quat_to_xyzw(q)
    return _quat_from_xyzw(-x, -y, -z, w)


def quat_rotate_vec3(q: Quaternion, v: Vec3) -> Vec3:
    """Rotate vector ``v`` by unit quaternion ``q``."""

    x, y, z, w = _quat_to_xyzw(q)
    vx, vy, vz = float(v.x), float(v.y), float(v.z)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    cx = vy * tz - vz * ty
    cy = vz * tx - vx * tz
    cz = vx * ty - vy * tx
    return Vec3(
        x=vx + w * tx + cx,
        y=vy + w * ty + cy,
        z=vz + w * tz + cz,
    )


def pose_compose(a: Pose, b: Pose) -> Pose:
    """Compose rigid transforms: ``T_c = T_a * T_b`` (b expressed in a)."""

    rot = quat_multiply(a.rotation, b.rotation)
    t = quat_rotate_vec3(a.rotation, b.position)
    return Pose(
        position=Vec3(
            x=float(a.position.x) + float(t.x),
            y=float(a.position.y) + float(t.y),
            z=float(a.position.z) + float(t.z),
        ),
        rotation=rot,
    )


def pose_inverse(p: Pose) -> Pose:
    q_inv = quat_inverse(p.rotation)
    t_inv = quat_rotate_vec3(q_inv, Vec3(x=-float(p.position.x), y=-float(p.position.y), z=-float(p.position.z)))
    return Pose(position=t_inv, rotation=q_inv)


def pose_scene_to_camera(p_scene: Pose, scene_from_camera: Pose) -> Pose:
    """Express ``p_scene`` in camera frame: ``T_camera_body = T_scene_camera^{-1} * T_scene_body``."""

    inv = pose_inverse(scene_from_camera)
    return pose_compose(inv, p_scene)


def pose_camera_to_scene(p_camera: Pose, scene_from_camera: Pose) -> Pose:
    """Inverse of :func:`pose_scene_to_camera`."""

    return pose_compose(scene_from_camera, p_camera)


def pose_log(p: Pose) -> tuple[Vec3, Vec3]:
    """SE(3) logarithm with small-angle translation ``e_lin ≈ p`` (see plan).

    Returns ``(e_lin_mm, e_ang_rad)`` for the relative transform ``p``.
    """

    x, y, z, w = _quat_to_xyzw(p.rotation)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    v_norm = math.sqrt(x * x + y * y + z * z)
    angle = 2.0 * math.atan2(v_norm, w)
    if angle < 1e-9:
        e_ang = Vec3(x=2.0 * x, y=2.0 * y, z=2.0 * z)
    elif v_norm <= 1e-15:
        e_ang = Vec3(x=0.0, y=0.0, z=0.0)
    else:
        inv = 1.0 / v_norm
        e_ang = Vec3(x=x * inv * angle, y=y * inv * angle, z=z * inv * angle)
    e_lin = Vec3(x=float(p.position.x), y=float(p.position.y), z=float(p.position.z))
    return e_lin, e_ang


def _quat_dot(a: Quaternion, b: Quaternion) -> float:
    ax, ay, az, aw = _quat_to_xyzw(a)
    bx, by, bz, bw = _quat_to_xyzw(b)
    return ax * bx + ay * by + az * bz + aw * bw


def pose_interpolate(start: Pose, end: Pose, s: float) -> Pose:
    """Linear position blend + quaternion slerp. ``s`` clamped to ``[0, 1]``."""

    t = max(0.0, min(1.0, float(s)))
    px = float(start.position.x) * (1.0 - t) + float(end.position.x) * t
    py = float(start.position.y) * (1.0 - t) + float(end.position.y) * t
    pz = float(start.position.z) * (1.0 - t) + float(end.position.z) * t
    if t <= 0.0:
        return start.model_copy(deep=True)
    if t >= 1.0:
        return end.model_copy(deep=True)
    dot = _quat_dot(start.rotation, end.rotation)
    q0 = start.rotation
    if dot < 0.0:
        q1 = Quaternion(w=-float(end.rotation.w), x=-float(end.rotation.x), y=-float(end.rotation.y), z=-float(end.rotation.z))
        dot = -dot
    else:
        q1 = end.rotation
    dot = min(1.0, max(-1.0, _quat_dot(q0, q1)))
    theta = math.acos(dot)
    if theta <= 1e-9:
        return Pose(position=Vec3(x=px, y=py, z=pz), rotation=q0)
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta
    x0, y0, z0, wv0 = _quat_to_xyzw(q0)
    x1, y1, z1, wv1 = _quat_to_xyzw(q1)
    q_rot = _quat_from_xyzw(
        w0 * x0 + w1 * x1,
        w0 * y0 + w1 * y1,
        w0 * z0 + w1 * z1,
        w0 * wv0 + w1 * wv1,
    )
    return Pose(position=Vec3(x=px, y=py, z=pz), rotation=q_rot)
