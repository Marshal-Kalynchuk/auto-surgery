"""Pure forceps-math helpers and forceps runtime asset defaults."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml

from auto_surgery.schemas.commands import Pose, Quaternion, Vec3

_PACKAGE_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_FORCEPS_CONTRACT_PATH = _PACKAGE_ROOT / "assets" / "forceps" / "dejavu_default.yaml"
_DEFAULT_FORCEPS_COLLISION_MESH = _PACKAGE_ROOT / "assets" / "forceps" / "shaft_tip_collision.obj"


def _read_payload(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _get_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _coerce_float(value: Any, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(fallback)
    return float(value)


def _coerce_int(value: Any, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return int(fallback)
    return int(value)


def _coerce_float_tuple(value: Any, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return fallback
    if len(value) != len(fallback):
        return fallback
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return fallback


def _coerce_color(value: Any, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)):
        return fallback
    if len(value) != len(fallback):
        return fallback
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return fallback


def _coerce_hex64(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if len(candidate) != 64:
        return None
    if any(c not in "0123456789abcdef" for c in candidate):
        return None
    return candidate


def _resolve_collision_mesh_path(filename: Any) -> Path:
    candidate = _DEFAULT_FORCEPS_COLLISION_MESH
    if isinstance(filename, str):
        candidate = Path(filename).expanduser()
        if not candidate.is_absolute():
            candidate = (_PACKAGE_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
    if candidate.is_file():
        return candidate
    return _DEFAULT_FORCEPS_COLLISION_MESH


def _resolve_assembly_params(payload: Mapping[str, Any]) -> "ForcepsAssemblyParams":
    from auto_surgery.env.sofa_scenes.forceps import ForcepsAssemblyParams

    defaults = ForcepsAssemblyParams()
    assembly = _get_mapping(payload.get("assembly"))
    constants = _get_mapping(assembly.get("constants"))
    stiffness = _get_mapping(constants.get("stiffness"))
    limits = _get_mapping(assembly.get("limits"))
    jaw_limits = _get_mapping(limits.get("jaw"))
    offsets = _get_mapping(assembly.get("offsets"))
    stiffness_visual_body = stiffness.get("visual_body_rgba", constants.get("visual_body_rgba"))
    stiffness_penalty = stiffness.get("collision_penalty", constants.get("collision_penalty"))

    return ForcepsAssemblyParams(
        mass=_coerce_float(constants.get("mass"), fallback=defaults.mass),
        scale=_coerce_float(constants.get("scale"), fallback=defaults.scale),
        body_color=_coerce_color(stiffness_visual_body, fallback=defaults.body_color),
        clasper_color=_coerce_color(stiffness_visual_body, fallback=defaults.clasper_color),
        stiffness=_coerce_float(stiffness_penalty, fallback=defaults.stiffness),
        solver_iterations=_coerce_int(constants.get("solver_iterations"), fallback=defaults.solver_iterations),
        solver_tolerance=_coerce_float(constants.get("solver_tolerance"), fallback=defaults.solver_tolerance),
        hinge_origin_local=_coerce_float_tuple(offsets.get("hinge_origin_local"), fallback=defaults.hinge_origin_local),
        hinge_axis_local=_coerce_float_tuple(offsets.get("hinge_axis_local"), fallback=defaults.hinge_axis_local),
        jaw_open_angle_rad=_coerce_float(jaw_limits.get("open_angle_rad"), fallback=defaults.jaw_open_angle_rad),
        jaw_closed_angle_rad=_coerce_float(jaw_limits.get("closed_angle_rad"), fallback=defaults.jaw_closed_angle_rad),
        tool_tip_offset_local=_coerce_float_tuple(offsets.get("tool_tip_offset_local"), fallback=defaults.tool_tip_offset_local),
        alarm_distance=_coerce_float(constants.get("alarm_distance"), fallback=defaults.alarm_distance),
        contact_distance=_coerce_float(constants.get("contact_distance"), fallback=defaults.contact_distance),
    )


def _resolve_collision_contract(payload: Mapping[str, Any]) -> tuple[str, str]:
    collision = _get_mapping(payload.get("collision_mesh"))
    mesh_path = _resolve_collision_mesh_path(collision.get("filename"))
    declared_hash = _coerce_hex64(collision.get("sha256"))
    if declared_hash is None and mesh_path.is_file():
        declared_hash = sha256(mesh_path.read_bytes()).hexdigest()
    return str(mesh_path), declared_hash or ""


@dataclass(frozen=True)
class ForcepsAssetContract:
    """Validated forceps default constants with collision mesh contract."""

    assembly: "ForcepsAssemblyParams"
    collision_mesh_path: str
    collision_mesh_sha256: str


def load_dejavu_forceps_defaults(*, contract_path: Path | str | None = None) -> ForcepsAssetContract:
    """Load `assets/forceps/dejavu_default.yaml` and coerce invalid values."""

    resolved_path = (
        Path(contract_path).expanduser().resolve()
        if contract_path is not None
        else _DEFAULT_FORCEPS_CONTRACT_PATH
    )
    payload = _read_payload(resolved_path)
    assembly = _resolve_assembly_params(payload)
    collision_mesh_path, collision_mesh_sha256 = _resolve_collision_contract(payload)
    return ForcepsAssetContract(
        assembly=assembly,
        collision_mesh_path=collision_mesh_path,
        collision_mesh_sha256=collision_mesh_sha256,
    )


_CAMERA_TO_SCENE_DEFAULT_SCALE: float = 1.0


@dataclass(frozen=True)
class ClasperVisualTransform:
    """Deterministic per-clasper visual transform."""

    translation: tuple[float, float, float]
    euler_xyz: tuple[float, float, float]


def _normalize_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = q
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def _quat_to_matrix(quaternion: tuple[float, float, float, float]) -> list[list[float]]:
    qx, qy, qz, qw = _normalize_quaternion(quaternion)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    ww = qw * qw
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return [
        [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
    ]


def _mat_mul_vec(matrix: list[list[float]], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    vx, vy, vz = vector
    return (
        matrix[0][0] * vx + matrix[0][1] * vy + matrix[0][2] * vz,
        matrix[1][0] * vx + matrix[1][1] * vy + matrix[1][2] * vz,
        matrix[2][0] * vx + matrix[2][1] * vy + matrix[2][2] * vz,
    )


def shaft_pose_from_tip_target(
    tip_target_scene: Pose,
    shaft_measured_scene: Pose,
    tool_tip_offset_local: tuple[float, float, float],
) -> Pose:
    """Infer shaft scene pose from tip target using measured shaft orientation.

    ``p_tip = R_shaft @ offset_local + p_shaft`` with pure translation offset in
    shaft frame; ``q_tip = q_shaft`` when offset carries no rotation.
    """

    rotation = _quat_to_matrix(
        (
            shaft_measured_scene.rotation.x,
            shaft_measured_scene.rotation.y,
            shaft_measured_scene.rotation.z,
            shaft_measured_scene.rotation.w,
        )
    )
    ox, oy, oz = tool_tip_offset_local
    wx = rotation[0][0] * ox + rotation[0][1] * oy + rotation[0][2] * oz
    wy = rotation[1][0] * ox + rotation[1][1] * oy + rotation[1][2] * oz
    wz = rotation[2][0] * ox + rotation[2][1] * oy + rotation[2][2] * oz
    return Pose(
        position=Vec3(
            x=float(tip_target_scene.position.x) - wx,
            y=float(tip_target_scene.position.y) - wy,
            z=float(tip_target_scene.position.z) - wz,
        ),
        rotation=tip_target_scene.rotation,
    )


def _jaw_angle_from_target(jaw_target: float, *, jaw_open_angle_rad: float = 0.30, jaw_closed_angle_rad: float = 0.0) -> float:
    target = min(1.0, max(0.0, float(jaw_target)))
    return jaw_open_angle_rad + (jaw_closed_angle_rad - jaw_open_angle_rad) * target


def _normalize_axis(axis: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = axis
    norm = (x * x + y * y + z * z) ** 0.5
    if norm <= 0.0:
        return (1.0, 0.0, 0.0)
    return (x / norm, y / norm, z / norm)


def _rotation_matrix_from_axis_angle(
    axis: tuple[float, float, float],
    angle: float,
) -> list[list[float]]:
    x, y, z = _normalize_axis(axis)
    if (x, y, z) == (1.0, 0.0, 0.0):
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        return [
            [1.0, 0.0, 0.0],
            [0.0, cos_angle, -sin_angle],
            [0.0, sin_angle, cos_angle],
        ]
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    one_minus_cos = 1.0 - cos_angle
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    return [
        [one_minus_cos * xx + cos_angle, one_minus_cos * xy - sin_angle * z, one_minus_cos * xz + sin_angle * y],
        [one_minus_cos * xy + sin_angle * z, one_minus_cos * yy + cos_angle, one_minus_cos * yz - sin_angle * x],
        [one_minus_cos * xz - sin_angle * y, one_minus_cos * yz + sin_angle * x, one_minus_cos * zz + cos_angle],
    ]


def _matrix_to_euler_xyz(matrix: list[list[float]]) -> tuple[float, float, float]:
    r00, _, _ = matrix[0]
    r10, r11, r12 = matrix[1]
    r20, r21, r22 = matrix[2]
    sin_pitch = max(-1.0, min(1.0, -r20))
    pitch = math.asin(sin_pitch)
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-12:
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    else:
        roll = math.atan2(-r12, r11)
        yaw = 0.0
    return (roll, pitch, yaw)


def _clasper_visual_transform(
    jaw_target: float,
    side: Literal["left", "right"],
    *,
    jaw_open_angle_rad: float = 0.30,
    jaw_closed_angle_rad: float = 0.0,
    hinge_origin_local: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hinge_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> ClasperVisualTransform:
    """Return a deterministic visual transform for one clasper."""

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    angle = _jaw_angle_from_target(jaw_target, jaw_open_angle_rad=jaw_open_angle_rad, jaw_closed_angle_rad=jaw_closed_angle_rad)
    side_angle = angle if side == "left" else -angle
    rotation = _rotation_matrix_from_axis_angle(hinge_axis_local, side_angle)
    ox, oy, oz = hinge_origin_local
    rotated = (
        rotation[0][0] * ox + rotation[0][1] * oy + rotation[0][2] * oz,
        rotation[1][0] * ox + rotation[1][1] * oy + rotation[1][2] * oz,
        rotation[2][0] * ox + rotation[2][1] * oy + rotation[2][2] * oz,
    )
    translation = (ox - rotated[0], oy - rotated[1], oz - rotated[2])
    euler_xyz = _matrix_to_euler_xyz(rotation)
    return ClasperVisualTransform(translation=translation, euler_xyz=euler_xyz)


DEFAULT_TOOL_TIP_OFFSET_LOCAL: tuple[float, float, float] = (0.0, 0.0, 9.4)
