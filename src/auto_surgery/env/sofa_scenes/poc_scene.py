"""Extract canonical viewpoint and viewport from SOFA POC `.scn` XML."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EPS = 1.0e-12
_TAGS = ("OffscreenCamera", "InteractiveCamera", "Camera")


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _vec3(txt: str, *, name: str) -> tuple[float, float, float]:
    xs = txt.split()
    if len(xs) != 3:
        raise ValueError(f"{name!r} expects three floats, got {txt!r}")
    return (float(xs[0]), float(xs[1]), float(xs[2]))


def parse_poc_scene(path: Path | str) -> "PocSceneViewpoint":
    """Parse POC scene XML; ignores ``${ENV}`` substitutions in unrelated ``filename=`` attrs."""
    p = Path(path)
    try:
        root = ET.parse(p).getroot()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read POC scene file '{p}'.") from exc
    chosen: ET.Element | None = None
    for preferred in _TAGS:
        for node in root.iter():
            if _local_tag(node.tag) == preferred:
                chosen = node
                break
        if chosen is not None:
            break
    if chosen is None:
        raise ValueError(f"No {_TAGS} camera in '{p}'.")
    pos_s, look_s = chosen.get("position"), chosen.get("lookAt") or chosen.get("lookat")
    if not pos_s or not look_s:
        raise ValueError(f"Camera in '{p}' needs position and lookAt.")
    fov_raw = chosen.get("fieldOfView") or chosen.get("field_of_view_deg")
    fov = float(fov_raw) if fov_raw else None

    vw: tuple[int, int] | None = None
    for vs in root.iter():
        if _local_tag(vs.tag) != "ViewerSetting":
            continue
        res = vs.get("resolution")
        if res:
            a, b = res.split()
            vw = (int(a), int(b))
            break
    return PocSceneViewpoint(
        position_scene_mm=_vec3(pos_s.strip(), name="position"),
        lookat_scene_mm=_vec3(look_s.strip(), name="lookAt"),
        viewport=vw,
        field_of_view_deg=fov,
    )


def _ortho_basis(cam: np.ndarray, look_at: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    fwd = look_at.astype(float) - cam.astype(float)
    fn = float(np.linalg.norm(fwd))
    if fn <= _EPS:
        return np.eye(3, dtype=float)
    fwd /= fn
    wu = np.asarray(world_up, dtype=float)
    right = np.cross(wu, -fwd)
    rn = float(np.linalg.norm(right))
    right = (right / rn) if rn > _EPS else np.array([1.0, 0.0, 0.0], dtype=float)
    down = np.cross(fwd, right)
    dn = float(np.linalg.norm(down))
    down = (down / dn) if dn > _EPS else np.array([0.0, -1.0, 0.0], dtype=float)
    return np.column_stack((right, down, fwd))


def _quat_unit(wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = map(float, wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= _EPS:
        raise ValueError("Degenerate quaternion from look-at.")
    return (w / n, x / n, y / n, z / n)


def _quat_from_rot(m: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(m, dtype=float)
    tr = float(a[0, 0] + a[1, 1] + a[2, 2])
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return _quat_unit((0.25 * s, (a[2, 1] - a[1, 2]) / s, (a[0, 2] - a[2, 0]) / s, (a[1, 0] - a[0, 1]) / s))
    if a[0, 0] >= a[1, 1] and a[0, 0] >= a[2, 2]:
        s = math.sqrt(1.0 + a[0, 0] - a[1, 1] - a[2, 2]) * 2.0
        return _quat_unit((((a[2, 1] - a[1, 2]) / s, 0.25 * s, (a[0, 1] + a[1, 0]) / s, (a[0, 2] + a[2, 0]) / s)))
    if a[1, 1] >= a[2, 2]:
        s = math.sqrt(1.0 + a[1, 1] - a[0, 0] - a[2, 2]) * 2.0
        return _quat_unit((((a[0, 2] - a[2, 0]) / s, (a[0, 1] + a[1, 0]) / s, 0.25 * s, (a[1, 2] + a[2, 1]) / s)))
    s = math.sqrt(1.0 + a[2, 2] - a[0, 0] - a[1, 1]) * 2.0
    return _quat_unit((((a[1, 0] - a[0, 1]) / s, (a[0, 2] + a[2, 0]) / s, (a[1, 2] + a[2, 1]) / s, 0.25 * s)))


def camera_pose_scene_from_look_mm(
    position_scene_mm: tuple[float, float, float],
    lookat_scene_mm: tuple[float, float, float],
    *,
    world_up_scene_mm: tuple[float, float, float] = (0.0, 1.0, 0.0),
):
    """SOFA-aligned camera pose (mm); lazy-imports schemas to dodge import cycles."""

    from auto_surgery.schemas.commands import Pose, Quaternion, Vec3

    cam = np.asarray(position_scene_mm, dtype=float)
    lk = np.asarray(lookat_scene_mm, dtype=float)
    wu = np.asarray(world_up_scene_mm, dtype=float)
    qw, qx, qy, qz = _quat_unit(_quat_from_rot(_ortho_basis(cam, lk, wu)))
    return Pose(
        position=Vec3(x=float(cam[0]), y=float(cam[1]), z=float(cam[2])),
        rotation=Quaternion(w=qw, x=qx, y=qy, z=qz),
    )


@dataclass(frozen=True)
class PocSceneViewpoint:
    position_scene_mm: tuple[float, float, float]
    lookat_scene_mm: tuple[float, float, float]
    viewport: tuple[int, int] | None
    field_of_view_deg: float | None


__all__ = ["PocSceneViewpoint", "camera_pose_scene_from_look_mm", "parse_poc_scene"]
