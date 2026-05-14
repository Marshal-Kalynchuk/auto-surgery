"""Scene geometry abstractions used by motion and safety logic."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from auto_surgery.schemas.commands import Vec3
from auto_surgery.schemas.scene import SceneConfig, TargetVolume

AABB = tuple[Vec3, Vec3]


def _import_trimesh():
    """Import trimesh lazily so modules can import without extra runtime setup."""

    try:
        return importlib.import_module("trimesh")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "trimesh is required for scene geometry operations. Install it from the `prep` dependency group."
        ) from exc


def _vec3_to_array(value: Vec3) -> np.ndarray:
    return np.array((float(value.x), float(value.y), float(value.z)), dtype=float)


def _array_to_vec3(values: Sequence[float]) -> Vec3:
    return Vec3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 0.0:
        return np.zeros(3, dtype=float)
    return vector / magnitude


def _normalise_no_nan(vector: np.ndarray) -> np.ndarray:
    normalized = _normalize(vector)
    if not np.isfinite(normalized).all():
        return np.array([0.0, 0.0, 1.0], dtype=float)
    if np.linalg.norm(normalized) <= 0.0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normalized


def _grid_key(point: np.ndarray, resolution_mm: float) -> tuple[int, int, int]:
    scale = 1.0 / float(max(resolution_mm, 1.0e-12))
    return (
        int(round(point[0] * scale)),
        int(round(point[1] * scale)),
        int(round(point[2] * scale)),
    )


@dataclass(frozen=True)
class SurfacePoint:
    """Surface point result with geometry and signed-distance metadata."""

    position: Vec3
    normal: Vec3
    signed_distance: float


@dataclass(frozen=True)
class RayHit:
    """Information about a ray intersection with the scene surface."""

    position: Vec3
    distance: float


class SceneGeometry(Protocol):
    """Abstract scene geometry contract used by motion and safety layers."""

    def closest_surface_point(self, p_scene: Vec3) -> SurfacePoint: ...

    def signed_distance(self, p_scene: Vec3) -> float: ...

    def ray_cast(
        self,
        origin: Vec3,
        direction: Vec3,
        max_distance_mm: float,
    ) -> RayHit | None: ...

    def bounds(self) -> AABB: ...


class MeshSceneGeometry:
    """Mesh-backed geometry using trimesh proximity and ray-casting queries."""

    def __init__(
        self,
        mesh_path: str,
        *,
        sdf_grid_resolution_mm: float | None = 1.0,
    ) -> None:
        self._mesh_path = str(Path(mesh_path).expanduser())
        self._sdf_cache_resolution_mm = sdf_grid_resolution_mm
        self._sdf_cache: dict[tuple[int, int, int], float] = {}

        trimesh = _import_trimesh()
        loaded = trimesh.load(self._mesh_path, process=False, force="mesh")
        if not hasattr(loaded, "faces") or not hasattr(loaded, "vertices"):
            raise ValueError(f"Mesh load did not return a triangulated mesh: {self._mesh_path}")
        if len(loaded.faces) == 0:
            raise ValueError(f"Mesh has no faces: {self._mesh_path}")

        self._mesh = loaded
        self._proximity = trimesh.proximity.ProximityQuery(self._mesh)

    def closest_surface_point(self, p_scene: Vec3) -> SurfacePoint:
        point = _vec3_to_array(p_scene)
        closest_points, distances, face_indices = self._proximity.on_surface([point])
        signed_distance = float(self.signed_distance(p_scene))

        closest_point = np.asarray(closest_points[0], dtype=float)
        if not np.isfinite(closest_point).all():
            closest_point = np.asarray(self._mesh.bounds[0], dtype=float)
            signed_distance = 0.0

        raw_normal = np.array((0.0, 0.0, 1.0), dtype=float)
        if np.size(face_indices):
            face_index = int(face_indices[0])
            if 0 <= face_index < len(self._mesh.faces):
                raw_normal = np.asarray(self._mesh.face_normals[face_index], dtype=float)
        normal = _normalise_no_nan(raw_normal)

        # Keep behavior explicit if query returns unexpected degeneracy.
        if np.isnan(distances[0]):
            closest_point = np.asarray(self._mesh.bounds[0], dtype=float)
            signed_distance = 0.0
            normal = np.array([0.0, 0.0, 1.0], dtype=float)

        return SurfacePoint(
            position=_array_to_vec3(closest_point),
            normal=_array_to_vec3(normal),
            signed_distance=float(signed_distance),
        )

    def signed_distance(self, p_scene: Vec3) -> float:
        point = _vec3_to_array(p_scene)
        if self._sdf_cache_resolution_mm is not None and self._sdf_cache_resolution_mm > 0.0:
            key = _grid_key(point, self._sdf_cache_resolution_mm)
            cached = self._sdf_cache.get(key)
            if cached is not None:
                return cached
        signed_distance = self._compute_signed_distance(point)
        if self._sdf_cache_resolution_mm is not None and self._sdf_cache_resolution_mm > 0.0:
            self._sdf_cache[_grid_key(point, self._sdf_cache_resolution_mm)] = signed_distance
        return signed_distance

    def _compute_signed_distance(self, point: np.ndarray) -> float:
        _, distances, _ = self._proximity.on_surface([point])
        unsigned = float(np.asarray(distances).reshape(()))
        is_inside = self._point_inside(point)
        return -unsigned if is_inside else unsigned

    def _point_inside(self, point: np.ndarray) -> bool:
        try:
            return bool(self._mesh.contains(np.asarray([point], dtype=float))[0])  # type: ignore[call-overload]
        except Exception:
            # Conservative fallback: treat edge/degenerate meshes as exterior.
            return False

    def ray_cast(
        self,
        origin: Vec3,
        direction: Vec3,
        max_distance_mm: float,
    ) -> RayHit | None:
        if max_distance_mm <= 0.0:
            return None

        origin_np = _vec3_to_array(origin)
        direction_np = _vec3_to_array(direction)
        norm = float(np.linalg.norm(direction_np))
        if norm <= 0.0:
            return None
        ray_direction = direction_np / norm

        try:
            locations, _, _ = self._mesh.ray.intersects_location(
                ray_origins=np.array([origin_np]),
                ray_directions=np.array([ray_direction]),
            )
        except Exception:
            return None

        if len(locations) == 0:
            return None

        deltas = np.asarray(locations) - origin_np
        distances = np.linalg.norm(deltas, axis=1)
        mask = distances > 1.0e-12
        if not np.any(mask):
            return None
        distances = distances[mask]
        points = np.asarray(locations)[mask]

        hit_index = int(np.argmin(distances))
        hit_distance = float(distances[hit_index])
        if hit_distance > max_distance_mm:
            return None

        return RayHit(
            position=_array_to_vec3(points[hit_index]),
            distance=hit_distance,
        )

    def bounds(self) -> AABB:
        mesh_bounds = self._mesh.bounds
        minimum = np.asarray(mesh_bounds[0], dtype=float)
        maximum = np.asarray(mesh_bounds[1], dtype=float)
        return _array_to_vec3(minimum), _array_to_vec3(maximum)


class SphereSceneGeometry:
    """Closed-form fallback scene geometry."""

    def __init__(self, center: Vec3, radius: float) -> None:
        if radius <= 0.0:
            raise ValueError("SphereSceneGeometry radius must be positive.")
        self._center = _vec3_to_array(center)
        self._radius = float(radius)

    def closest_surface_point(self, p_scene: Vec3) -> SurfacePoint:
        position = _vec3_to_array(p_scene)
        delta = position - self._center
        distance = float(np.linalg.norm(delta))
        direction = _normalise_no_nan(delta)

        signed_distance = distance - self._radius
        surface_position = self._center + direction * self._radius
        return SurfacePoint(
            position=_array_to_vec3(surface_position),
            normal=_array_to_vec3(direction),
            signed_distance=signed_distance,
        )

    def signed_distance(self, p_scene: Vec3) -> float:
        position = _vec3_to_array(p_scene)
        return float(np.linalg.norm(position - self._center) - self._radius)

    def ray_cast(
        self,
        origin: Vec3,
        direction: Vec3,
        max_distance_mm: float,
    ) -> RayHit | None:
        if max_distance_mm <= 0.0:
            return None

        origin_np = _vec3_to_array(origin)
        direction_np = _vec3_to_array(direction)
        ray_norm = float(np.linalg.norm(direction_np))
        if ray_norm <= 0.0:
            return None
        dir_unit = direction_np / ray_norm

        oc = origin_np - self._center
        b = 2.0 * float(np.dot(oc, dir_unit))
        c = float(np.dot(oc, oc) - self._radius * self._radius)
        discriminant = b * b - 4.0 * c

        if discriminant < 0.0:
            return None

        sqrt_d = float(np.sqrt(discriminant))
        t1 = (-b - sqrt_d) / 2.0
        t2 = (-b + sqrt_d) / 2.0

        candidates: list[float] = []
        if t1 > 1.0e-12:
            candidates.append(t1)
        if t2 > 1.0e-12:
            candidates.append(t2)
        if not candidates:
            return None

        hit_distance = min(candidates)
        if hit_distance > max_distance_mm:
            return None

        return RayHit(
            position=_array_to_vec3(origin_np + dir_unit * hit_distance),
            distance=hit_distance,
        )

    def bounds(self) -> AABB:
        offset = np.array([self._radius, self._radius, self._radius], dtype=float)
        return (
            _array_to_vec3(self._center - offset),
            _array_to_vec3(self._center + offset),
        )


def _target_volume_fallback(scene_config: SceneConfig) -> TargetVolume:
    if scene_config.target_volumes:
        return scene_config.target_volumes[0]
    return TargetVolume(
        label="general",
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        half_extents_scene=Vec3(x=0.001, y=0.001, z=0.001),
    )


def _target_volume_radius(target: TargetVolume) -> float:
    return max(
        float(abs(target.half_extents_scene.x)),
        float(abs(target.half_extents_scene.y)),
        float(abs(target.half_extents_scene.z)),
    )


def _resolve_surface_mesh_path(scene_config: SceneConfig) -> Path | None:
    assets = scene_config.tissue_assets
    if assets is None:
        return None

    rel_path = Path(assets.surface_mesh_relative_path).expanduser()
    candidates = [rel_path]
    if not rel_path.is_absolute():
        candidates = [Path(scene_config.tissue_scene_path).parent / rel_path, rel_path]

    for path in candidates:
        expanded = Path(path).expanduser()
        if expanded.is_file():
            return expanded.resolve()
    return None


def build_scene_geometry(scene_config: SceneConfig) -> SceneGeometry:
    """Build scene-specific geometry with mesh fallback to a target-volume sphere."""

    surface_mesh_path = _resolve_surface_mesh_path(scene_config)
    if surface_mesh_path is not None:
        try:
            return MeshSceneGeometry(str(surface_mesh_path))
        except Exception:
            pass

    target = _target_volume_fallback(scene_config)
    fallback_radius = _target_volume_radius(target)
    radius = fallback_radius if fallback_radius > 0.0 else 0.001
    return SphereSceneGeometry(center=target.center_scene, radius=radius)
