"""Pure sampling kernel for per-episode scene and motion randomization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.motion import MotionGeneratorConfig
from auto_surgery.schemas.randomization import (
    CameraRandomization,
    EpisodeRandomizationConfig,
    EpisodeSpec,
    LightingRandomization,
    ToneAugmentationRandomization,
    MeshPerturbationRandomization,
    SampleRecord,
    TissueMaterialRandomization,
    TissueTopologyRandomization,
    VisualTintRandomization,
)
from auto_surgery.schemas.scene import (
    BulgeSpec,
    DirectionalLight,
    LightingSpec,
    MeshPerturbation,
    VisualToneAugmentation,
    SceneConfig,
    SpotLight,
    TargetVolume,
    TissueMaterial,
    TissueTopology,
    VisualOverrides,
)
from auto_surgery.schemas.sensors import CameraIntrinsics


_EPS = 1.0e-12

WORLD_UP_SCENE = Vec3(x=0.0, y=1.0, z=0.0)

_AXIS_NAMES = (
    "tissue_material",
    "tissue_topology",
    "tissue_mesh",
    "camera",
    "lighting",
    "visual_tint",
    "tone_augmentation",
    "motion",
)

_CAMERA_ATTEMPTS = 8
_CAMERA_MIN_DISTANCE = 1.0e-2
_CAMERA_DISTANCE_SCALE = (0.25, 4.0)


def _named_subrng(episode_seed: int, name: str) -> np.random.Generator:
    """Create an axis-specific sub-RNG that is stable under axis insertion.

    Parameters
    ----------
    episode_seed:
        Master episode seed.
    name:
        Axis name (must match _AXIS_NAMES).
    """
    if not isinstance(name, str):
        raise TypeError("Axis name must be a string.")
    return np.random.default_rng(
        np.random.SeedSequence(
            entropy=int(episode_seed),
            spawn_key=tuple(ord(character) for character in name),
        )
    )


def _as_array(values: Vec3 | tuple[float, float, float] | list[float]) -> np.ndarray:
    if isinstance(values, Vec3):
        return np.array([float(values.x), float(values.y), float(values.z)], dtype=float)
    return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)


def _to_tuple(values: np.ndarray) -> tuple[float, float, float]:
    if values.shape != (3,):
        raise ValueError("Expected a 3D vector.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _normalize(vector: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= _EPS:
        if fallback is not None:
            return np.array(fallback, dtype=float)
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vector / magnitude


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


def _clamp_vec3(vector: Vec3, lower: float, upper: float) -> Vec3:
    return Vec3(
        x=_clamp(vector.x, lower, upper),
        y=_clamp(vector.y, lower, upper),
        z=_clamp(vector.z, lower, upper),
    )


def _vector_to_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = axis
    cos_theta = math.cos(angle)
    one_minus_cos = 1.0 - cos_theta
    sin_theta = math.sin(angle)
    return np.array(
        [
            [
                cos_theta + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_theta,
                x * z * one_minus_cos + y * sin_theta,
            ],
            [
                y * x * one_minus_cos + z * sin_theta,
                cos_theta + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_theta,
            ],
            [
                z * x * one_minus_cos - y * sin_theta,
                z * y * one_minus_cos + x * sin_theta,
                cos_theta + z * z * one_minus_cos,
            ],
        ],
        dtype=float,
    )
    

def _rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return _vector_to_rotation_matrix(axis, angle) @ np.asarray(vector, dtype=float)


def _matrix_to_quaternion(matrix: np.ndarray) -> Quaternion:
    m = np.asarray(matrix, dtype=float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return Quaternion(
            w=0.25 * s,
            x=(m[2, 1] - m[1, 2]) / s,
            y=(m[0, 2] - m[2, 0]) / s,
            z=(m[1, 0] - m[0, 1]) / s,
        )
    if m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return Quaternion(
            w=(m[2, 1] - m[1, 2]) / s,
            x=0.25 * s,
            y=(m[0, 1] + m[1, 0]) / s,
            z=(m[0, 2] + m[2, 0]) / s,
        )
    if m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return Quaternion(
            w=(m[0, 2] - m[2, 0]) / s,
            x=(m[0, 1] + m[1, 0]) / s,
            y=0.25 * s,
            z=(m[1, 2] + m[2, 1]) / s,
        )
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return Quaternion(
        w=(m[1, 0] - m[0, 1]) / s,
        x=(m[0, 2] + m[2, 0]) / s,
        y=(m[1, 2] + m[2, 1]) / s,
        z=0.25 * s,
    )


def _look_at_rotation_matrix(
    camera_position: np.ndarray,
    look_at: np.ndarray,
    world_up: np.ndarray,
) -> np.ndarray:
    forward = _as_array(Vec3(x=look_at[0], y=look_at[1], z=look_at[2])) - _as_array(
        Vec3(x=camera_position[0], y=camera_position[1], z=camera_position[2])
    )
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= _EPS:
        return np.eye(3, dtype=float)
    forward = forward / forward_norm
    right = np.cross(world_up, -forward)
    right = _normalize(
        right,
        fallback=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    down = _normalize(np.cross(forward, right), fallback=np.array([0.0, -1.0, 0.0], dtype=float))
    return np.column_stack((right, down, forward))


def _apply_roll(rotation: np.ndarray, roll_rad: float) -> np.ndarray:
    if abs(float(roll_rad)) <= _EPS:
        return rotation
    cos_r = math.cos(float(roll_rad))
    sin_r = math.sin(float(roll_rad))
    roll_matrix = np.array(
        [
            [cos_r, -sin_r, 0.0],
            [sin_r, cos_r, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rotation @ roll_matrix


def _sample_range_like(
    distribution: Any,
    rng: np.random.Generator,
    default: float,
) -> float:
    if distribution is None:
        return float(default)

    sample = getattr(distribution, "sample", None)
    if callable(sample):
        return float(sample(rng))

    low = float(distribution.low)
    high = float(distribution.high)
    if type(distribution).__name__ == "LogRange":
        return float(10 ** rng.uniform(math.log10(low), math.log10(high)))

    return float(rng.uniform(min(low, high), max(low, high)))


def _sample_vec3_like(
    distribution: Any,
    rng: np.random.Generator,
    default: Vec3,
) -> Vec3:
    if distribution is None:
        return default

    sample = getattr(distribution, "sample", None)
    if callable(sample):
        sampled = sample(rng)
        if isinstance(sampled, Vec3):
            return sampled
        return Vec3(x=float(sampled[0]), y=float(sampled[1]), z=float(sampled[2]))

    return Vec3(
        x=_sample_range_like(getattr(distribution, "x"), rng, default.x),
        y=_sample_range_like(getattr(distribution, "y"), rng, default.y),
        z=_sample_range_like(getattr(distribution, "z"), rng, default.z),
    )


def _sample_choice_like(
    distribution: Any,
    rng: np.random.Generator,
    default: Any,
) -> Any:
    if distribution is None:
        return default

    sample = getattr(distribution, "sample", None)
    if callable(sample):
        return sample(rng)

    options = list(distribution.options)
    if not options:
        return default
    if len(options) == 1:
        return options[0]

    raw_weights = getattr(distribution, "weights", None)
    if raw_weights is None:
        return options[int(rng.integers(0, len(options)))]

    weights = np.asarray(raw_weights, dtype=float)
    if float(np.sum(weights)) <= 0.0:
        return options[int(rng.integers(0, len(options)))]
    weights = weights / float(np.sum(weights))
    return options[int(rng.choice(len(options), p=weights))]


def _sample_weighted_label(
    volumes: Sequence[TargetVolume],
    weight_map: dict[str, float] | None,
    rng: np.random.Generator,
) -> TargetVolume:
    if not volumes:
        return TargetVolume(
            label="general",
            center_scene=Vec3(x=0.0, y=0.0, z=0.0),
            half_extents_scene=Vec3(x=0.0, y=0.0, z=0.0),
        )
    if not weight_map:
        return volumes[int(rng.integers(0, len(volumes)))]
    weights = []
    for volume in volumes:
        weights.append(float(weight_map.get(volume.label, 0.0)))
    if sum(weights) <= 0.0:
        return volumes[int(rng.integers(0, len(volumes)))]
    normalized = np.asarray(weights, dtype=float)
    normalized /= float(np.sum(normalized))
    return volumes[int(rng.choice(len(volumes), p=normalized))]


def _has_fields(randomization: Any, fields: Sequence[str]) -> bool:
    return any(getattr(randomization, field) is not None for field in fields)


def _sample_tissue_material(
    base: TissueMaterial,
    randomization: TissueMaterialRandomization | None,
    rng: np.random.Generator,
) -> tuple[TissueMaterial, dict[str, float]]:
    if randomization is None:
        return base, {}

    updates: dict[str, float] = {}
    young = _sample_range_like(
        randomization.young_modulus_pa,
        rng,
        default=float(base.young_modulus_pa),
    )
    if randomization.young_modulus_pa is not None:
        updates["young_modulus_pa"] = young

    poisson = _sample_range_like(
        randomization.poisson_ratio,
        rng,
        default=float(base.poisson_ratio),
    )
    if randomization.poisson_ratio is not None:
        updates["poisson_ratio"] = poisson

    total_mass = _sample_range_like(
        randomization.total_mass_kg,
        rng,
        default=float(base.total_mass_kg),
    )
    if randomization.total_mass_kg is not None:
        updates["total_mass_kg"] = total_mass

    rayleigh = _sample_range_like(
        randomization.rayleigh_stiffness,
        rng,
        default=float(base.rayleigh_stiffness),
    )
    if randomization.rayleigh_stiffness is not None:
        updates["rayleigh_stiffness"] = rayleigh

    if not updates:
        return base, {}
    return base.model_copy(update=updates), updates


def _sample_tissue_topology(
    base: TissueTopology,
    randomization: TissueTopologyRandomization | None,
    rng: np.random.Generator,
) -> tuple[TissueTopology, dict[str, tuple[int, int, int]]]:
    if randomization is None or randomization.sparse_grid_n is None:
        return base, {}
    sampled = _sample_choice_like(randomization.sparse_grid_n, rng, default=base.sparse_grid_n)
    if not isinstance(sampled, tuple):
        sampled = tuple(sampled)
    sampled = (int(sampled[0]), int(sampled[1]), int(sampled[2]))
    return TissueTopology(sparse_grid_n=sampled), {"sparse_grid_n": sampled}


def _sample_tissue_mesh(
    base: MeshPerturbation,
    randomization: MeshPerturbationRandomization | None,
    target_volumes: list[TargetVolume],
    rng: np.random.Generator,
) -> tuple[MeshPerturbation, dict[str, Any]]:
    if randomization is None:
        return base, {}
    if (
        not _has_fields(
            randomization,
            (
                "scale_x",
                "scale_y",
                "scale_z",
                "translation_scene",
                "bulge_target_volume_weights",
                "bulge_offset_within_volume_frac",
                "bulge_radius_scene",
                "bulge_amplitude_scene",
            ),
        )
        and (randomization.bulge_probability or 0.0) == 0.0
    ):
        return base, {}

    updates: dict[str, Any] = {}
    scale = _as_array(base.scale)
    scale = np.array(
        [
            _sample_range_like(
                randomization.scale_x,
                rng,
                default=scale[0],
            ),
            _sample_range_like(
                randomization.scale_y,
                rng,
                default=scale[1],
            ),
            _sample_range_like(
                randomization.scale_z,
                rng,
                default=scale[2],
            ),
        ],
        dtype=float,
    )
    updates["scale"] = _to_tuple(scale)

    translation = _sample_vec3_like(
        randomization.translation_scene,
        rng,
        default=base.translation_scene,
    )
    updates["translation_scene"] = translation.model_dump()

    bulge = base.bulge
    bulge_probability = float(
        0.0 if randomization.bulge_probability is None else randomization.bulge_probability
    )
    if bulge_probability > 0.0 and rng.random() < bulge_probability:
        target = _sample_weighted_label(
            target_volumes,
            randomization.bulge_target_volume_weights,
            rng,
        )
        frac = _clamp_vec3(
            _sample_vec3_like(
                randomization.bulge_offset_within_volume_frac,
                rng,
                default=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            -1.0,
            1.0,
        )
        half_extents = _as_array(target.half_extents_scene)
        center = _as_array(target.center_scene) + _as_array(frac) * half_extents
        radius = _sample_range_like(
            randomization.bulge_radius_scene,
            rng,
            default=5.0,
        )
        amplitude = _sample_range_like(
            randomization.bulge_amplitude_scene,
            rng,
            default=2.0,
        )
        bulge = BulgeSpec(
            center_scene=Vec3(x=float(center[0]), y=float(center[1]), z=float(center[2])),
            radius_scene=radius,
            amplitude_scene=amplitude,
        )

    updates["bulge"] = None if bulge is None else bulge.model_dump()
    return MeshPerturbation(
        scale=_to_tuple(scale),
        translation_scene=translation,
        bulge=bulge,
    ), updates


def _sample_camera(
    base: SceneConfig,
    randomization: CameraRandomization | None,
    rng: np.random.Generator,
) -> tuple[Pose, CameraIntrinsics, dict[str, Any]]:
    if randomization is None:
        return (
            base.camera_extrinsics_scene,
            base.camera_intrinsics,
            {},
        )
    if not _has_fields(
        randomization,
        (
            "lookat_target_volume_weights",
            "lookat_offset_within_volume_frac",
            "base_offset_scene",
            "distance_jitter_scale",
            "azimuth_deg",
            "elevation_deg",
            "roll_deg",
            "fx_jitter_pct",
            "fy_jitter_pct",
            "principal_point_offset_px",
            "resolution_choices",
        ),
    ):
        return base.camera_extrinsics_scene, base.camera_intrinsics, {}

    target_volumes = list(base.target_volumes) if base.target_volumes else []
    target = _sample_weighted_label(
        target_volumes,
        randomization.lookat_target_volume_weights,
        rng,
    )
    offset_fraction = _clamp_vec3(
        _sample_vec3_like(
            randomization.lookat_offset_within_volume_frac,
            rng,
            default=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        -1.0,
        1.0,
    )

    lookat = _as_array(target.center_scene) + _as_array(target.half_extents_scene) * _as_array(
        offset_fraction
    )
    base_offset = _as_array(randomization.base_offset_scene) if randomization.base_offset_scene is not None else _as_array(base.camera_extrinsics_scene.position) - lookat
    base_offset_norm = float(np.linalg.norm(base_offset))
    if base_offset_norm <= _EPS:
        base_offset = _as_array(Vec3(x=0.0, y=0.0, z=1.0))
        base_offset_norm = 1.0

    azimuth_deg = float(
        _sample_range_like(
            randomization.azimuth_deg,
            rng,
            default=0.0,
        )
    )
    elevation_deg = float(
        _sample_range_like(
            randomization.elevation_deg,
            rng,
            default=0.0,
        )
    )
    roll_deg = float(
        _sample_range_like(
            randomization.roll_deg,
            rng,
            default=0.0,
        )
    )
    distance_scale = float(
        _sample_range_like(
            randomization.distance_jitter_scale,
            rng,
            default=1.0,
        )
    )
    distance_scale = _clamp(distance_scale, *_CAMERA_DISTANCE_SCALE)

    world_up_array = _as_array(WORLD_UP_SCENE)
    sampled_position = _as_array(base.camera_extrinsics_scene.position)
    sampled_roll = roll_deg
    for _ in range(_CAMERA_ATTEMPTS):
        rotated = _rotate_vector(base_offset, world_up_array, math.radians(azimuth_deg))
        right = _normalize(
            np.cross(world_up_array, -rotated),
            fallback=np.array([1.0, 0.0, 0.0], dtype=float),
        )
        rotated = _rotate_vector(rotated, right, math.radians(elevation_deg))
        rotated = _normalize(rotated) * base_offset_norm * distance_scale
        sampled_position = lookat + rotated
        sampled_distance = float(np.linalg.norm(sampled_position - lookat))
        if sampled_distance >= _CAMERA_MIN_DISTANCE:
            break
    else:
        raise ValueError("Could not sample a non-degenerate camera placement.")

    rotation_matrix = _look_at_rotation_matrix(
        sampled_position,
        lookat,
        world_up_array,
    )
    rotation_matrix = _apply_roll(rotation_matrix, math.radians(sampled_roll))
    sampled_rotation = _matrix_to_quaternion(rotation_matrix)

    base_intrinsics = base.camera_intrinsics
    resolution = _sample_choice_like(
        randomization.resolution_choices,
        rng,
        default=(base_intrinsics.width, base_intrinsics.height),
    )
    width = int(resolution[0])
    height = int(resolution[1])

    fx = float(
        base_intrinsics.fx
        * (1.0 + _sample_range_like(randomization.fx_jitter_pct, rng, default=0.0) / 100.0)
    )
    fy = float(
        base_intrinsics.fy
        * (1.0 + _sample_range_like(randomization.fy_jitter_pct, rng, default=0.0) / 100.0)
    )

    cx_scale = float(width) / float(base_intrinsics.width)
    cy_scale = float(height) / float(base_intrinsics.height)
    cx = float(base_intrinsics.cx * cx_scale)
    cy = float(base_intrinsics.cy * cy_scale)
    principal_offset = _sample_vec3_like(
        randomization.principal_point_offset_px,
        rng,
        default=Vec3(x=0.0, y=0.0, z=0.0),
    )
    cx = _clamp(cx + principal_offset.x, 0.0, float(width - 1))
    cy = _clamp(cy + principal_offset.y, 0.0, float(height - 1))

    sampled_intrinsics = CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
    )
    sampled_pose = Pose(
        position=Vec3(
            x=float(sampled_position[0]),
            y=float(sampled_position[1]),
            z=float(sampled_position[2]),
        ),
        rotation=sampled_rotation,
    )

    record: dict[str, Any] = {
        "lookat_scene": {
            "x": float(lookat[0]),
            "y": float(lookat[1]),
            "z": float(lookat[2]),
        },
        "position_scene": {
            "x": float(sampled_position[0]),
            "y": float(sampled_position[1]),
            "z": float(sampled_position[2]),
        },
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "roll_deg": sampled_roll,
        "distance_jitter_scale": distance_scale,
        "intrinsics": sampled_intrinsics.model_dump(),
    }
    return sampled_pose, sampled_intrinsics, record


def _sample_lighting(
    base: LightingSpec,
    randomization: LightingRandomization | None,
    rng: np.random.Generator,
) -> tuple[LightingSpec, dict[str, Any]]:
    if randomization is None:
        return base, {}
    if not _has_fields(
        randomization,
        (
            "directional_direction_scene",
            "directional_intensity",
            "directional_color_rgb_tint",
            "spot_position_scene",
            "spot_direction_scene",
            "spot_cone_half_angle_deg",
            "spot_intensity",
            "spot_color_rgb_tint",
            "background_rgb",
        ),
    ):
        return base, {}

    updates: dict[str, Any] = {}
    directional = base.directional
    if any(
        getattr(randomization, field) is not None
        for field in (
            "directional_direction_scene",
            "directional_intensity",
            "directional_color_rgb_tint",
        )
    ):
        base_direction = (
            base.directional
            if base.directional is not None
            else DirectionalLight(
                direction_scene=Vec3(x=0.0, y=-1.0, z=0.0),
                intensity=1.0,
                color_rgb=(1.0, 1.0, 1.0),
            )
        )
        base_direction_vec = _as_array(base_direction.direction_scene)
        base_color = _as_array(Vec3(x=base_direction.color_rgb[0], y=base_direction.color_rgb[1], z=base_direction.color_rgb[2]))
        direction = _sample_vec3_like(
            randomization.directional_direction_scene,
            rng,
            default=base_direction.direction_scene,
        )
        intensity = _sample_range_like(
            randomization.directional_intensity,
            rng,
            default=float(base_direction.intensity),
        )
        tint = _sample_vec3_like(
            randomization.directional_color_rgb_tint,
            rng,
            default=Vec3(x=1.0, y=1.0, z=1.0),
        )
        tinted = base_color * _as_array(tint)
        directional = DirectionalLight(
            direction_scene=direction,
            intensity=intensity,
            color_rgb=(float(tinted[0]), float(tinted[1]), float(tinted[2])),
        )
        updates["directional"] = directional.model_dump()

    spot = base.spot
    if any(
        getattr(randomization, field) is not None
        for field in (
            "spot_position_scene",
            "spot_direction_scene",
            "spot_cone_half_angle_deg",
            "spot_intensity",
            "spot_color_rgb_tint",
        )
    ):
        base_spot = (
            base.spot
            if base.spot is not None
            else SpotLight(
                position_scene=Vec3(x=0.0, y=1.0, z=1.0),
                direction_scene=Vec3(x=0.0, y=-1.0, z=0.0),
                cone_half_angle_deg=30.0,
                intensity=1.0,
                color_rgb=(1.0, 1.0, 1.0),
            )
        )
        position = _sample_vec3_like(
            randomization.spot_position_scene,
            rng,
            default=base_spot.position_scene,
        )
        direction = _sample_vec3_like(
            randomization.spot_direction_scene,
            rng,
            default=base_spot.direction_scene,
        )
        cone = _sample_range_like(
            randomization.spot_cone_half_angle_deg,
            rng,
            default=float(base_spot.cone_half_angle_deg),
        )
        intensity = _sample_range_like(
            randomization.spot_intensity,
            rng,
            default=float(base_spot.intensity),
        )
        tint = _sample_vec3_like(
            randomization.spot_color_rgb_tint,
            rng,
            default=Vec3(x=1.0, y=1.0, z=1.0),
        )
        base_color = _as_array(
            Vec3(
                x=base_spot.color_rgb[0],
                y=base_spot.color_rgb[1],
                z=base_spot.color_rgb[2],
            )
        )
        tinted = base_color * _as_array(tint)
        spot = SpotLight(
            position_scene=position,
            direction_scene=direction,
            cone_half_angle_deg=cone,
            intensity=intensity,
            color_rgb=(float(tinted[0]), float(tinted[1]), float(tinted[2])),
        )
        updates["spot"] = spot.model_dump()

    if randomization.background_rgb is not None:
        background = _sample_vec3_like(
            randomization.background_rgb,
            rng,
            default=Vec3(
                x=float(base.background_rgb[0]),
                y=float(base.background_rgb[1]),
                z=float(base.background_rgb[2]),
            ),
        )
        updates["background_rgb"] = _to_tuple(_as_array(background))
    else:
        background = Vec3(
            x=float(base.background_rgb[0]),
            y=float(base.background_rgb[1]),
            z=float(base.background_rgb[2]),
        )

    return (
        LightingSpec(
            directional=directional,
            spot=spot,
            background_rgb=(float(background.x), float(background.y), float(background.z)),
        ),
        {
            "directional": updates.get("directional"),
            "spot": updates.get("spot"),
            "background_rgb": updates.get("background_rgb", {}),
        }
    )


def _sample_visual_tint(
    base: VisualOverrides | None,
    randomization: VisualTintRandomization | None,
    rng: np.random.Generator,
) -> tuple[VisualOverrides | None, dict[str, Any]]:
    if randomization is None:
        return base, {}
    if not _has_fields(
        randomization,
        (
            "tissue_texture_tint_rgb",
            "forceps_body_color_rgb",
            "forceps_clasper_color_rgb",
        ),
    ):
        return base or VisualOverrides(), {}

    base_visual = base or VisualOverrides()
    updates: dict[str, Any] = {}

    if randomization.tissue_texture_tint_rgb is not None:
        sampled = _sample_vec3_like(
            randomization.tissue_texture_tint_rgb,
            rng,
            default=Vec3(
                x=1.0,
                y=1.0,
                z=1.0,
            ),
        )
        updates["tissue_texture_tint_rgb"] = sampled.model_dump()
        base_visual = base_visual.model_copy(
            update={"tissue_texture_tint_rgb": (sampled.x, sampled.y, sampled.z)}
        )

    if randomization.forceps_body_color_rgb is not None:
        sampled = _sample_vec3_like(
            randomization.forceps_body_color_rgb,
            rng,
            default=Vec3(x=1.0, y=1.0, z=1.0),
        )
        base_alpha = (
            float(base_visual.body_color[3]) if base_visual.body_color is not None else 1.0
        )
        updates["body_color"] = (
            float(sampled.x),
            float(sampled.y),
            float(sampled.z),
            base_alpha,
        )
        base_visual = base_visual.model_copy(
            update={"body_color": updates["body_color"]}
        )

    if randomization.forceps_clasper_color_rgb is not None:
        sampled = _sample_vec3_like(
            randomization.forceps_clasper_color_rgb,
            rng,
            default=Vec3(x=1.0, y=1.0, z=1.0),
        )
        base_alpha = (
            float(base_visual.clasper_color[3])
            if base_visual.clasper_color is not None
            else 1.0
        )
        updates["clasper_color"] = (
            float(sampled.x),
            float(sampled.y),
            float(sampled.z),
            base_alpha,
        )
        base_visual = base_visual.model_copy(
            update={"clasper_color": updates["clasper_color"]}
        )

    return base_visual, updates


def _sample_tone_augmentation(
    base: VisualToneAugmentation,
    randomization: ToneAugmentationRandomization | None,
    rng: np.random.Generator,
) -> tuple[VisualToneAugmentation, dict[str, Any]]:
    if randomization is None:
        return base, {}
    if not _has_fields(
        randomization,
        ("brightness_scale", "contrast_scale", "gamma", "saturation_scale"),
    ):
        return base, {}

    updates: dict[str, Any] = {}
    brightness = _sample_range_like(
        randomization.brightness_scale,
        rng,
        default=float(base.brightness_scale),
    )
    if randomization.brightness_scale is not None:
        updates["brightness_scale"] = brightness
    contrast = _sample_range_like(
        randomization.contrast_scale,
        rng,
        default=float(base.contrast_scale),
    )
    if randomization.contrast_scale is not None:
        updates["contrast_scale"] = contrast
    gamma = _sample_range_like(
        randomization.gamma,
        rng,
        default=float(base.gamma),
    )
    if randomization.gamma is not None:
        updates["gamma"] = gamma
    saturation = _sample_range_like(
        randomization.saturation_scale,
        rng,
        default=float(base.saturation_scale),
    )
    if randomization.saturation_scale is not None:
        updates["saturation_scale"] = saturation

    sampled = base.model_copy(
        update={
            "brightness_scale": brightness,
            "contrast_scale": contrast,
            "gamma": gamma,
            "saturation_scale": saturation,
        }
    )
    return sampled, updates


def _sample_motion(
    base: MotionGeneratorConfig,
    rng: np.random.Generator,
) -> tuple[MotionGeneratorConfig, dict[str, int]]:
    seed = int(rng.integers(0, 2**64, dtype=np.uint64))
    return base.model_copy(update={"seed": seed}), {"seed": seed}


def sample_episode(
    base_scene: SceneConfig,
    base_motion: MotionGeneratorConfig,
    randomization: EpisodeRandomizationConfig,
    episode_seed: int,
) -> EpisodeSpec:
    """Sample one deterministic episode realization for scene, motion, and records."""
    seed = int(episode_seed)
    scene = base_scene.model_copy(deep=True)
    motion = base_motion.model_copy(deep=True)
    randomization = randomization or EpisodeRandomizationConfig()

    scene_material, tissue_material_record = _sample_tissue_material(
        scene.tissue_material, randomization.tissue_material, _named_subrng(seed, "tissue_material")
    )
    scene = scene.model_copy(update={"tissue_material": scene_material})

    scene_topology, tissue_topology_record = _sample_tissue_topology(
        scene.tissue_topology, randomization.tissue_topology, _named_subrng(seed, "tissue_topology")
    )
    scene = scene.model_copy(update={"tissue_topology": scene_topology})

    scene_mesh, tissue_mesh_record = _sample_tissue_mesh(
        scene.tissue_mesh_perturbation,
        randomization.tissue_mesh,
        scene.target_volumes,
        _named_subrng(seed, "tissue_mesh"),
    )
    scene = scene.model_copy(update={"tissue_mesh_perturbation": scene_mesh})

    sampled_pose, sampled_intrinsics, camera_record = _sample_camera(
        scene,
        randomization.camera,
        _named_subrng(seed, "camera"),
    )
    scene = scene.model_copy(
        update={
            "camera_extrinsics_scene": sampled_pose,
            "camera_intrinsics": sampled_intrinsics,
        }
    )

    sampled_lighting, lighting_record = _sample_lighting(
        scene.lighting,
        randomization.lighting,
        _named_subrng(seed, "lighting"),
    )
    scene = scene.model_copy(update={"lighting": sampled_lighting})

    sampled_visual, visual_record = _sample_visual_tint(
        scene.tool.visual_overrides,
        randomization.visual_tint,
        _named_subrng(seed, "visual_tint"),
    )
    scene = scene.model_copy(
        update={"tool": scene.tool.model_copy(update={"visual_overrides": sampled_visual})}
    )
    sampled_tone, tone_record = _sample_tone_augmentation(
        scene.tone_augmentation,
        randomization.tone_augmentation,
        _named_subrng(seed, "tone_augmentation"),
    )
    scene = scene.model_copy(update={"tone_augmentation": sampled_tone})

    motion, _ = _sample_motion(
        motion,
        _named_subrng(seed, "motion"),
    )

    return EpisodeSpec(
        episode_seed=seed,
        scene=scene,
        motion=motion,
        sample_record=SampleRecord(
            tissue_material=tissue_material_record,
            tissue_topology=tissue_topology_record,
            tissue_mesh=tissue_mesh_record,
            camera=camera_record,
            lighting=lighting_record,
            visual_tint=visual_record,
            tone_augmentation=tone_record,
        ),
    )

