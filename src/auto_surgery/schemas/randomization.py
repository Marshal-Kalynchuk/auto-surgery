"""Typed distribution and randomization contracts for piece-4."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from auto_surgery.schemas.commands import Vec3
from auto_surgery.schemas.motion import MotionGeneratorConfig
from auto_surgery.schemas.scene import SceneConfig
from auto_surgery.randomization.distributions import Choice, LogRange, Range, Vec3Range


class TissueMaterialRandomization(BaseModel):
    model_config = {"extra": "forbid"}

    young_modulus_pa: LogRange | None = None
    poisson_ratio: Range | None = None
    total_mass_kg: Range | None = None
    rayleigh_stiffness: Range | None = None


class TissueTopologyRandomization(BaseModel):
    model_config = {"extra": "forbid"}

    sparse_grid_n: Choice[tuple[int, int, int]] | None = None


class MeshPerturbationRandomization(BaseModel):
    model_config = {"extra": "forbid"}

    scale_x: Range | None = None
    scale_y: Range | None = None
    scale_z: Range | None = None
    translation_scene: Vec3Range | None = None
    bulge_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    bulge_target_volume_weights: dict[str, float] | None = None
    bulge_offset_within_volume_frac: Vec3Range | None = None
    bulge_radius_scene: Range | None = None
    bulge_amplitude_scene: Range | None = None


class CameraRandomization(BaseModel):
    model_config = {"extra": "forbid"}

    lookat_target_volume_weights: dict[str, float] | None = None
    lookat_offset_within_volume_frac: Vec3Range | None = None
    base_offset_scene: Vec3 | None = None
    distance_jitter_scale: Range | None = None
    azimuth_deg: Range | None = None
    elevation_deg: Range | None = None
    roll_deg: Range | None = None
    fx_jitter_pct: Range | None = None
    fy_jitter_pct: Range | None = None
    principal_point_offset_px: Vec3Range | None = None
    resolution_choices: Choice[tuple[int, int]] | None = None


class LightingRandomization(BaseModel):
    model_config = {"extra": "forbid"}

    directional_direction_scene: Vec3Range | None = None
    directional_intensity: Range | None = None
    directional_color_rgb_tint: Vec3Range | None = None
    spot_position_scene: Vec3Range | None = None
    spot_direction_scene: Vec3Range | None = None
    spot_cone_half_angle_deg: Range | None = None
    spot_intensity: Range | None = None
    spot_color_rgb_tint: Vec3Range | None = None
    background_rgb: Vec3Range | None = None


class VisualTintRandomization(BaseModel):
    model_config = {"extra": "forbid"}

    tissue_texture_tint_rgb: Vec3Range | None = None
    forceps_body_color_rgb: Vec3Range | None = None
    forceps_clasper_color_rgb: Vec3Range | None = None


class ToneAugmentationRandomization(BaseModel):
    model_config = {"extra": "forbid"}

    brightness_scale: Range | None = None
    contrast_scale: Range | None = None
    gamma: Range | None = None
    saturation_scale: Range | None = None


class EpisodeRandomizationConfig(BaseModel):
    """Top-level config consumed by sample_episode."""

    model_config = {"extra": "forbid"}

    tissue_material: TissueMaterialRandomization | None = None
    tissue_topology: TissueTopologyRandomization | None = None
    tissue_mesh: MeshPerturbationRandomization | None = None
    camera: CameraRandomization | None = None
    lighting: LightingRandomization | None = None
    visual_tint: VisualTintRandomization | None = None
    tone_augmentation: ToneAugmentationRandomization | None = None


class SampleRecord(BaseModel):
    """Per-axis log of what the sampler drew.

    `episode_seed` is not duplicated here; it lives on the parent `EpisodeSpec`.
    """

    model_config = {"extra": "forbid"}

    tissue_material: dict[str, float] = Field(default_factory=dict)
    tissue_topology: dict[str, tuple[int, int, int]] = Field(default_factory=dict)
    tissue_mesh: dict[str, Any] = Field(default_factory=dict)
    camera: dict[str, Any] = Field(default_factory=dict)
    lighting: dict[str, Any] = Field(default_factory=dict)
    visual_tint: dict[str, Any] = Field(default_factory=dict)
    tone_augmentation: dict[str, Any] = Field(default_factory=dict)


class EpisodeSpec(BaseModel):
    """Output of sample_episode: the realised episode description."""

    model_config = {"extra": "forbid"}

    episode_seed: int
    scene: SceneConfig
    motion: MotionGeneratorConfig
    sample_record: SampleRecord
