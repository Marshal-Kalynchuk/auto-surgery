from __future__ import annotations

import json

from auto_surgery.config import load_motion_config, load_scene_config
from auto_surgery.randomization import sample_episode
from auto_surgery.randomization.distributions import LogRange
from auto_surgery.randomization.presets import load_randomization_preset
from auto_surgery.schemas.randomization import (
    CameraRandomization,
    EpisodeRandomizationConfig,
    Range,
    SampleRecord,
    ToneAugmentationRandomization,
    TissueMaterialRandomization,
)


def _spec_bytes(spec: object) -> bytes:
    return json.dumps(
        spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def test_sample_episode_byte_trace_is_deterministic_for_fixed_seed() -> None:
    scene = load_scene_config("configs/scenes/dejavu_brain.yaml")
    motion = load_motion_config("configs/motion/default.yaml")
    randomization = load_randomization_preset("minimal")

    first = sample_episode(scene, motion, randomization, episode_seed=1729)
    second = sample_episode(scene, motion, randomization, episode_seed=1729)

    assert _spec_bytes(first) == _spec_bytes(second)
    assert _spec_bytes(first) == _spec_bytes(first.model_copy(deep=True))


def test_sample_episode_default_randomization_config_returns_base_artifacts() -> None:
    scene = load_scene_config("configs/scenes/dejavu_brain.yaml")
    motion = load_motion_config("configs/motion/default.yaml")

    spec = sample_episode(scene, motion, EpisodeRandomizationConfig(), episode_seed=123)

    assert spec.episode_seed == 123
    assert spec.scene == scene
    assert spec.motion.model_dump(exclude={"seed"}) == motion.model_dump(exclude={"seed"})
    assert spec.sample_record == SampleRecord()


def test_sample_episode_axes_use_independent_named_rng_substreams() -> None:
    scene = load_scene_config("configs/scenes/dejavu_brain.yaml")
    motion = load_motion_config("configs/motion/default.yaml")
    seed = 2026

    material_only = EpisodeRandomizationConfig(
        tissue_material=TissueMaterialRandomization(
            young_modulus_pa=LogRange(low=1_500.0, high=1_500.0),
            poisson_ratio=Range(low=0.42, high=0.42),
        )
    )
    camera_only = EpisodeRandomizationConfig(
        camera=CameraRandomization(
            fx_jitter_pct=Range(low=0.0, high=0.0),
            fy_jitter_pct=Range(low=0.0, high=0.0),
        )
    )

    material_only_spec = sample_episode(scene, motion, material_only, episode_seed=seed)
    camera_only_spec = sample_episode(scene, motion, camera_only, episode_seed=seed)
    both_spec = sample_episode(
        scene,
        motion,
        EpisodeRandomizationConfig(
            tissue_material=material_only.tissue_material,
            camera=camera_only.camera,
        ),
        episode_seed=seed,
    )

    assert both_spec.scene.tissue_material == material_only_spec.scene.tissue_material
    assert both_spec.scene.camera_extrinsics_scene == camera_only_spec.scene.camera_extrinsics_scene
    assert both_spec.scene.camera_intrinsics == camera_only_spec.scene.camera_intrinsics
    assert both_spec.sample_record.tissue_material == material_only_spec.sample_record.tissue_material
    assert both_spec.sample_record.camera == camera_only_spec.sample_record.camera


def test_sample_episode_tone_augmentation_axis_is_sampled_and_recorded() -> None:
    scene = load_scene_config("configs/scenes/dejavu_brain.yaml")
    motion = load_motion_config("configs/motion/default.yaml")
    randomization = EpisodeRandomizationConfig(
        tone_augmentation=ToneAugmentationRandomization(
            brightness_scale=Range(low=1.15, high=1.15),
            contrast_scale=Range(low=0.9, high=0.9),
            gamma=Range(low=2.0, high=2.0),
            saturation_scale=Range(low=1.1, high=1.1),
        )
    )

    spec = sample_episode(scene, motion, randomization, episode_seed=314)

    assert spec.scene.tone_augmentation.brightness_scale == 1.15
    assert spec.scene.tone_augmentation.contrast_scale == 0.9
    assert spec.scene.tone_augmentation.gamma == 2.0
    assert spec.scene.tone_augmentation.saturation_scale == 1.1
    assert spec.sample_record.tone_augmentation["brightness_scale"] == 1.15
    assert spec.sample_record.tone_augmentation["contrast_scale"] == 0.9
    assert spec.sample_record.tone_augmentation["gamma"] == 2.0
    assert spec.sample_record.tone_augmentation["saturation_scale"] == 1.1
