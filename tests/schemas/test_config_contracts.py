from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from auto_surgery.config import load_motion_config, load_scene_config, load_scene_motion_shaping
from auto_surgery.schemas.commands import Vec3
from auto_surgery.schemas.motion import MotionGeneratorConfig, MotionShaping
from auto_surgery.schemas.scene import (
    SceneConfig,
    SceneGeometryEnvelope,
    SphereEnvelope,
    TargetVolume,
    TissueAssetSpec,
    ToolSpec,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_CONFIG_PATH = REPO_ROOT / "configs" / "scenes" / "dejavu_brain.yaml"
MOTION_CONFIG_PATH = REPO_ROOT / "configs" / "motion" / "default.yaml"


def test_scene_yaml_loads_and_has_target_volumes() -> None:
    scene_cfg = load_scene_config(SCENE_CONFIG_PATH)
    assert scene_cfg.tool.tool_id == "dejavu_forceps"
    assert scene_cfg.tissue_scene_path is not None
    assert (REPO_ROOT / scene_cfg.tissue_scene_path).is_file()
    assert scene_cfg.tone_augmentation.is_identity()
    assert len(scene_cfg.target_volumes) >= 1
    assert scene_cfg.target_volumes[0].label == "general"


def test_load_scene_motion_shaping_loads_dejavu_brain_row() -> None:
    shaping = load_scene_motion_shaping("dejavu_brain")
    assert shaping.max_linear_mm_s == pytest.approx(45.0)
    assert shaping.max_angular_rad_s == pytest.approx(1.20)
    assert shaping.max_linear_accel_mm_s2 == pytest.approx(80.0)
    assert shaping.max_angular_accel_rad_s2 == pytest.approx(2.20)


def test_load_scene_motion_shaping_missing_scene_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        load_scene_motion_shaping("nonexistent_scene")


def test_motion_generator_with_default_motion_shaping_noop_when_shaping_set() -> None:
    custom = MotionShaping(
        max_linear_mm_s=1.0,
        max_angular_rad_s=1.0,
        max_linear_accel_mm_s2=1.0,
        max_angular_accel_rad_s2=1.0,
        bias_gain_max=0.0,
        bias_ramp_distance_mm=1.0,
        orientation_bias_gain=0.0,
        orientation_deadband_rad=0.0,
    )
    cfg = load_motion_config(MOTION_CONFIG_PATH).model_copy(
        update={"motion_shaping": custom, "motion_shaping_enabled": True},
    )
    out = cfg.with_default_motion_shaping("dejavu_brain")
    assert out is cfg
    assert out.motion_shaping is custom


def test_motion_generator_with_default_motion_shaping_populates_when_none() -> None:
    cfg = load_motion_config(MOTION_CONFIG_PATH).model_copy(update={"motion_shaping": None})
    out = cfg.with_default_motion_shaping("dejavu_brain")
    assert out.motion_shaping is not None
    assert out.motion_shaping_enabled is True
    assert out.motion_shaping.max_linear_mm_s == pytest.approx(45.0)


def test_motion_yaml_loads_and_matches_expected_ranges() -> None:
    motion_cfg = load_motion_config(MOTION_CONFIG_PATH)
    assert motion_cfg.seed == 0
    assert motion_cfg.primitive_count_min == 8
    assert motion_cfg.primitive_count_max == 20
    assert motion_cfg.weight_reach == 1.0


def test_motion_loads_legacy_motion_keys_only(tmp_path: Path) -> None:
    legacy_payload = {
        "seed": 7,
        "weight_approach": 2.2,
        "weight_dwell": 1.1,
        "weight_retract": 0.8,
        "weight_sweep": 0.4,
        "weight_rotate": 0.3,
        "weight_probe": 0.5,
        "approach_duration_range_s": [0.4, 1.0],
        "dwell_duration_range_s": [0.2, 0.5],
        "retract_duration_range_s": [0.3, 0.7],
        "sweep_duration_range_s": [0.6, 1.2],
        "sweep_arc_range_rad": [0.2, 0.7],
    }
    legacy_payload_path = tmp_path / "legacy_motion.yaml"
    legacy_payload_path.write_text(yaml.safe_dump(legacy_payload), encoding="utf-8")

    motion_cfg = load_motion_config(legacy_payload_path)
    assert motion_cfg.weight_reach == 2.2
    assert motion_cfg.weight_hold == 1.1
    assert motion_cfg.weight_drag == 0.8
    assert motion_cfg.weight_brush == 0.4
    assert motion_cfg.weight_grip == 0.3
    assert motion_cfg.weight_contact_reach == 0.5
    assert motion_cfg.reach_duration_range_s == (0.4, 1.0)
    assert motion_cfg.hold_duration_range_s == (0.2, 0.5)
    assert motion_cfg.drag_duration_range_s == (0.3, 0.7)
    assert motion_cfg.brush_duration_range_s == (0.6, 1.2)
    assert motion_cfg.brush_arc_range_rad == (0.2, 0.7)

    dumped = motion_cfg.model_dump()
    assert "weight_approach" not in dumped
    assert "approach_duration_range_s" not in dumped


def test_empty_target_volumes_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["target_volumes"] = []
    bad_scene = tmp_path / "scene_bad.yaml"
    bad_scene.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_scene_config(bad_scene)


def test_scene_loads_reject_empty_document(tmp_path: Path) -> None:
    empty_scene = tmp_path / "scene_empty.yaml"
    empty_scene.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError):
        load_scene_config(empty_scene)


def test_scene_loads_reject_non_mapping_payload(tmp_path: Path) -> None:
    non_mapping_scene = tmp_path / "scene_scalar.yaml"
    non_mapping_scene.write_text("null", encoding="utf-8")

    with pytest.raises(RuntimeError):
        load_scene_config(non_mapping_scene)


def test_motion_loads_reject_non_mapping_payload(tmp_path: Path) -> None:
    non_mapping_motion = tmp_path / "motion_scalar.yaml"
    non_mapping_motion.write_text("null", encoding="utf-8")

    with pytest.raises(RuntimeError):
        load_motion_config(non_mapping_motion)


def test_motion_loads_reject_empty_document(tmp_path: Path) -> None:
    empty_motion = tmp_path / "motion_empty.yaml"
    empty_motion.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError):
        load_motion_config(empty_motion)


def test_motion_loads_reject_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist-motion.yaml"

    with pytest.raises(RuntimeError):
        load_motion_config(missing)


def test_scene_loads_reject_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist-scene.yaml"

    with pytest.raises(RuntimeError):
        load_scene_config(missing)


def test_extra_fields_are_rejected(tmp_path: Path) -> None:
    scene_payload = yaml.safe_load(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    scene_payload["extra_scene_key"] = "nope"
    scene_bad = tmp_path / "scene_extra.yaml"
    scene_bad.write_text(yaml.safe_dump(scene_payload), encoding="utf-8")

    motion_payload = yaml.safe_load(MOTION_CONFIG_PATH.read_text(encoding="utf-8"))
    motion_payload["extra_motion_key"] = 0.0
    motion_bad = tmp_path / "motion_extra.yaml"
    motion_bad.write_text(yaml.safe_dump(motion_payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_scene_config(scene_bad)

    with pytest.raises(ValidationError):
        load_motion_config(motion_bad)


def test_scene_rejects_invalid_tone_augmentation(tmp_path: Path) -> None:
    payload = yaml.safe_load(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["tone_augmentation"] = {
        "gamma": 0.0,
        "brightness_scale": 1.0,
    }
    scene_bad = tmp_path / "tmp_scene_invalid_tone_augmentation.yaml"
    scene_bad.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_scene_config(scene_bad)


def test_loads_motion_config_rejects_invalid_range_ordering(tmp_path: Path) -> None:
    motion_payload = yaml.safe_load(MOTION_CONFIG_PATH.read_text(encoding="utf-8"))
    motion_payload["reach_duration_range_s"] = [1.4, 0.6]
    bad_motion = tmp_path / "motion_inverted_range.yaml"
    bad_motion.write_text(yaml.safe_dump(motion_payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_motion_config(bad_motion)


def test_loads_motion_config_rejects_invalid_probability(tmp_path: Path) -> None:
    motion_payload = yaml.safe_load(MOTION_CONFIG_PATH.read_text(encoding="utf-8"))
    motion_payload["jaw_change_probability"] = 1.25
    bad_motion = tmp_path / "motion_probability.yaml"
    bad_motion.write_text(yaml.safe_dump(motion_payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_motion_config(bad_motion)


def test_scene_loads_reject_missing_tissue_path(tmp_path: Path) -> None:
    payload = yaml.safe_load(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    payload.pop("tissue_scene_path", None)
    bad_scene = tmp_path / "scene_missing_tissue.yaml"
    bad_scene.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_scene_config(bad_scene)


def test_scene_loads_reject_legacy_scene_xml_path_only(tmp_path: Path) -> None:
    payload = yaml.safe_load(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["scene_xml_path"] = payload["tissue_scene_path"]
    payload.pop("tissue_scene_path", None)
    bad_scene = tmp_path / "scene_legacy_key.yaml"
    bad_scene.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_scene_config(bad_scene)


def test_scene_config_auto_builds_workspace_envelope_when_tissue_assets_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_SURGERY_DEJAVU_ROOT", str(tmp_path))
    mesh_rel = Path("scenes/brain/data/surface_full.obj")
    mesh_file = tmp_path / mesh_rel
    mesh_file.parent.mkdir(parents=True)
    mesh_file.write_text("v 0 0 0\nv 10 0 0\nv 0 10 0\nf 1 2 3\n", encoding="utf-8")
    tissue_scn = tmp_path / "tissue.scn"
    tissue_scn.write_text("", encoding="utf-8")
    vol = TargetVolume(
        label="general",
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        half_extents_scene=Vec3(x=1.0, y=1.0, z=1.0),
    )
    scene = SceneConfig(
        tissue_scene_path=tissue_scn,
        tissue_assets=TissueAssetSpec(surface_mesh_relative_path=str(mesh_rel)),
        target_volumes=[vol],
    )
    assert scene.tool.workspace_envelope is not None
    assert isinstance(scene.tool.workspace_envelope, SceneGeometryEnvelope)
    assert scene.tool.workspace_envelope.surface_mesh_path == mesh_file.resolve()


def test_scene_config_respects_explicit_workspace_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_SURGERY_DEJAVU_ROOT", str(tmp_path))
    mesh_rel = Path("scenes/brain/data/surface_full.obj")
    mesh_file = tmp_path / mesh_rel
    mesh_file.parent.mkdir(parents=True)
    mesh_file.write_text("v 0 0 0\nv 10 0 0\nv 0 10 0\nf 1 2 3\n", encoding="utf-8")
    tissue_scn = tmp_path / "tissue.scn"
    tissue_scn.write_text("", encoding="utf-8")
    explicit = SphereEnvelope(
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        radius_mm=5.0,
        outer_margin_mm=1.0,
        inner_margin_mm=0.5,
    )
    vol = TargetVolume(
        label="general",
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        half_extents_scene=Vec3(x=1.0, y=1.0, z=1.0),
    )
    scene = SceneConfig(
        tissue_scene_path=tissue_scn,
        tissue_assets=TissueAssetSpec(surface_mesh_relative_path=str(mesh_rel)),
        tool=ToolSpec(workspace_envelope=explicit),
        target_volumes=[vol],
    )
    assert scene.tool.workspace_envelope is explicit


def test_scene_config_skips_envelope_when_dejavu_root_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTO_SURGERY_DEJAVU_ROOT", raising=False)
    tissue_scn = tmp_path / "tissue.scn"
    tissue_scn.write_text("", encoding="utf-8")
    vol = TargetVolume(
        label="general",
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        half_extents_scene=Vec3(x=1.0, y=1.0, z=1.0),
    )
    scene = SceneConfig(
        tissue_scene_path=tissue_scn,
        tissue_assets=TissueAssetSpec(),
        target_volumes=[vol],
    )
    assert scene.tool.workspace_envelope is None
