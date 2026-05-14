from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from auto_surgery.config import load_motion_config, load_scene_config


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
