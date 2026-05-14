"""YAML-backed configuration loaders used by CLI and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from auto_surgery.schemas.motion import MotionGeneratorConfig, MotionShaping
    from auto_surgery.schemas.scene import SceneConfig


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MOTION_SHAPING_DEFAULTS_PATH = Path(__file__).resolve().parent / "config" / "motion_shaping_defaults.yaml"


def _load_yaml(path: str | Path) -> dict[str, object]:
    """Load a YAML document into a plain dict for model validation."""

    requested = Path(path)
    if requested.is_absolute():
        candidate_paths = (requested,)
    else:
        candidate_paths = (requested, _PROJECT_ROOT / requested)

    resolved_path: Path | None = None
    last_error: OSError | None = None
    for path_candidate in candidate_paths:
        try:
            raw = path_candidate.read_text(encoding="utf-8")
            resolved_path = path_candidate
            break
        except OSError as exc:
            last_error = exc
            resolved_path = path_candidate
    else:
        raise RuntimeError(
            f"Cannot read config file '{requested}'. Tried: {', '.join(str(p) for p in candidate_paths)}."
            f" Last error: {last_error}"
        )

    assert resolved_path is not None
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in config file '{resolved_path}': {exc}") from exc

    if not isinstance(payload, dict):
        payload_type = type(payload).__name__
        raise RuntimeError(
            f"Config file '{resolved_path}' must contain a YAML mapping, got {payload_type}."
        )

    if not payload:
        raise RuntimeError(f"Config file '{resolved_path}' is empty.")

    return payload


def load_scene_config(path: str | Path) -> SceneConfig:
    """Load and validate a scene configuration from YAML."""

    from auto_surgery.schemas.scene import SceneConfig

    payload = _load_yaml(path)
    return SceneConfig.model_validate(payload)


def load_motion_config(path: str | Path) -> MotionGeneratorConfig:
    """Load and validate a motion configuration from YAML."""

    from auto_surgery.schemas.motion import MotionGeneratorConfig

    payload = _load_yaml(path)
    return MotionGeneratorConfig.model_validate(payload)


def load_scene_motion_shaping(scene_id: str) -> "MotionShaping":
    """Load per-scene ``MotionShaping`` defaults from ``motion_shaping_defaults.yaml``."""

    from auto_surgery.schemas.motion import MotionShaping

    payload = _load_yaml(_MOTION_SHAPING_DEFAULTS_PATH)
    key = str(scene_id).strip()
    if key not in payload:
        available = sorted(str(k) for k in payload)
        raise KeyError(
            f"No motion_shaping defaults for scene_id={key!r}; available: {available}",
        )
    row = payload[key]
    if not isinstance(row, dict):
        raise KeyError(
            f"motion_shaping defaults for scene_id={key!r} must be a mapping, got {type(row).__name__}.",
        )
    return MotionShaping.model_validate(row)


__all__ = ["load_motion_config", "load_scene_config", "load_scene_motion_shaping"]
