"""Preset selection and validation helpers for piece-4 randomization presets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from auto_surgery.schemas import EpisodeRandomizationConfig

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_PRESET_DIR: Final = _REPO_ROOT / "configs" / "randomization"

_PRESET_FILES: Final = {
    "minimal": "minimal.yaml",
    "default": "default.yaml",
    "aggressive": "aggressive.yaml",
}


def list_randomization_presets() -> tuple[str, ...]:
    """Return the built-in preset names in sorted order."""

    return tuple(sorted(_PRESET_FILES))


def preset_directory(override_dir: Path | None = None) -> Path:
    """Return the preset directory.

    Parameters
    ----------
    override_dir:
        Optional alternate directory for tests or custom deployments.
    """

    if override_dir is None:
        return _PRESET_DIR
    return Path(override_dir)


def _is_named_preset(selector: str) -> bool:
    return (
        "/" not in selector
        and "\\" not in selector
        and selector in _PRESET_FILES
    )


def _resolve_explicit_path(selector: str, base_dir: Path) -> Path:
    candidate = Path(selector).expanduser()
    if candidate.is_absolute():
        return candidate
    for candidate_path in (
        Path.cwd() / candidate,
        _REPO_ROOT / candidate,
        base_dir / candidate,
    ):
        if candidate_path.exists():
            return candidate_path
    return base_dir / candidate


def resolve_randomization_preset_path(selector: str, *, preset_dir: Path | None = None) -> Path:
    """Resolve a preset name or path to an explicit YAML path.

    V1 has no per-axis overlay semantics: this resolver only loads a single preset
    file and performs no merging.
    """

    resolved_dir = preset_directory(preset_dir)
    if _is_named_preset(selector):
        return resolved_dir / _PRESET_FILES[selector]

    candidate = _resolve_explicit_path(selector, resolved_dir)
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        yaml_candidate = candidate.with_suffix(".yaml")
        if yaml_candidate.exists():
            return yaml_candidate
        yml_candidate = candidate.with_suffix(".yml")
        if yml_candidate.exists():
            return yml_candidate
    if candidate.exists():
        return candidate

    available = ", ".join(list_randomization_presets())
    raise FileNotFoundError(
        f"Randomization preset '{selector}' not found. "
        f"Expected one of: {available}, or an explicit existing .yaml/.yml path."
    )


def load_randomization_preset(
    selector: str | Path,
    *,
    preset_dir: Path | None = None,
) -> EpisodeRandomizationConfig:
    """Load and validate an :class:`EpisodeRandomizationConfig` preset."""

    preset_path = resolve_randomization_preset_path(str(selector), preset_dir=preset_dir)
    payload = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"Randomization preset payload must be a mapping: {preset_path}")
    try:
        return EpisodeRandomizationConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid EpisodeRandomizationConfig in preset: {preset_path}") from exc
