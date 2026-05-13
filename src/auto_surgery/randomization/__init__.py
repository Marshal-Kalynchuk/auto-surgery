"""Top-level randomization package API for piece-4."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_surgery.randomization.distributions import Choice, LogRange, Range, Vec3Range
    from auto_surgery.randomization.mesh_warp import warp_tissue_meshes
    from auto_surgery.randomization.presets import (
        list_randomization_presets,
        load_randomization_preset,
        resolve_randomization_preset_path,
    )
    from auto_surgery.randomization.sampler import _AXIS_NAMES, _named_subrng, sample_episode
    from auto_surgery.randomization.scn_template import render_scene_template


__all__ = [
    "Choice",
    "LogRange",
    "Range",
    "Vec3Range",
    "warp_tissue_meshes",
    "render_scene_template",
    "_AXIS_NAMES",
    "_named_subrng",
    "sample_episode",
    "list_randomization_presets",
    "load_randomization_preset",
    "resolve_randomization_preset_path",
]

_EXPORTS = {
    "Choice": ("auto_surgery.randomization.distributions", "Choice"),
    "LogRange": ("auto_surgery.randomization.distributions", "LogRange"),
    "Range": ("auto_surgery.randomization.distributions", "Range"),
    "Vec3Range": ("auto_surgery.randomization.distributions", "Vec3Range"),
    "warp_tissue_meshes": ("auto_surgery.randomization.mesh_warp", "warp_tissue_meshes"),
    "render_scene_template": ("auto_surgery.randomization.scn_template", "render_scene_template"),
    "_AXIS_NAMES": ("auto_surgery.randomization.sampler", "_AXIS_NAMES"),
    "_named_subrng": ("auto_surgery.randomization.sampler", "_named_subrng"),
    "sample_episode": ("auto_surgery.randomization.sampler", "sample_episode"),
    "list_randomization_presets": ("auto_surgery.randomization.presets", "list_randomization_presets"),
    "load_randomization_preset": ("auto_surgery.randomization.presets", "load_randomization_preset"),
    "resolve_randomization_preset_path": ("auto_surgery.randomization.presets", "resolve_randomization_preset_path"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'auto_surgery.randomization' has no attribute {name!r}")

    module_name, export_name = target
    module = import_module(module_name)
    value = getattr(module, export_name)
    globals()[name] = value
    return value
