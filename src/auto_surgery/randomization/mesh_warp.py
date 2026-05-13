"""Mesh perturbation helpers for tissue assets."""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import numpy as np

from auto_surgery.schemas.scene import MeshPerturbation

_OUTPUT_PREFIX = "auto-surgery-tissue-warp-"
_EPS = 1.0e-9


def _import_trimesh():
    """Import trimesh lazily so environments can import this module without SOFA extras."""

    try:
        return importlib.import_module("trimesh")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "trimesh is required for mesh perturbation. Install it from the `prep` dependency group."
        ) from exc


def _as_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser()


def _as_vec3(values: tuple[float, float, float] | list[float] | object) -> np.ndarray:
    if isinstance(values, (list, tuple)) and len(values) == 3:
        return np.asarray(values, dtype=float)
    if hasattr(values, "x") and hasattr(values, "y") and hasattr(values, "z"):
        return np.asarray([values.x, values.y, values.z], dtype=float)
    raise TypeError("Expected a 3D vector.")


def _warp_one(
    src_path: Path,
    perturbation: MeshPerturbation,
    output_dir: Path | None,
    *,
    suffix: str,
) -> Path:
    trimesh = _import_trimesh()
    mesh = trimesh.load(src_path, process=False, force="mesh")
    if not hasattr(mesh, "vertices"):
        raise ValueError(f"Mesh loader did not return a valid mesh for: {src_path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    scale = _as_vec3(perturbation.scale)
    translation = _as_vec3(perturbation.translation_scene)
    vertices *= scale
    vertices += translation

    if perturbation.bulge is not None:
        center = _as_vec3(perturbation.bulge.center_scene)
        radius = float(perturbation.bulge.radius_scene)
        amplitude = float(perturbation.bulge.amplitude_scene)
        if radius <= 0.0:
            raise ValueError(f"Bulge radius must be positive: {radius}")

        diff = vertices - center
        distance = np.linalg.norm(diff, axis=1, keepdims=True)
        gauss = amplitude * np.exp(-0.5 * (distance / radius) ** 2)
        direction = diff / np.maximum(distance, _EPS)
        vertices += gauss * direction

    mesh.vertices = vertices

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix=_OUTPUT_PREFIX,
        mode="wb",
        delete=False,
        dir=output_dir,
    ) as handle:
        output_path = Path(handle.name)

    mesh.export(output_path, file_type="obj")
    return output_path


def warp_tissue_meshes(
    volume_obj_path: str | Path,
    surface_obj_path: str | Path,
    perturbation: MeshPerturbation,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Apply geometry perturbation to both tissue meshes and return rendered `.obj` paths."""

    volume_path = _as_path(volume_obj_path)
    surface_path = _as_path(surface_obj_path)
    if perturbation.is_identity():
        return volume_path, surface_path

    output_directory = Path(output_dir) if output_dir is not None else None
    return (
        _warp_one(volume_path, perturbation, output_directory, suffix="-volume.obj"),
        _warp_one(surface_path, perturbation, output_directory, suffix="-surface.obj"),
    )
