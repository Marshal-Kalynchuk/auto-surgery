from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from auto_surgery.randomization.mesh_warp import warp_tissue_meshes
from auto_surgery.schemas.scene import BulgeSpec, MeshPerturbation, Vec3

trimesh = pytest.importorskip("trimesh")


def _write_tetra_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=int,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)


def _read_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh", process=False)
    return np.asarray(mesh.vertices, dtype=float)


def test_identity_perturbation_returns_original_paths(tmp_path: Path) -> None:
    volume = tmp_path / "volume.obj"
    surface = tmp_path / "surface.obj"
    _write_tetra_mesh(volume)
    _write_tetra_mesh(surface)

    out_volume, out_surface = warp_tissue_meshes(volume, surface, MeshPerturbation())

    assert out_volume == volume
    assert out_surface == surface
    assert not (tmp_path / "auto-surgery-tissue-warp-volume.obj").exists()


def test_scale_perturbation_scales_vertices_and_writes_obj(tmp_path: Path) -> None:
    volume = tmp_path / "volume.obj"
    surface = tmp_path / "surface.obj"
    _write_tetra_mesh(volume)
    _write_tetra_mesh(surface)

    perturb = MeshPerturbation(scale=(2.0, 3.0, 4.0))
    out_volume, out_surface = warp_tissue_meshes(volume, surface, perturb)

    volume_before = _read_vertices(volume)
    volume_after = _read_vertices(out_volume)
    assert np.allclose(volume_after, volume_before * np.array([2.0, 3.0, 4.0]))
    assert out_volume.name.startswith("auto-surgery-tissue-warp-")
    assert out_surface.name.startswith("auto-surgery-tissue-warp-")
    assert out_volume.suffix == ".obj"
    assert out_surface.suffix == ".obj"


def test_bulge_alters_vertices_and_is_local(tmp_path: Path) -> None:
    volume = tmp_path / "volume.obj"
    surface = tmp_path / "surface.obj"
    _write_tetra_mesh(volume)
    _write_tetra_mesh(surface)

    local = MeshPerturbation(
        bulge=BulgeSpec(
            center_scene=Vec3(x=0.15, y=0.15, z=0.15),
            radius_scene=2.0,
            amplitude_scene=0.2,
        ),
    )
    out_volume, _ = warp_tissue_meshes(volume, surface, local)
    volume_before = _read_vertices(volume)
    volume_after = _read_vertices(out_volume)
    deltas = np.linalg.norm(volume_after - volume_before, axis=1)
    assert deltas.max() > 1e-6

    distant = MeshPerturbation(
        bulge=BulgeSpec(
            center_scene=Vec3(x=10_000.0, y=10_000.0, z=10_000.0),
            radius_scene=0.1,
            amplitude_scene=1.0,
        ),
    )
    out_volume_far, _ = warp_tissue_meshes(volume, surface, distant)
    volume_far = _read_vertices(out_volume_far)
    assert np.allclose(volume_far, volume_before, atol=1e-8)
