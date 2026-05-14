from __future__ import annotations

from pathlib import Path

import pytest

from auto_surgery.schemas.commands import Vec3
from auto_surgery.schemas.scene import (
    OrientationBias,
    SceneGeometryEnvelope,
    SphereEnvelope,
    ToolSpec,
    WorkspaceEnvelope,
)


def test_workspace_envelope_signed_distance_to_envelope_is_callable() -> None:
    scene_envelope = SceneGeometryEnvelope(
        outer_margin_mm=0.03,
        inner_margin_mm=0.01,
        no_go_regions=[],
    )
    sphere_envelope = SphereEnvelope(
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        radius_mm=0.5,
        outer_margin_mm=0.03,
        inner_margin_mm=0.01,
    )

    assert scene_envelope.signed_distance_to_envelope(Vec3(x=0.2, y=0.0, z=0.0)) == 0.0
    assert (
        sphere_envelope.signed_distance_to_envelope(Vec3(x=0.5, y=0.0, z=0.0))
        == 0.0
    )


def test_tool_spec_accepts_workspace_envelopes() -> None:
    scene_envelope: WorkspaceEnvelope = SceneGeometryEnvelope(
        outer_margin_mm=0.03,
        inner_margin_mm=0.01,
        no_go_regions=[],
    )
    sphere_envelope: WorkspaceEnvelope = SphereEnvelope(
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        radius_mm=0.2,
        outer_margin_mm=0.03,
        inner_margin_mm=0.01,
    )

    tool_with_scene = ToolSpec(workspace_envelope=scene_envelope)
    tool_with_sphere = ToolSpec(workspace_envelope=sphere_envelope)

    assert isinstance(tool_with_scene.workspace_envelope, SceneGeometryEnvelope)
    assert isinstance(tool_with_sphere.workspace_envelope, SphereEnvelope)
    assert tool_with_scene.orientation_bias.forward_axis_local == Vec3(x=0.0, y=0.0, z=1.0)
    assert tool_with_scene.orientation_bias.surface_normal_blend == 0.7
    assert tool_with_sphere.orientation_bias.deadband_rad >= 0.0
    assert OrientationBias().gain == 0.0


def test_scene_envelope_inside_aabb_returns_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_surgery.env import scene_geometry as scene_geom_mod

    class _FakeGeom:
        def bounds(self) -> tuple[Vec3, Vec3]:
            return Vec3(x=0.0, y=0.0, z=0.0), Vec3(x=10.0, y=10.0, z=10.0)

    monkeypatch.setattr(scene_geom_mod, "MeshSceneGeometry", lambda _path: _FakeGeom())

    env = SceneGeometryEnvelope(
        surface_mesh_path=Path("/fake.obj"),
        outer_margin_mm=1.0,
        inner_margin_mm=0.0,
        no_go_regions=[],
    )
    d = env.signed_distance_to_envelope(Vec3(x=5.0, y=5.0, z=5.0))
    assert d == pytest.approx(6.0)


def test_scene_envelope_outside_aabb_returns_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_surgery.env import scene_geometry as scene_geom_mod

    class _FakeGeom:
        def bounds(self) -> tuple[Vec3, Vec3]:
            return Vec3(x=0.0, y=0.0, z=0.0), Vec3(x=1.0, y=1.0, z=1.0)

    monkeypatch.setattr(scene_geom_mod, "MeshSceneGeometry", lambda _path: _FakeGeom())

    env = SceneGeometryEnvelope(
        surface_mesh_path=Path("/fake.obj"),
        outer_margin_mm=0.01,
        inner_margin_mm=0.0,
        no_go_regions=[],
    )
    assert env.signed_distance_to_envelope(Vec3(x=10.0, y=0.5, z=0.5)) < 0.0


def test_scene_envelope_outer_margin_inflates_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_surgery.env import scene_geometry as scene_geom_mod

    class _FakeGeom:
        def bounds(self) -> tuple[Vec3, Vec3]:
            return Vec3(x=0.0, y=0.0, z=0.0), Vec3(x=10.0, y=10.0, z=10.0)

    monkeypatch.setattr(scene_geom_mod, "MeshSceneGeometry", lambda _path: _FakeGeom())

    tight = SceneGeometryEnvelope(
        surface_mesh_path=Path("/fake.obj"),
        outer_margin_mm=0.01,
        inner_margin_mm=0.0,
        no_go_regions=[],
    )
    loose = SceneGeometryEnvelope(
        surface_mesh_path=Path("/fake.obj"),
        outer_margin_mm=5.0,
        inner_margin_mm=0.0,
        no_go_regions=[],
    )
    p = Vec3(x=5.0, y=5.0, z=5.0)
    assert loose.signed_distance_to_envelope(p) > tight.signed_distance_to_envelope(p)


def test_scene_geometry_envelope_model_dump_omits_private_geometry(tmp_path: Path) -> None:
    mesh = tmp_path / "m.obj"
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    env = SceneGeometryEnvelope(
        surface_mesh_path=mesh,
        outer_margin_mm=0.01,
        inner_margin_mm=0.0,
        no_go_regions=[],
    )
    dumped = env.model_dump(mode="python")
    assert "_mesh_geometry" not in dumped
    assert "surface_mesh_path" in dumped
