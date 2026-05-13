"""Tests for the scene geometry abstraction."""

from __future__ import annotations

import pytest

from auto_surgery.env.scene_geometry import RayHit, SphereSceneGeometry, SurfacePoint
from auto_surgery.schemas.commands import Vec3


def _norm(vec: Vec3) -> float:
    return float((vec.x ** 2 + vec.y ** 2 + vec.z ** 2) ** 0.5)


def test_sphere_scene_geometry_methods() -> None:
    geometry = SphereSceneGeometry(center=Vec3(x=0.0, y=0.0, z=0.0), radius=2.0)

    closest = geometry.closest_surface_point(Vec3(x=3.0, y=0.0, z=0.0))
    assert isinstance(closest, SurfacePoint)
    assert closest.position == Vec3(x=2.0, y=0.0, z=0.0)
    assert closest.normal == Vec3(x=1.0, y=0.0, z=0.0)
    assert closest.signed_distance == pytest.approx(1.0, abs=1.0e-12)
    assert _norm(closest.normal) == pytest.approx(1.0, rel=1.0e-12)

    assert geometry.signed_distance(Vec3(x=3.0, y=0.0, z=0.0)) == pytest.approx(1.0)
    assert geometry.signed_distance(Vec3(x=1.0, y=0.0, z=0.0)) < 0.0
    assert geometry.signed_distance(Vec3(x=0.0, y=0.0, z=0.0)) == pytest.approx(-2.0)

    hit = geometry.ray_cast(
        origin=Vec3(x=-5.0, y=0.0, z=0.0),
        direction=Vec3(x=1.0, y=0.0, z=0.0),
        max_distance_m=10.0,
    )
    assert isinstance(hit, RayHit)
    assert hit.distance == pytest.approx(3.0, abs=1.0e-12)
    assert hit.position == Vec3(x=-2.0, y=0.0, z=0.0)

    assert geometry.ray_cast(
        origin=Vec3(x=-5.0, y=0.0, z=0.0),
        direction=Vec3(x=1.0, y=0.0, z=0.0),
        max_distance_m=1.0,
    ) is None

    bounds_min, bounds_max = geometry.bounds()
    assert bounds_min == Vec3(x=-2.0, y=-2.0, z=-2.0)
    assert bounds_max == Vec3(x=2.0, y=2.0, z=2.0)
