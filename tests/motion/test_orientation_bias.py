from __future__ import annotations

from pathlib import Path

import pytest

from auto_surgery.motion.generator import SurgicalMotionGenerator
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.motion import MotionGeneratorConfig, MotionShaping
from auto_surgery.schemas.scene import OrientationBias, SceneConfig, ToolSpec


def _motion_config(
    *,
    orientation_bias_gain: float = 1.0,
    orientation_deadband_rad: float = 0.0,
) -> MotionGeneratorConfig:
    return MotionGeneratorConfig(
        seed=0,
        primitive_count_min=0,
        primitive_count_max=0,
        motion_shaping=MotionShaping(
            max_linear_m_s=1.0,
            max_angular_rad_s=1.0,
            max_linear_accel_m_s2=1.0,
            max_angular_accel_rad_s2=1.0,
            bias_gain_max=0.0,
            bias_ramp_distance_m=1.0,
            orientation_bias_gain=orientation_bias_gain,
            orientation_deadband_rad=orientation_deadband_rad,
        ),
    )


def _scene_config(
    *,
    tool_gain: float = 1.0,
    surface_normal_blend: float = 1.0,
) -> SceneConfig:
    return SceneConfig(
        tissue_scene_path=Path("test_tissue.obj"),
        tool=ToolSpec(
            orientation_bias=OrientationBias(
                forward_axis_local=Vec3(x=0.0, y=0.0, z=1.0),
                surface_normal_blend=surface_normal_blend,
                gain=tool_gain,
                deadband_rad=0.0,
            )
        ),
    )


class _SurfacePoint:
    def __init__(self, position: Vec3):
        self.position = position


class _FakeSceneGeometry:
    def __init__(self, closest_surface_position: Vec3):
        self._closest_surface_position = closest_surface_position

    def bounds(self) -> tuple[Vec3, Vec3]:
        return Vec3(x=-1.0, y=-1.0, z=-1.0), Vec3(x=1.0, y=1.0, z=1.0)

    def closest_surface_point(self, _tip_position_scene: Vec3) -> _SurfacePoint:
        return _SurfacePoint(position=self._closest_surface_position)


class _Primitive:
    def __init__(self, allow_orientation_bias: bool):
        self.allow_orientation_bias = allow_orientation_bias


def _identity_pose() -> Pose:
    return Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _mk_generator(*, geometry: _FakeSceneGeometry, motion: MotionGeneratorConfig) -> SurgicalMotionGenerator:
    generator = SurgicalMotionGenerator(motion, _scene_config())
    generator._sequencer._scene_geometry = geometry  # type: ignore[attr-defined]
    return generator


def test_orientation_bias_drives_axis_toward_surface_direction() -> None:
    generator = _mk_generator(
        geometry=_FakeSceneGeometry(Vec3(x=1.0, y=0.0, z=0.0)),
        motion=_motion_config(
            orientation_bias_gain=1.0,
            orientation_deadband_rad=0.0,
        ),
    )
    raw_angular = [0.0, 0.0, 0.0]
    final, biased = generator._compute_orientation_bias(
        raw_angular=raw_angular,
        tip_pose_scene=_identity_pose(),
        primitive=_Primitive(allow_orientation_bias=True),
        in_contact=False,
    )

    assert biased is True
    assert final[1] > 0.0
    assert final[0] == pytest.approx(0.0, abs=1.0e-12)
    assert final[2] == pytest.approx(0.0, abs=1.0e-12)


def test_orientation_bias_deadband_blocks_small_angle() -> None:
    generator = _mk_generator(
        geometry=_FakeSceneGeometry(Vec3(x=0.05, y=0.0, z=0.99875)),
        motion=_motion_config(
            orientation_bias_gain=1.0,
            orientation_deadband_rad=0.2,
        ),
    )
    raw_angular = [0.12, -0.21, 0.33]
    final, biased = generator._compute_orientation_bias(
        raw_angular=raw_angular,
        tip_pose_scene=_identity_pose(),
        primitive=_Primitive(allow_orientation_bias=True),
        in_contact=False,
    )
    assert biased is False
    assert final == pytest.approx(raw_angular)


def test_orientation_bias_suppressed_during_contact_or_forbidden_primitive() -> None:
    generator = _mk_generator(
        geometry=_FakeSceneGeometry(Vec3(x=1.0, y=0.0, z=0.0)),
        motion=_motion_config(
            orientation_bias_gain=1.0,
            orientation_deadband_rad=0.0,
        ),
    )
    raw_angular = [0.12, -0.21, 0.33]

    in_contact_bias, in_contact_active = generator._compute_orientation_bias(
        raw_angular=raw_angular,
        tip_pose_scene=_identity_pose(),
        primitive=_Primitive(allow_orientation_bias=True),
        in_contact=True,
    )
    assert in_contact_active is False
    assert in_contact_bias == pytest.approx(raw_angular)

    no_primitive_flag, primitive_active = generator._compute_orientation_bias(
        raw_angular=raw_angular,
        tip_pose_scene=_identity_pose(),
        primitive=_Primitive(allow_orientation_bias=False),
        in_contact=False,
    )
    assert primitive_active is False
    assert no_primitive_flag == pytest.approx(raw_angular)
