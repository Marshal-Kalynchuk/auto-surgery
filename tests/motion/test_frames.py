from __future__ import annotations

import pytest

from auto_surgery.motion.frames import (
    TwistCamera,
    TwistSceneShaft,
    TwistSceneTip,
    to_scene_shaft,
    to_scene_tip,
)
from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3


def _identity_twist() -> Twist:
    return Twist(
        linear=Vec3(x=0.01, y=-0.02, z=0.03),
        angular=Vec3(x=0.04, y=-0.05, z=0.06),
    )


def _identity_pose() -> Pose:
    return Pose(
        position=Vec3(x=0.0, y=0.0, z=0.0),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )


def _expect_scene_tip(twist: TwistSceneTip) -> Twist:
    if not isinstance(twist, TwistSceneTip):
        raise TypeError("Expected TwistSceneTip for this path.")
    return twist.data()


def test_frame_newtypes_are_distinct() -> None:
    twist = _identity_twist()

    twist_camera = TwistCamera(twist)
    twist_scene_tip = TwistSceneTip(twist)
    twist_scene_shaft = TwistSceneShaft(twist)

    assert type(twist_camera) is not type(twist_scene_tip)
    assert type(twist_scene_tip) is not type(twist_scene_shaft)
    assert type(twist_camera) is not type(twist_scene_shaft)

    assert _expect_scene_tip(twist_scene_tip) == twist
    assert isinstance(twist_camera, TwistCamera)
    assert isinstance(twist_scene_tip, TwistSceneTip)
    assert isinstance(twist_scene_shaft, TwistSceneShaft)

    with pytest.raises(TypeError):
        _expect_scene_tip(TwistCamera(twist))


def test_frame_helpers_are_callable() -> None:
    twist = _identity_twist()
    camera_pose = _identity_pose()
    dof_pose = _identity_pose()

    twist_camera = TwistCamera(twist)
    twist_scene_tip = to_scene_tip(twist_camera, camera_pose)
    twist_scene_shaft = to_scene_shaft(
        twist_scene_tip,
        dof_pose=dof_pose,
        tip_offset_local=Vec3(x=0.0, y=0.0, z=0.0),
    )

    assert isinstance(twist_scene_tip, TwistSceneTip)
    assert isinstance(twist_scene_shaft, TwistSceneShaft)
    assert twist_scene_tip.data() == twist
    assert twist_scene_shaft.data() == twist
