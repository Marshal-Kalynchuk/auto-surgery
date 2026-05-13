"""Typed twist wrappers that make coordinate frames explicit.

The motion stack mixes raw :class:`Twist` values in multiple coordinate frames
today.  This module introduces lightweight frame-named wrappers so callers can
encode the contract at the type level before refactoring all call sites.

Current contracts:
- ``TwistCamera`` values represent camera-frame tool-tip twists.
- ``TwistSceneTip`` values represent scene-frame tool-tip twists.
- ``TwistSceneShaft`` values represent scene-frame shaft-origin twists.

For this phase, conversions are intentionally minimal placeholders. The
signatures and distinct wrapper types are established now, and conversion bodies
can be replaced by real adjoint transforms in later sections.
"""

from __future__ import annotations

from dataclasses import dataclass

from auto_surgery.schemas.commands import Pose, Twist, Vec3


@dataclass(frozen=True)
class _TwistFrame:
    """Common storage for a wrapped twist value."""

    _twist: Twist

    def data(self) -> Twist:
        return self._twist


class TwistCamera(_TwistFrame):
    """Velocity command expressed in the camera frame at the tool tip."""


class TwistSceneTip(_TwistFrame):
    """Velocity command expressed at the tool tip in scene coordinates."""


class TwistSceneShaft(_TwistFrame):
    """Velocity command expressed at the shaft origin in scene coordinates."""


def to_scene_tip(twist_camera: TwistCamera, camera_pose: Pose) -> TwistSceneTip:
    """Convert ``TwistCamera`` into ``TwistSceneTip``.

    Args:
        twist_camera: Twist reported in the camera frame.
        camera_pose: Camera pose in the scene frame.

    Returns:
        Wrapped scene-tip twist.

    This is currently a placeholder; the frame transform logic will be wired in the
    follow-up section that refactors call sites.
    """

    _ = camera_pose
    return TwistSceneTip(twist_camera.data())


def to_scene_shaft(
    twist_scene_tip: TwistSceneTip,
    dof_pose: Pose,
    tip_offset_local: Vec3,
) -> TwistSceneShaft:
    """Convert ``TwistSceneTip`` into ``TwistSceneShaft``.

    Args:
        twist_scene_tip: Twist at the tip, in scene coordinates.
        dof_pose: Dof pose in scene coordinates.
        tip_offset_local: Tool-tip offset in the local shaft frame.

    Returns:
        Wrapped scene-shaft twist.

    This is currently a placeholder; shaft-tip offset kinematics will be added when
    the section refactors this path.
    """

    _ = dof_pose
    _ = tip_offset_local
    return TwistSceneShaft(twist_scene_tip.data())
