"""Tests for scene-frame target orientation sampling in ``Sequencer._sample_target_pose``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from auto_surgery.motion.frames import quat_rotate_vec3
from auto_surgery.motion.sequencer import Sequencer
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.motion import MotionGeneratorConfig, MotionShaping
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import OrientationBias, SceneConfig, ToolSpec
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState, Twist


def _identity_pose() -> Pose:
    return Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _step() -> StepResult:
    pose = _identity_pose()
    return StepResult(
        sensors=SensorBundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            tool=ToolState(
                pose=pose,
                twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
                jaw=0.0,
                wrench=Vec3(x=0.0, y=0.0, z=0.0),
                in_contact=False,
            ),
            cameras=[
                CameraView(
                    camera_id="cam",
                    timestamp_ns=0,
                    extrinsics=pose,
                    intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=640, height=480),
                )
            ],
            safety=SafetyStatus(
                motion_enabled=True,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=0,
            ),
        ),
        dt=0.05,
        sim_step_index=0,
        is_capture_tick=True,
    )


class _FakeSurfacePoint:
    def __init__(self, position: Vec3, normal: Vec3) -> None:
        self.position = position
        self.normal = normal


class _FakeSceneGeometry:
    def closest_surface_point(self, _tip_position_scene: Vec3) -> _FakeSurfacePoint:
        return _FakeSurfacePoint(
            position=Vec3(x=1.0, y=0.0, z=0.0),
            normal=Vec3(x=1.0, y=0.0, z=0.0),
        )

    def bounds(self) -> tuple[Vec3, Vec3]:
        return Vec3(x=-1.0, y=-1.0, z=-1.0), Vec3(x=1.0, y=1.0, z=1.0)


def _motion_config(*, jitter_rad: float = 0.0) -> MotionGeneratorConfig:
    return MotionGeneratorConfig(
        seed=0,
        primitive_count_min=0,
        primitive_count_max=0,
        motion_shaping=MotionShaping(
            max_linear_mm_s=1.0,
            max_angular_rad_s=1.0,
            max_linear_accel_mm_s2=1.0,
            max_angular_accel_rad_s2=1.0,
            bias_gain_max=0.0,
            bias_ramp_distance_mm=1.0,
            orientation_bias_gain=0.0,
            orientation_deadband_rad=0.0,
        ),
        target_orientation_jitter_rad=jitter_rad,
    )


def _scene_config(*, surface_normal_blend: float) -> SceneConfig:
    return SceneConfig(
        tissue_scene_path=Path("test_tissue.obj"),
        tool=ToolSpec(
            orientation_bias=OrientationBias(
                forward_axis_local=Vec3(x=0.0, y=0.0, z=1.0),
                surface_normal_blend=surface_normal_blend,
                gain=0.0,
                deadband_rad=0.0,
            )
        ),
    )


def _unit_forward(rotation: Quaternion) -> np.ndarray:
    forward = quat_rotate_vec3(rotation, Vec3(x=0.0, y=0.0, z=1.0))
    v = np.array([float(forward.x), float(forward.y), float(forward.z)], dtype=float)
    n = float(np.linalg.norm(v))
    assert n > 1e-9
    return v / n


def test_surface_normal_blend_one_pulls_forward_toward_surface_normal_more_than_mid_blend() -> None:
    """Higher ``surface_normal_blend`` increases alignment of tool +Z with the sampled surface normal."""
    step = _step()
    start = _identity_pose()
    seq_full = Sequencer(
        _motion_config(jitter_rad=0.0),
        _scene_config(surface_normal_blend=1.0),
        scene_geometry=_FakeSceneGeometry(),
    )
    seq_mid = Sequencer(
        _motion_config(jitter_rad=0.0),
        _scene_config(surface_normal_blend=0.5),
        scene_geometry=_FakeSceneGeometry(),
    )
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    f_full = _unit_forward(seq_full._sample_target_pose(step, start).rotation)
    f_mid = _unit_forward(seq_mid._sample_target_pose(step, start).rotation)
    assert float(np.dot(f_full, ex)) > float(np.dot(f_mid, ex))


def test_surface_normal_blend_zero_vs_one_changes_sampled_forward() -> None:
    step = _step()
    start = _identity_pose()
    seq_full = Sequencer(
        _motion_config(jitter_rad=0.0),
        _scene_config(surface_normal_blend=1.0),
        scene_geometry=_FakeSceneGeometry(),
    )
    seq_none = Sequencer(
        _motion_config(jitter_rad=0.0),
        _scene_config(surface_normal_blend=0.0),
        scene_geometry=_FakeSceneGeometry(),
    )
    f_full = _unit_forward(seq_full._sample_target_pose(step, start).rotation)
    f_none = _unit_forward(seq_none._sample_target_pose(step, start).rotation)
    assert float(np.dot(f_full, f_none)) < 0.999


def test_target_orientation_jitter_changes_rotation_at_same_seed() -> None:
    step = _step()
    start = _identity_pose()
    seq0 = Sequencer(_motion_config(jitter_rad=0.0), _scene_config(surface_normal_blend=1.0), scene_geometry=_FakeSceneGeometry())
    seq1 = Sequencer(_motion_config(jitter_rad=0.5), _scene_config(surface_normal_blend=1.0), scene_geometry=_FakeSceneGeometry())
    r0 = seq0._sample_target_pose(step, start).rotation
    r1 = seq1._sample_target_pose(step, start).rotation
    assert (r0.w, r0.x, r0.y, r0.z) != pytest.approx((r1.w, r1.x, r1.y, r1.z), abs=1e-9)
