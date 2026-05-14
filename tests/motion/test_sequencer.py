from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from auto_surgery.motion.primitives import ContactReach, Hold, Reach
from auto_surgery.motion.sequencer import MotionGeneratorConfig, PrimitiveKind, Sequencer, _EPS, _sample_point_in_volume
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.commands import Twist
from auto_surgery.schemas.scene import SceneConfig, TargetVolume
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState


def _identity_pose() -> Pose:
    return Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _step() -> StepResult:
    return StepResult(
        sensors=SensorBundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            tool=ToolState(
                pose=_identity_pose(),
                twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
                jaw=0.0,
                wrench=Vec3(x=0.0, y=0.0, z=0.0),
                in_contact=False,
            ),
            cameras=[
                CameraView(
                    camera_id="cam",
                    timestamp_ns=0,
                    extrinsics=_identity_pose(),
                    intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=640, height=480),
                )
            ],
            safety=SafetyStatus(motion_enabled=True, command_blocked=False, block_reason=None, cycle_id_echo=0),
        ),
        dt=0.0,
        sim_step_index=0,
        is_capture_tick=True,
    )


def _scene() -> SceneConfig:
    return SceneConfig(tissue_scene_path="")


def _sequencer_config(seed: int = 0, **overrides: Any) -> MotionGeneratorConfig:
    base = MotionGeneratorConfig(
        seed=seed,
        primitive_count_min=6,
        primitive_count_max=6,
        reach_duration_range_s=(0.4, 0.4),
        hold_duration_range_s=(0.25, 0.25),
        drag_duration_range_s=(0.2, 0.2),
        drag_distance_range_mm=(4.0, 4.0),
        brush_duration_range_s=(0.3, 0.3),
        brush_arc_range_rad=(0.2, 0.2),
        rotate_duration_range_s=(0.3, 0.3),
        rotate_angle_range_rad=(0.2, 0.2),
        probe_duration_range_s=(0.2, 0.2),
        probe_hold_range_s=(0.1, 0.1),
        target_orientation_jitter_rad=0.0,
        jaw_value_range=(0.0, 1.0),
        jaw_change_probability=0.0,
        weight_reach=1.0,
        weight_hold=1.0,
        weight_drag=1.0,
        weight_brush=1.0,
        weight_grip=1.0,
        weight_contact_reach=1.0,
    )
    return MotionGeneratorConfig(**{**base.__dict__, **overrides})


def test_sequencer_is_deterministic_with_shared_seed() -> None:
    cfg = _sequencer_config(123)
    scene = _scene()
    step = _step()

    left = Sequencer(cfg, scene)
    right = Sequencer(cfg, scene)
    left.reset(step)
    right.reset(step)

    left_seq = [type(left.next_primitive(step, 0.0)).__name__ for _ in range(6)]
    right_seq = [type(right.next_primitive(step, 0.0)).__name__ for _ in range(6)]
    assert left_seq == right_seq


def test_first_primitive_targets_first_volume_center_at_midpoint_duration() -> None:
    first_center = Vec3(x=0.12, y=0.34, z=-0.02)
    volumes = [
        TargetVolume(
            label="tumor",
            center_scene=first_center,
            half_extents_scene=Vec3(x=0.0, y=0.0, z=0.0),
            shape="sphere",
        )
    ]
    cfg = _sequencer_config(seed=2)
    scene = _scene()
    scene = SceneConfig(tissue_scene_path=scene.tissue_scene_path, target_volumes=volumes)
    seq = Sequencer(cfg, scene)
    seq.reset(_step())

    first = seq.next_primitive(_step(), 0.0)
    assert isinstance(first, Reach)
    assert first.target_pose_scene.position == first_center
    assert first.duration_s == pytest.approx(0.4)


def test_next_primitive_is_lazy_for_first_output() -> None:
    cfg = _sequencer_config(seed=3)
    seq = Sequencer(cfg, _scene())
    seq.reset(_step())

    with patch.object(Sequencer, "_sample_kind") as sample_kind:
        first = seq.next_primitive(_step(), 0.0)
        assert sample_kind.call_count == 0
        assert first is not None

        sample_kind.return_value = PrimitiveKind.HOLD
        second = seq.next_primitive(_step(), 0.0)
        assert sample_kind.call_count == 1
        assert second is not None
        assert isinstance(second, Hold)


def test_next_primitive_respects_zero_length_plan() -> None:
    cfg = _sequencer_config(seed=12, primitive_count_min=0, primitive_count_max=0)
    seq = Sequencer(cfg, _scene())
    seq.reset(_step())

    assert seq.next_primitive(_step(), 0.0) is None
    assert seq.next_primitive(_step(), 0.0) is None


def test_next_primitive_skips_first_rule_when_plan_empty() -> None:
    cfg = _sequencer_config(seed=12, primitive_count_min=0, primitive_count_max=0)
    seq = Sequencer(cfg, _scene())
    seq.reset(_step())

    with patch.object(Sequencer, "_build_first_approach") as build_first_approach, patch.object(
        Sequencer, "_sample_kind"
    ) as sample_kind:
        assert seq.next_primitive(_step(), 0.0) is None
        assert build_first_approach.call_count == 0
        assert sample_kind.call_count == 0


def test_contact_reach_duration_matches_reach_range() -> None:
    cfg = _sequencer_config(
        seed=4,
        weight_reach=0.0,
        weight_hold=0.0,
        weight_drag=0.0,
        weight_brush=0.0,
        weight_grip=0.0,
        weight_contact_reach=1.0,
    )

    seq = Sequencer(cfg, _scene())
    seq.reset(_step())

    seq.next_primitive(_step(), 0.0)
    contact = seq.next_primitive(_step(), 0.0)
    assert contact is not None
    assert isinstance(contact, ContactReach)
    assert contact.duration_s == pytest.approx(cfg.reach_duration_range_s[0])


def test_sweep_axis_bias_keeps_camera_z_within_scale() -> None:
    cfg = _sequencer_config(seed=7, sweep_axis_bias_scale=0.2)
    seq = Sequencer(cfg, _scene())
    seq.reset(_step())

    for _ in range(200):
        axis = seq._sample_sweep_axis()
        assert axis.shape == (3,)
        assert abs(axis[2]) <= cfg.sweep_axis_bias_scale + _EPS
        assert np.isclose(float(np.linalg.norm(axis)), 1.0, atol=1e-6)


def test_sample_point_in_volume_handles_sphere_and_bbox() -> None:
    rng = np.random.default_rng(0)
    center = np.array([1.0, -2.0, 0.5], dtype=float)

    for _ in range(200):
        point = _sample_point_in_volume(
            rng=rng,
            shape="sphere",
            center=center,
            half_extents=np.array([0.05, 0.0, 0.0], dtype=float),
        )
        assert np.linalg.norm(point - center) <= 0.050001

    for _ in range(200):
        point = _sample_point_in_volume(
            rng=rng,
            shape="bbox",
            center=center,
            half_extents=np.array([0.03, 0.04, 0.05], dtype=float),
        )
        delta = point - center
        assert abs(delta[0]) <= 0.030001
        assert abs(delta[1]) <= 0.040001
        assert abs(delta[2]) <= 0.050001
