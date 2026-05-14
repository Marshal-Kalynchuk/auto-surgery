from __future__ import annotations

import pytest

from auto_surgery.motion import SurgicalMotionGenerator
from auto_surgery.motion.generator import MotionGeneratorConfig, RealisedPrimitive
from auto_surgery.schemas.commands import ControlFrame, ControlMode, Quaternion, Pose, Twist, Vec3
from auto_surgery.schemas.scene import SceneConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState


def _identity_pose() -> Pose:
    return Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _step(
    *,
    sim_step_index: int,
    dt: float,
    in_contact: bool = False,
) -> StepResult:
    cam_pose = _identity_pose()
    bundle = SensorBundle(
        timestamp_ns=sim_step_index * 1_000_000_000,
        sim_time_s=sim_step_index * dt,
        tool=ToolState(
            pose=cam_pose,
            twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
            jaw=0.2,
            wrench=Vec3(x=0.0, y=0.0, z=0.0),
            in_contact=in_contact,
        ),
        cameras=[
            CameraView(
                camera_id="cam",
                timestamp_ns=sim_step_index * 1_000_000_000,
                extrinsics=cam_pose,
                intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=640, height=480),
            ),
        ],
        safety=SafetyStatus(
            motion_enabled=True,
            command_blocked=False,
            block_reason=None,
            cycle_id_echo=sim_step_index,
        ),
    )
    return StepResult(sensors=bundle, dt=dt, sim_step_index=sim_step_index, is_capture_tick=True)


def _scene() -> SceneConfig:
    return SceneConfig(tissue_scene_path="")


def _compact_config() -> MotionGeneratorConfig:
    return MotionGeneratorConfig(
        seed=0,
        primitive_count_min=1,
        primitive_count_max=1,
        reach_duration_range_s=(0.05, 0.05),
        hold_duration_range_s=(0.25, 0.25),
        drag_duration_range_s=(0.15, 0.15),
        drag_distance_range_mm=(0.0, 0.0),
        brush_duration_range_s=(0.20, 0.20),
        brush_arc_range_rad=(0.2, 0.2),
        rotate_duration_range_s=(0.2, 0.2),
        rotate_angle_range_rad=(0.2, 0.2),
        probe_duration_range_s=(0.2, 0.2),
        probe_hold_range_s=(0.1, 0.1),
        target_orientation_jitter_rad=0.0,
        jaw_value_range=(0.0, 1.0),
        jaw_change_probability=0.0,
        weight_reach=1.0,
        weight_hold=0.0,
        weight_drag=0.0,
        weight_brush=0.0,
        weight_grip=0.0,
        weight_contact_reach=0.0,
    )


def test_next_command_requires_reset() -> None:
    generator = SurgicalMotionGenerator(_compact_config(), _scene())
    with pytest.raises(RuntimeError, match="reset\\(\\) must be called"):
        generator.next_command(_step(sim_step_index=0, dt=0.0))


def test_generator_emits_scripted_camera_twist_commands() -> None:
    generator = SurgicalMotionGenerator(_compact_config(), _scene())
    initial_step = _step(sim_step_index=0, dt=0.0)
    first_cmd = generator.reset(initial_step)
    second_cmd = generator.next_command(_step(sim_step_index=1, dt=0.05))

    assert first_cmd.cycle_id == 0
    assert second_cmd.cycle_id == 1
    assert first_cmd.control_mode == ControlMode.CARTESIAN_TWIST
    assert first_cmd.frame == ControlFrame.CAMERA
    assert first_cmd.enable is True
    assert first_cmd.source == "scripted"
    assert first_cmd.cartesian_twist is not None


def test_finalize_is_idempotent_and_records_realised_only_once() -> None:
    generator = SurgicalMotionGenerator(_compact_config(), _scene())
    step0 = _step(sim_step_index=0, dt=0.0)
    generator.reset(step0)
    generator.next_command(_step(sim_step_index=1, dt=0.05))
    generator.finalize(_step(sim_step_index=2, dt=0.05))
    generator.finalize(_step(sim_step_index=3, dt=0.05))

    assert len(generator.realised_sequence) == 1
    realized = generator.realised_sequence[0]
    assert isinstance(realized, RealisedPrimitive)
    assert realized.started_at_tick == 0


def test_zero_length_plan_results_in_immediate_dwell() -> None:
    cfg = _compact_config().model_copy(update={"primitive_count_min": 0, "primitive_count_max": 0})
    generator = SurgicalMotionGenerator(cfg, _scene())
    generator.reset(_step(sim_step_index=0, dt=0.0))

    second = generator.next_command(_step(sim_step_index=1, dt=0.05))
    third = generator.next_command(_step(sim_step_index=2, dt=0.05))

    assert generator.realised_sequence == ()
    assert second.cartesian_twist == third.cartesian_twist


def test_realised_sequence_tracks_completed_primitives() -> None:
    generator = SurgicalMotionGenerator(_compact_config(), _scene())
    generator.reset(_step(sim_step_index=0, dt=0.0))

    first = _step(sim_step_index=1, dt=0.05)
    second = _step(sim_step_index=2, dt=0.05)
    _ = generator.next_command(first)
    _ = generator.next_command(second)
    seq = generator.realised_sequence

    assert len(seq) == 1
    completed = seq[0]
    assert completed.started_at_tick == 0
    assert completed.ended_at_tick == 2
    assert completed.early_terminated is False
