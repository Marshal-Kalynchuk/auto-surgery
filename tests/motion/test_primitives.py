from __future__ import annotations

import pytest

from auto_surgery.motion.profile import min_jerk_retract_duration
from auto_surgery.motion.primitives import ActivePrimitive, Approach, Dwell, PrimitiveOutput, Probe, evaluate
from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState


def _identity_pose() -> Pose:
    return Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _offset_pose(*, x: float, y: float, z: float) -> Pose:
    return Pose(position=Vec3(x=x, y=y, z=z), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _step(*, sim_step_index: int = 0, dt: float = 0.0, camera_position: Vec3 | None = None, in_contact: bool = False) -> StepResult:
    if camera_position is None:
        camera_position = Vec3(x=0.0, y=0.0, z=0.0)
    cam_pose = Pose(position=camera_position, rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    bundle = SensorBundle(
        timestamp_ns=sim_step_index,
        sim_time_s=0.0,
        tool=ToolState(
            pose=_identity_pose(),
            twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
            jaw=0.0,
            wrench=Vec3(x=0.0, y=0.0, z=0.0),
            in_contact=in_contact,
        ),
        cameras=[
            CameraView(
                camera_id="cam",
                timestamp_ns=sim_step_index,
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


def test_primitive_dataclasses_are_frozen() -> None:
    approach = Approach(
        target_pose_scene=_identity_pose(),
        duration_s=1.0,
        jaw_target_start=0.0,
        jaw_target_end=1.0,
    )

    with pytest.raises(AttributeError):
        approach.duration_s = 2.0


def test_evaluate_approach_produces_minimum_jerk_twist() -> None:
    active = ActivePrimitive(
        primitive=Approach(
            target_pose_scene=_offset_pose(x=1.0, y=0.0, z=0.0),
            duration_s=1.0,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.5,
    )
    output = evaluate(active, _step(sim_step_index=1, dt=0.1))
    assert output.twist_camera.linear.x == pytest.approx(1.875, abs=1e-3)
    assert output.is_finished is False


def test_evaluate_approach_uses_camera_extrinsics_for_scene_geometry() -> None:
    step = _step(sim_step_index=1, dt=0.1, camera_position=Vec3(x=1.0, y=0.0, z=0.0))
    active = ActivePrimitive(
        primitive=Approach(
            target_pose_scene=_offset_pose(x=2.0, y=0.0, z=0.0),
            duration_s=1.0,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        started_at_pose_scene=_offset_pose(x=0.5, y=0.0, z=0.0),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.5,
    )
    output = evaluate(active, step)
    assert output.twist_camera.linear.x == pytest.approx(0.9375, abs=1e-3)


def test_jaw_target_interpolates_with_min_jerk_profile() -> None:
    primitive = Approach(
        target_pose_scene=_identity_pose(),
        duration_s=2.0,
        jaw_target_start=0.2,
        jaw_target_end=0.8,
    )
    active = ActivePrimitive(
        primitive=primitive,
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=2.0,
        elapsed_s=1.0,
    )
    output = evaluate(active, _step(sim_step_index=1, dt=0.1))
    assert output.jaw_target == pytest.approx(0.5)


def test_dwell_outputs_zero_twist() -> None:
    active = ActivePrimitive(
        primitive=Dwell(duration_s=1.0, jaw_target_start=None, jaw_target_end=None),
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.25,
    )
    output = evaluate(active, _step(sim_step_index=1, dt=0.05))
    assert output == PrimitiveOutput(
        twist_camera=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
        jaw_target=0.0,
        is_finished=False,
    )


def test_probe_stays_in_contact_hold_then_retracts_and_finishes() -> None:
    hold_after_contact_s = 0.2
    retract_distance_m = 0.004
    retract_speed_m_per_s = 0.05
    probe = Probe(
        target_pose_scene=_identity_pose(),
        hold_after_contact_s=hold_after_contact_s,
        retract_distance_m=retract_distance_m,
        retract_peak_speed_m_per_s=retract_speed_m_per_s,
        duration_s=1.0,
        jaw_target_start=None,
        jaw_target_end=None,
    )
    active = ActivePrimitive(
        primitive=probe,
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=hold_after_contact_s - 0.01,
        in_post_contact_phase=True,
        post_contact_started_at_s=0.0,
    )

    output = evaluate(active, _step(sim_step_index=1, dt=0.01))
    assert output.twist_camera == Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0))
    assert output.is_finished is False

    active = ActivePrimitive(
        primitive=probe,
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=hold_after_contact_s + min_jerk_retract_duration(
            distance_m=retract_distance_m,
            peak_retract_speed_m_per_s=retract_speed_m_per_s,
        ) + 0.01,
        in_post_contact_phase=True,
        post_contact_started_at_s=0.0,
    )

    output = evaluate(active, _step(sim_step_index=2, dt=0.01))
    assert output.is_finished is True
    assert output.twist_camera == Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0))
