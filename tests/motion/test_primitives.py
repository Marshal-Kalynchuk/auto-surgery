from __future__ import annotations

import math

import pytest

from auto_surgery.motion.primitives import ActivePrimitive, Brush, ContactReach, Drag, Grip, Hold, PrimitiveOutput, Reach, evaluate
from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState


def _identity_pose() -> Pose:
    return Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _offset_pose(*, x: float, y: float, z: float) -> Pose:
    return Pose(position=Vec3(x=x, y=y, z=z), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _step(
    *,
    sim_step_index: int = 0,
    dt: float = 0.0,
    camera_position: Vec3 | None = None,
    camera_rotation: Quaternion | None = None,
    tip_pose: Pose | None = None,
    in_contact: bool = False,
    wrench: Vec3 | None = None,
) -> StepResult:
    if camera_position is None:
        camera_position = Vec3(x=0.0, y=0.0, z=0.0)
    if camera_rotation is None:
        camera_rotation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    if tip_pose is None:
        tip_pose = _identity_pose()
    if wrench is None:
        wrench = Vec3(x=0.0, y=0.0, z=0.0)
    cam_pose = Pose(position=camera_position, rotation=camera_rotation)
    bundle = SensorBundle(
        timestamp_ns=sim_step_index,
        sim_time_s=0.0,
        tool=ToolState(
            pose=tip_pose,
            twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
            jaw=0.0,
            wrench=wrench,
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


@pytest.mark.parametrize(
    "make_primitive",
    [
        lambda: Reach(
            target_pose_scene=_identity_pose(),
            duration_s=1.0,
            jaw_target_start=0.0,
            jaw_target_end=1.0,
        ),
        lambda: Hold(duration_s=1.0, jaw_target_start=0.0, jaw_target_end=0.0),
        lambda: ContactReach(
            direction_hint_scene=None,
            max_search_mm=100.0,
            peak_speed_mm_per_s=50.0,
            duration_s=1.0,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        lambda: Grip(
            approach=ContactReach(
                direction_hint_scene=Vec3(x=0.0, y=0.0, z=1.0),
                max_search_mm=100.0,
                peak_speed_mm_per_s=50.0,
                jaw_target_start=0.0,
                jaw_target_end=1.0,
                duration_s=0.2,
            ),
            duration_s=1.0,
        ),
        lambda: Drag(
            direction_hint_scene=None,
            distance_mm=50.0,
            normal_force_target=0.1,
            duration_s=1.0,
            jaw_target_start=0.0,
            jaw_target_end=0.0,
        ),
        lambda: Brush(
            amplitude_mm=5.0,
            frequency_hz=2.0,
            duration_s=1.0,
            jaw_target_start=0.0,
            jaw_target_end=0.0,
        ),
    ],
    ids=["reach", "hold", "contact_reach", "grip", "drag", "brush"],
)
def test_primitive_dataclasses_are_frozen(make_primitive: object) -> None:
    primitive = make_primitive()

    with pytest.raises(AttributeError):
        primitive.duration_s = 2.0


def test_evaluate_reach_produces_minimum_jerk_twist() -> None:
    active = ActivePrimitive(
        primitive=Reach(
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


def test_evaluate_reach_uses_camera_orientation_for_frame_axes() -> None:
    camera_angle = math.pi / 2.0
    active = ActivePrimitive(
        primitive=Reach(
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
    step = _step(
        sim_step_index=1,
        dt=0.1,
        camera_position=Vec3(x=1.0, y=2.0, z=3.0),
        camera_rotation=Quaternion(
            w=math.cos(camera_angle / 2.0),
            x=0.0,
            y=0.0,
            z=math.sin(camera_angle / 2.0),
        ),
    )
    output = evaluate(active, step)
    assert output.twist_camera.linear.x == pytest.approx(0.0, abs=1e-3)
    assert output.twist_camera.linear.y == pytest.approx(-2.8125, abs=1e-3)
    assert output.twist_camera.linear.z == pytest.approx(0.0, abs=1e-6)


def test_jaw_target_interpolates_with_min_jerk_profile() -> None:
    primitive = Reach(
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


def test_hold_outputs_zero_twist() -> None:
    active = ActivePrimitive(
        primitive=Hold(duration_s=1.0, jaw_target_start=None, jaw_target_end=None),
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


def test_reach_like_contact_primitive_finishes_on_contact() -> None:
    primitive = ContactReach(
        direction_hint_scene=Vec3(x=1.0, y=0.0, z=0.0),
        max_search_mm=100.0,
        peak_speed_mm_per_s=50.0,
        jaw_target_start=None,
        jaw_target_end=None,
        duration_s=1.0,
    )
    active = ActivePrimitive(
        primitive=primitive,
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.5,
    )

    output = evaluate(active, _step(sim_step_index=1, dt=0.1))
    assert output.twist_camera.linear.x > 0.0
    assert output.is_finished is False

    output = evaluate(active, _step(sim_step_index=2, dt=0.1, in_contact=True))
    assert output.is_finished is True


def test_grip_phases_eventually_finish() -> None:
    primitive = Grip(
        approach=ContactReach(
            direction_hint_scene=Vec3(x=1.0, y=0.0, z=0.0),
            max_search_mm=100.0,
            peak_speed_mm_per_s=50.0,
            jaw_target_start=0.0,
            jaw_target_end=0.0,
            duration_s=0.4,
        ),
        lift_distance_mm=10.0,
        lift_duration_s=0.2,
        release_after_s=0.2,
        jaw_close_duration_s=0.2,
        duration_s=1.2,
        jaw_target_start=0.0,
        jaw_target_end=0.0,
    )
    active = ActivePrimitive(
        primitive=primitive,
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=1.2,
        elapsed_s=0.2,
    )
    contact_phase = evaluate(active, _step(sim_step_index=1, dt=0.1))
    assert contact_phase.twist_camera.linear.x > 0.0
    assert contact_phase.is_finished is False

    active = ActivePrimitive(
        primitive=primitive,
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=1.2,
        elapsed_s=0.5,
    )
    closing_phase = evaluate(active, _step(sim_step_index=2, dt=0.1))
    assert closing_phase.twist_camera == Twist(
        linear=Vec3(x=0.0, y=0.0, z=0.0),
        angular=Vec3(x=0.0, y=0.0, z=0.0),
    )
    assert closing_phase.jaw_target > 0.0
    assert closing_phase.is_finished is False

    active = ActivePrimitive(
        primitive=primitive,
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=1.2,
        elapsed_s=1.2,
    )
    completed = evaluate(active, _step(sim_step_index=3, dt=0.1))
    assert completed.is_finished is True


@pytest.mark.parametrize(
    "primitive",
    [
        Drag(
            direction_hint_scene=Vec3(x=1.0, y=0.0, z=0.0),
            distance_mm=50.0,
            normal_force_target=0.1,
            duration_s=1.0,
            jaw_target_start=0.0,
            jaw_target_end=0.0,
        ),
        Brush(
            amplitude_mm=5.0,
            frequency_hz=2.0,
            duration_s=1.0,
            jaw_target_start=0.0,
            jaw_target_end=0.0,
        ),
    ],
)
def test_drag_and_brush_motion_has_nonzero_twist(primitive: Hold | ContactReach | Drag | Brush) -> None:
    active = ActivePrimitive(
        primitive=primitive,
        started_at_pose_scene=_identity_pose(),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.25,
    )

    output = evaluate(active, _step(sim_step_index=1, dt=0.05))
    assert output.is_finished is False
    assert output.twist_camera != Twist(
        linear=Vec3(x=0.0, y=0.0, z=0.0),
        angular=Vec3(x=0.0, y=0.0, z=0.0),
    )
