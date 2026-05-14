from __future__ import annotations

import pytest

from auto_surgery.motion.primitives import ActivePrimitive, ContactReach, evaluate, tip_desired_pose_scene
from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState


def _pose(*, x: float, y: float, z: float) -> Pose:
    return Pose(
        position=Vec3(x=float(x), y=float(y), z=float(z)),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )


def _step(*, sim_step_index: int, dt: float, in_contact: bool) -> StepResult:
    cam_pose = _pose(x=0.0, y=0.0, z=0.0)
    bundle = SensorBundle(
        timestamp_ns=sim_step_index * 1_000_000_000,
        sim_time_s=sim_step_index * dt,
        tool=ToolState(
            pose=cam_pose,
            twist=Twist(
                linear=Vec3(x=0.0, y=0.0, z=0.0),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            jaw=0.0,
            wrench=Vec3(x=0.0, y=0.0, z=0.0),
            in_contact=in_contact,
        ),
        cameras=[
            CameraView(
                camera_id="cam",
                timestamp_ns=sim_step_index * 1_000_000_000,
                extrinsics=cam_pose,
                intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=640, height=480),
            )
        ],
        safety=SafetyStatus(
            motion_enabled=True,
            command_blocked=False,
            block_reason=None,
            cycle_id_echo=sim_step_index,
        ),
    )
    return StepResult(sensors=bundle, dt=dt, sim_step_index=sim_step_index, is_capture_tick=True)


class _FakeSurfacePoint:
    def __init__(self, position: Vec3, normal: Vec3, signed_distance: float) -> None:
        self.position = position
        self.normal = normal
        self.signed_distance = float(signed_distance)


class _FakeSceneGeometry:
    def __init__(self, surface_point: _FakeSurfacePoint) -> None:
        self.surface_point = surface_point

    def closest_surface_point(self, _tip_position_scene: Vec3) -> _FakeSurfacePoint:
        return self.surface_point


def test_tip_desired_pose_contact_reach_moves_toward_surface() -> None:
    """Pose path uses scene geometry to bias the goal toward the reported surface point."""
    geometry = _FakeSceneGeometry(
        _FakeSurfacePoint(
            position=Vec3(x=50.0, y=0.0, z=0.0),
            normal=Vec3(x=1.0, y=0.0, z=0.0),
            signed_distance=50.0,
        )
    )
    active = ActivePrimitive(
        primitive=ContactReach(
            direction_hint_scene=None,
            max_search_mm=200.0,
            peak_speed_mm_per_s=100.0,
            duration_s=1.0,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        started_at_pose_scene=_pose(x=0.0, y=0.0, z=0.0),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.5,
    )
    step = _step(sim_step_index=1, dt=0.05, in_contact=False)
    desired = tip_desired_pose_scene(active, step, scene_geometry=geometry)
    assert float(desired.position.x) > 0.0
    assert float(desired.position.y) == pytest.approx(0.0)
    assert float(desired.position.z) == pytest.approx(0.0)


def test_evaluate_contact_reach_twist_points_along_hint_when_no_geometry() -> None:
    """Legacy twist evaluator still used for jaw/finish bookkeeping; direction hint sets search axis."""
    active = ActivePrimitive(
        primitive=ContactReach(
            direction_hint_scene=Vec3(x=1.0, y=0.0, z=0.0),
            max_search_mm=200.0,
            peak_speed_mm_per_s=100.0,
            duration_s=1.0,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        started_at_pose_scene=_pose(x=0.0, y=0.0, z=0.0),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.5,
    )
    step = _step(sim_step_index=1, dt=0.05, in_contact=False)
    output = evaluate(active, step)
    assert output.twist_camera.linear.x > 0.0
    assert output.twist_camera.linear.y == pytest.approx(0.0)
    assert output.twist_camera.linear.z == pytest.approx(0.0)


def test_evaluate_contact_reach_marks_finished_when_in_contact() -> None:
    active = ActivePrimitive(
        primitive=ContactReach(
            direction_hint_scene=Vec3(x=1.0, y=0.0, z=0.0),
            max_search_mm=200.0,
            peak_speed_mm_per_s=100.0,
            duration_s=1.0,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        started_at_pose_scene=_pose(x=0.0, y=0.0, z=0.0),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.5,
    )
    step = _step(sim_step_index=1, dt=0.05, in_contact=True)
    output = evaluate(active, step)
    assert output.is_finished is True


def test_evaluate_contact_reach_marks_finished_when_duration_elapsed() -> None:
    active = ActivePrimitive(
        primitive=ContactReach(
            direction_hint_scene=Vec3(x=1.0, y=0.0, z=0.0),
            max_search_mm=200.0,
            peak_speed_mm_per_s=100.0,
            duration_s=1.0,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        started_at_pose_scene=_pose(x=0.0, y=0.0, z=0.0),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=1.0,
    )
    step = _step(sim_step_index=1, dt=0.05, in_contact=False)
    output = evaluate(active, step)
    assert output.is_finished is True
