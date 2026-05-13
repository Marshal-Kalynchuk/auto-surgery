from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_surgery.motion.generator import SurgicalMotionGenerator
from auto_surgery.motion.primitives import ContactReach
from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3
from auto_surgery.schemas.motion import MotionGeneratorConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneConfig
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


def _motion_config() -> MotionGeneratorConfig:
    return MotionGeneratorConfig(seed=0)


def _generator(*, scene_geometry: _FakeSceneGeometry) -> SurgicalMotionGenerator:
    generator = SurgicalMotionGenerator(
        _motion_config(),
        SceneConfig(tissue_scene_path=Path("test_tissue.obj")),
    )
    generator._sequencer._scene_geometry = scene_geometry
    return generator


def test_evaluate_contact_reach_drives_toward_surface() -> None:
    geometry = _FakeSceneGeometry(
        _FakeSurfacePoint(
            position=Vec3(x=0.2, y=0.0, z=0.0),
            normal=Vec3(x=1.0, y=0.0, z=0.0),
            signed_distance=0.2,
        )
    )
    generator = _generator(scene_geometry=geometry)
    active = SimpleNamespace(
        primitive=ContactReach(
            direction_hint_scene=None,
            max_search_m=0.2,
            peak_speed_m_per_s=0.1,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        started_at_tick=0,
        started_at_pose_scene=_pose(x=0.0, y=0.0, z=0.0),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.0,
    )
    step = _step(sim_step_index=1, dt=0.05, in_contact=False)
    output = generator._evaluate_contact_reach(
        active=active,
        last_step=step,
        tip_now=_pose(x=0.0, y=0.0, z=0.0).position,
        tip_pose_scene=_pose(x=0.0, y=0.0, z=0.0),
    )

    twist = output.data()
    assert twist.linear.x > 0.0
    assert twist.linear.y == pytest.approx(0.0)
    assert twist.linear.z == pytest.approx(0.0)


@pytest.mark.parametrize("in_contact,signed_distance", [(True, 0.2), (False, 1e-4)])
def test_evaluate_contact_reach_terminates_on_contact_or_threshold(in_contact: bool, signed_distance: float) -> None:
    geometry = _FakeSceneGeometry(
        _FakeSurfacePoint(
            position=Vec3(x=0.2, y=0.0, z=0.0),
            normal=Vec3(x=1.0, y=0.0, z=0.0),
            signed_distance=signed_distance,
        )
    )
    generator = _generator(scene_geometry=geometry)
    active = SimpleNamespace(
        primitive=ContactReach(
            direction_hint_scene=None,
            max_search_m=0.2,
            peak_speed_m_per_s=0.1,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        started_at_tick=0,
        started_at_pose_scene=_pose(x=0.0, y=0.0, z=0.0),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.0,
    )
    step = _step(sim_step_index=1, dt=0.05, in_contact=in_contact)
    output = generator._evaluate_contact_reach(
        active=active,
        last_step=step,
        tip_now=_pose(x=0.0, y=0.0, z=0.0).position,
        tip_pose_scene=_pose(x=0.0, y=0.0, z=0.0),
    )

    twist = output.data()
    assert twist == Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0))
