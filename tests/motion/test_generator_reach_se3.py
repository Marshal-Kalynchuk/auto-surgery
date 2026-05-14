from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from auto_surgery.motion.frames import pose_interpolate
from auto_surgery.motion.fsm import _ActivePrimitive
from auto_surgery.motion.generator import SurgicalMotionGenerator
from auto_surgery.motion.primitives import ActivePrimitive, Hold, Reach, tip_desired_pose_scene
from auto_surgery.schemas.commands import ControlFrame, ControlMode, Pose, Quaternion, SafetyMetadata, Twist, Vec3
from auto_surgery.schemas.motion import MotionGeneratorConfig, MotionShaping
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState


def _pose(x: float, y: float, z: float) -> Pose:
    return Pose(
        position=Vec3(x=float(x), y=float(y), z=float(z)),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )


def _step(*, sim_step_index: int, dt: float, tool_pose: Pose, tool_jaw: float = 0.0) -> StepResult:
    bundle = SensorBundle(
        timestamp_ns=sim_step_index * 1_000_000_000,
        sim_time_s=sim_step_index * dt,
        tool=ToolState(
            pose=tool_pose,
            twist=Twist(
                linear=Vec3(x=0.0, y=0.0, z=0.0),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            jaw=tool_jaw,
            wrench=Vec3(x=0.0, y=0.0, z=0.0),
            in_contact=False,
        ),
        cameras=[
            CameraView(
                camera_id="cam",
                timestamp_ns=sim_step_index * 1_000_000_000,
                extrinsics=Pose(
                    position=Vec3(x=0.0, y=0.0, z=0.0),
                    rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
                intrinsics=CameraIntrinsics(
                    fx=1.0,
                    fy=1.0,
                    cx=0.0,
                    cy=0.0,
                    width=640,
                    height=480,
                ),
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


def _motion_config() -> MotionGeneratorConfig:
    return MotionGeneratorConfig(
        seed=0,
        motion_shaping_enabled=True,
        motion_shaping=MotionShaping(
            max_linear_mm_s=200.0,
            max_angular_rad_s=2.0,
            max_linear_accel_mm_s2=400.0,
            max_angular_accel_rad_s2=4.0,
            bias_gain_max=0.0,
            bias_ramp_distance_mm=1.0,
            orientation_bias_gain=0.0,
            orientation_deadband_rad=0.0,
            frustum_margin_mm=None,
        ),
    )


class _StubFsm:
    def __init__(self, active: _ActivePrimitive) -> None:
        self._active = active

    def reset(self) -> None:
        return None

    def finalize(self, _last_step: StepResult) -> None:
        return None

    def step(self, *_args: object, **_kwargs: object) -> _ActivePrimitive:
        return self._active

    @property
    def completed(self) -> tuple[object, ...]:
        return ()


def test_tip_desired_reach_matches_min_jerk_pose_interpolation() -> None:
    target_rot = Quaternion(w=math.cos(math.pi / 4.0), x=0.0, y=0.0, z=math.sin(math.pi / 4.0))
    reach = Reach(
        target_pose_scene=Pose(position=Vec3(x=1.0, y=0.0, z=0.0), rotation=target_rot),
        duration_s=1.0,
        jaw_target_start=None,
        jaw_target_end=None,
        end_on_contact=True,
    )
    started = _pose(0.0, 0.0, 0.0)
    active = ActivePrimitive(
        primitive=reach,
        started_at_pose_scene=started,
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.5,
    )
    last = _step(sim_step_index=1, dt=0.5, tool_pose=started)
    got = tip_desired_pose_scene(active, last, scene_geometry=None)
    from auto_surgery.motion.primitives import min_jerk_position_scalar

    s = min_jerk_position_scalar(0.5)
    expect = pose_interpolate(started, reach.target_pose_scene, s)
    assert got.position.x == pytest.approx(expect.position.x, rel=1e-9, abs=1e-9)
    assert got.position.y == pytest.approx(expect.position.y, rel=1e-9, abs=1e-9)
    assert got.position.z == pytest.approx(expect.position.z, rel=1e-9, abs=1e-9)


def test_next_command_emits_scene_pose_for_hold() -> None:
    active_fsm = _ActivePrimitive(
        primitive=Hold(
            duration_s=2.0,
            jaw_target_start=None,
            jaw_target_end=None,
        ),
        started_at_tick=0,
        started_at_pose_scene=_pose(0.0, 0.0, 0.0),
        started_at_jaw=0.2,
        duration_s=2.0,
        elapsed_s=1.0,
    )
    generator = SurgicalMotionGenerator(
        _motion_config(),
        SimpleNamespace(
            tool=SimpleNamespace(initial_jaw=0.2),
        ),
    )
    generator._fsm = _StubFsm(active=active_fsm)

    generator.reset(_step(sim_step_index=0, dt=0.0, tool_pose=_pose(0.5, 0.0, 0.0)))
    cmd = generator.next_command(_step(sim_step_index=1, dt=1.0, tool_pose=_pose(0.5, 0.0, 0.0)))

    assert cmd.control_mode == ControlMode.CARTESIAN_POSE
    assert cmd.frame == ControlFrame.SCENE
    assert cmd.cartesian_pose_target is not None
    assert cmd.cartesian_twist is None
    assert cmd.safety is not None
    assert cmd.safety == SafetyMetadata(
        clamped_linear=False,
        clamped_angular=False,
        biased_linear=False,
        biased_angular=False,
        scaled_by=None,
        signed_distance_to_envelope_mm=None,
        signed_distance_to_surface_mm=None,
        pose_error_norm_mm=None,
        pose_error_norm_rad=None,
    )
    assert cmd.cartesian_pose_target.position.x == pytest.approx(0.0)
    assert cmd.cartesian_pose_target.position.y == pytest.approx(0.0)
    assert cmd.cartesian_pose_target.position.z == pytest.approx(0.0)
