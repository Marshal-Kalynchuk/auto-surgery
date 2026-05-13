from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from auto_surgery.motion.primitives import Hold, Reach
from auto_surgery.schemas.commands import Pose, Quaternion, SafetyMetadata, Twist, Vec3
from auto_surgery.schemas.motion import MotionGeneratorConfig, MotionShaping
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState


from auto_surgery.motion.generator import _evaluate_reach, SurgicalMotionGenerator
from auto_surgery.motion.fsm import _ActivePrimitive
from auto_surgery.motion.frames import TwistSceneTip


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
    config = MotionGeneratorConfig(seed=0, motion_shaping_enabled=True)
    config.__dict__["motion_shaping"] = MotionShaping(
        max_linear_m_s=0.2,
        max_angular_rad_s=0.2,
        max_linear_accel_m_s2=1.0,
        max_angular_accel_rad_s2=1.0,
        bias_gain_max=1.0,
        bias_ramp_distance_m=1.0,
        orientation_bias_gain=0.0,
        orientation_deadband_rad=0.0,
    )
    return config


class _BiasEnvelope:
    outer_margin_m = 1.0

    def __init__(self, signed_distance: float) -> None:
        self.signed_distance_value = float(signed_distance)

    def signed_distance(self, p_scene: Vec3) -> float:
        return self.signed_distance_value


class _StubFsm:
    def __init__(self, active: _ActivePrimitive) -> None:
        self._active = active

    def reset(self) -> None:
        return None

    def finalize(self, _last_step: StepResult) -> None:
        return None

    def step(self, *_args: object, **_kwargs: object) -> _ActivePrimitive:
        return self._active


def test_evaluate_reach_returns_unified_se3_minjerk() -> None:
    active = _ActivePrimitive(
        primitive=Reach(
            target_pose_scene=Pose(
                position=Vec3(x=1.0, y=0.0, z=0.0),
                rotation=Quaternion(
                    w=math.cos(math.pi / 4.0),
                    x=0.0,
                    y=0.0,
                    z=math.sin(math.pi / 4.0),
                ),
            ),
            duration_s=1.0,
            jaw_target_start=None,
            jaw_target_end=None,
            end_on_contact=True,
        ),
        started_at_tick=0,
        started_at_pose_scene=_pose(0.0, 0.0, 0.0),
        started_at_jaw=0.0,
        duration_s=1.0,
        elapsed_s=0.5,
    )
    result = _evaluate_reach(
        active=active,
        last_step=_step(sim_step_index=1, dt=0.5, tool_pose=_pose(0.0, 0.0, 0.0)),
    )

    assert isinstance(result, TwistSceneTip)
    twist = result.data()
    tau = 0.5
    expected_speed = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    assert twist.linear.x == pytest.approx(expected_speed)
    assert abs(twist.angular.z) == pytest.approx(expected_speed * (math.pi / 2.0))
    assert twist.linear.x > 0.0
    assert abs(twist.angular.z) > 0.0


def test_next_command_applies_linear_bias_and_populates_safety() -> None:
    active = _ActivePrimitive(
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
            tool=SimpleNamespace(
                workspace_envelope=_BiasEnvelope(0.5),
                initial_jaw=0.2,
            )
        ),
    )
    generator._fsm = _StubFsm(active=active)

    generator.reset(_step(sim_step_index=0, dt=0.0, tool_pose=_pose(0.5, 0.0, 0.0)))
    cmd = generator.next_command(_step(sim_step_index=1, dt=1.0, tool_pose=_pose(0.5, 0.0, 0.0)))

    assert cmd.safety is not None
    assert cmd.safety == SafetyMetadata(
        clamped_linear=False,
        clamped_angular=False,
        biased_linear=True,
        biased_angular=False,
        scaled_by=None,
        signed_distance_to_envelope_m=0.5,
        signed_distance_to_surface_m=None,
    )
    assert cmd.cartesian_twist.linear.x == pytest.approx(-0.25)
    assert cmd.cartesian_twist.linear.y == pytest.approx(0.0)
    assert cmd.cartesian_twist.linear.z == pytest.approx(0.0)