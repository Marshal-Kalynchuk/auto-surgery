from __future__ import annotations

import pytest

from auto_surgery.env.sofa_orchestration import SofaEnvironment
from auto_surgery.schemas.commands import (
    ControlMode,
    Pose,
    Quaternion,
    RobotCommand,
    Twist,
    Vec3,
)
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneConfig, SceneGeometryEnvelope, SceneGraph, ToolSpec
from auto_surgery.schemas.sensors import (
    CameraIntrinsics,
    CameraView,
    JointState,
    SafetyStatus,
    SensorBundle,
    ToolState,
)


class _LinearWorkspaceEnvelope(SceneGeometryEnvelope):
    def signed_distance_to_envelope(self, p_scene: Vec3) -> float:
        return 1.0 - float(p_scene.x)


class _PredictiveBackend:
    def __init__(self, tool_pose: Pose) -> None:
        self._tool_pose = tool_pose
        self._control_rate_hz = 250.0
        self._frame_index = 0
        self.last_command: RobotCommand | None = None
        self._sensors = _sensor_bundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            cycle_id_echo=-1,
            tool_pose=tool_pose,
        )

    def reset(self, config: EnvConfig) -> StepResult:
        del config
        self._frame_index = 0
        return StepResult(
            sensors=self._sensors,
            dt=0.0,
            sim_step_index=self._frame_index,
            is_capture_tick=True,
        )

    def step(self, action: RobotCommand) -> StepResult:
        self.last_command = action
        self._frame_index += 1
        return StepResult(
            sensors=self._sensors,
            dt=1.0 / self._control_rate_hz,
            sim_step_index=self._frame_index,
            is_capture_tick=True,
        )

    def set_control_rate_hz(self, control_rate_hz: float) -> None:
        self._control_rate_hz = float(control_rate_hz)

    def set_initial_jaw(self, jaw: float) -> None:
        del jaw

    def set_tool_jaw_target(self, jaw: float) -> None:
        del jaw

    def get_sensors(self) -> SensorBundle:
        return self._sensors

    def get_scene(self) -> SceneGraph:
        return SceneGraph()

    def get_joint_state(self) -> JointState:
        return JointState(positions={}, velocities={})

    def get_contacts(self) -> list:
        return []


def _sensor_bundle(
    *,
    timestamp_ns: int,
    sim_time_s: float,
    cycle_id_echo: int,
    tool_pose: Pose,
) -> SensorBundle:
    return SensorBundle(
        timestamp_ns=timestamp_ns,
        sim_time_s=sim_time_s,
        tool=ToolState(
            pose=tool_pose,
            twist=Twist(
                linear=Vec3(x=0.0, y=0.0, z=0.0),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            jaw=0.0,
            wrench=Vec3(x=0.0, y=0.0, z=0.0),
            in_contact=False,
        ),
        cameras=[
            CameraView(
                camera_id="test_camera",
                timestamp_ns=timestamp_ns,
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
            motion_enabled=False,
            command_blocked=False,
            block_reason=None,
            cycle_id_echo=cycle_id_echo,
        ),
    )


def _build_environment(initial_tool_x: float) -> tuple[SofaEnvironment, _PredictiveBackend, EnvConfig]:
    pose = Pose(
        position=Vec3(x=initial_tool_x, y=0.0, z=0.0),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    backend = _PredictiveBackend(pose)

    scene = SceneConfig(
        tissue_scene_path="/tmp/orchestration_predictive_scene.xml",
        tool=ToolSpec(
            workspace_envelope=_LinearWorkspaceEnvelope(
                outer_margin_mm=1.0,
                inner_margin_mm=0.0,
                no_go_regions=[],
            )
        ),
    )

    def backend_factory(_scene_path: str, _extra: dict[str, object] | None) -> _PredictiveBackend:
        return backend

    env = SofaEnvironment(
        sofa_scene_path="/tmp/orchestration_predictive_scene.xml",
        scene_config=scene,
        sofa_backend_factory=backend_factory,
    )
    config = EnvConfig(
        seed=123,
        scene=scene,
        control_rate_hz=1.0,
        frame_rate_hz=30.0,
    )
    return env, backend, config


def _command(*, cycle_id: int, linear_x: float) -> RobotCommand:
    return RobotCommand(
        timestamp_ns=1_000 * (cycle_id + 1),
        cycle_id=cycle_id,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=linear_x, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
    )


def test_orchestration_predictive_scaling_updates_command_and_safety() -> None:
    env, backend, config = _build_environment(initial_tool_x=0.3)
    env.reset(config)

    result = env.step(_command(cycle_id=0, linear_x=0.3))

    assert result.sensors.safety.command_blocked is False
    assert backend.last_command is not None
    assert float(backend.last_command.cartesian_twist.linear.x) == pytest.approx(0.2)
    assert backend.last_command.safety is not None
    assert backend.last_command.safety.clamped_linear is True
    assert backend.last_command.safety.scaled_by == pytest.approx(2.0 / 3.0)
    assert backend.last_command.safety.signed_distance_to_envelope_mm == pytest.approx(0.5)


def test_orchestration_predictive_hard_block_rejects_out_of_bounds_motion() -> None:
    env, backend, config = _build_environment(initial_tool_x=0.3)
    env.reset(config)

    command = _command(cycle_id=0, linear_x=1.5)
    blocked, reason = env._command_block_reason(command)

    assert blocked is True
    assert reason == "tip_outside_workspace_envelope"
    assert command.safety is not None
    assert command.safety.clamped_linear is False
    assert command.safety.signed_distance_to_envelope_mm < 0.0

    result = env.step(command)
    assert result.sensors.safety.command_blocked is True
    assert backend.last_command is not None
    assert float(backend.last_command.cartesian_twist.linear.x) == pytest.approx(0.0)
