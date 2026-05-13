from __future__ import annotations

from pathlib import Path
import pytest
from typing import Any

from auto_surgery.env.protocol import Environment
from auto_surgery.env.protocol import EnvironmentWithSensors
from auto_surgery.env.real import RealEnvironment
from auto_surgery.env.sofa_backend import _NativeSofaBackend
from auto_surgery.config import load_motion_config, load_scene_config
from auto_surgery.env.sofa import (
    SofaEnvironment,
    SofaNotIntegratedError,
    discover_sofa_runtime_contract,
    resolve_sofa_runtime_import_candidates,
)
from auto_surgery.env.sofa_tools import build_forceps_observer
from auto_surgery.schemas.commands import ControlMode, Pose, Quaternion, RobotCommand
from auto_surgery.schemas.manifests import EnvConfig, SceneConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph, SlotRecord, ToolSpec
from auto_surgery.motion import SurgicalMotionGenerator
from auto_surgery.schemas.sensors import (
    CameraIntrinsics,
    CameraView,
    Contact,
    JointState,
    SafetyStatus,
    SensorBundle,
    ToolState,
    Twist,
    Vec3,
)


def _rollout(env: Environment) -> int:
    state = env.reset(
        EnvConfig(
            seed=7,
            scene=_scene_config_with_jaw(initial_jaw=0.0),
        )
    )
    assert state.sim_step_index == 0
    cmd = RobotCommand(
        timestamp_ns=100,
        cycle_id=0,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=0.1, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
    )
    result = env.step(cmd)
    assert result.sensors.timestamp_ns == cmd.timestamp_ns
    if isinstance(env, EnvironmentWithSensors):
        obs = env.get_sensors()
        assert obs.safety.cycle_id_echo == 0
    return 1


_TEST_SCENE_PATH = (
    Path("src/auto_surgery/env/sofa_scenes/brain_dejavu_episodic.scn.template").resolve()
)


def _scene_config_with_jaw(
    *, initial_jaw: float, scene_path: str | Path = _TEST_SCENE_PATH
) -> SceneConfig:
    return SceneConfig(
        tissue_scene_path=Path(scene_path),
        tool=ToolSpec(initial_jaw=initial_jaw),
    )


class _SyntheticSimEnvironment(Environment):
    """Tiny protocol implementation for non-SOFA contract tests."""

    def __init__(self) -> None:
        self._frame_index = 0
        self._sim_time_s = 0.0
        self._sim_step_index = 0
        self._last_accepted_cycle_id = -1
        self._jaw_target = 0.0
        self._control_rate_hz = 250.0
        self._frame_rate_hz = 30.0
        self._frame_decimation = max(1, round(self._control_rate_hz / self._frame_rate_hz))
        self._joint_state = JointState(positions={}, velocities={})
        self._scene = SceneGraph(slots=[])
        self._sensors = self._build_sensor_bundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            safety=SafetyStatus(
                motion_enabled=False,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=-1,
            ),
        )

    def reset(self, config: EnvConfig) -> StepResult:
        self._frame_index = 0
        self._scene = SceneGraph(
            frame_index=self._frame_index,
            slots=[SlotRecord(slot_id="tool_0", pose={"x": 0.0, "y": 0.0, "z": 0.1})],
            events=[
                {"type": "reset", "seed": config.seed},
            ],
        )
        self._last_accepted_cycle_id = -1
        self._sim_step_index = 0
        self._sim_time_s = 0.0
        self._jaw_target = float(config.scene.tool.initial_jaw)
        self._control_rate_hz = config.control_rate_hz
        self._frame_rate_hz = config.frame_rate_hz
        self._frame_decimation = max(1, round(self._control_rate_hz / self._frame_rate_hz))
        self._sensors = self._build_sensor_bundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            safety=SafetyStatus(
                motion_enabled=False,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=-1,
            ),
        )
        return StepResult(
            sensors=self._sensors,
            dt=0.0,
            sim_step_index=0,
            is_capture_tick=True,
        )

    def step(self, action: RobotCommand) -> StepResult:
        blocked, reason = self._command_block_reason(action)
        if not blocked:
            self._last_accepted_cycle_id = action.cycle_id
        dt = 1.0 / self._control_rate_hz
        self._sim_time_s += dt
        self._sim_step_index += 1
        if not blocked:
            self._jaw_target = (
                action.tool_jaw_target if action.tool_jaw_target is not None else self._jaw_target
            )
        safety = SafetyStatus(
            motion_enabled=action.enable and not blocked,
            command_blocked=blocked,
            block_reason=reason,
            cycle_id_echo=action.cycle_id,
        )
        self._sensors = self._build_sensor_bundle(
            timestamp_ns=action.timestamp_ns,
            sim_time_s=self._sim_time_s,
            safety=safety,
        )
        return StepResult(
            sensors=self._sensors,
            dt=dt,
            sim_step_index=self._sim_step_index,
            is_capture_tick=(self._sim_step_index % self._frame_decimation == 0),
        )

    def get_joint_state(self) -> JointState:
        return self._joint_state

    def get_sensors(self) -> SensorBundle:
        return self._sensors.model_copy(deep=True)

    def get_scene(self) -> SceneGraph:
        return self._scene.model_copy(deep=True)

    def get_contacts(self) -> list[Contact]:
        return []

    def _command_block_reason(self, command: RobotCommand) -> tuple[bool, str | None]:
        if command.cycle_id <= self._last_accepted_cycle_id:
            return True, "stale_cycle_id"
        if not command.enable:
            return True, "disabled"
        return False, None

    def _build_sensor_bundle(
        self,
        *,
        timestamp_ns: int,
        sim_time_s: float,
        safety: SafetyStatus,
    ) -> SensorBundle:
        tool_pose = Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
        tool_twist = Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0))
        tool_state = ToolState(
            pose=tool_pose,
            twist=tool_twist,
            jaw=self._jaw_target,
            wrench=Vec3(x=0.0, y=0.0, z=0.0),
            in_contact=False,
        )
        camera = CameraView(
            camera_id="test_stub_cam",
            timestamp_ns=timestamp_ns,
            extrinsics=tool_pose,
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=640, height=480),
        )
        return SensorBundle(
            timestamp_ns=timestamp_ns,
            sim_time_s=sim_time_s,
            tool=tool_state,
            cameras=[camera],
            safety=safety,
        )


def _joint_command(step_index: int, *, enable: bool = True, cycle_id: int | None = None) -> RobotCommand:
    return RobotCommand(
        timestamp_ns=100_000 + step_index,
        cycle_id=step_index if cycle_id is None else cycle_id,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=float(step_index) * 0.01, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=enable,
        source="test",
    )


def test_synthetic_sim_matches_environment_protocol() -> None:
    sim = _SyntheticSimEnvironment()
    assert isinstance(sim, Environment)
    assert _rollout(sim) >= 1


def test_real_environment_delegates_to_impl() -> None:
    inner = _SyntheticSimEnvironment()
    real = RealEnvironment(impl=inner)
    assert isinstance(real, Environment)
    assert _rollout(real) >= 1


def test_environment_protocol_is_minimal() -> None:
    class _MinimalEnvironment:
        def reset(self, config: EnvConfig) -> StepResult:
            return StepResult(
                sensors=SensorBundle(
                    timestamp_ns=0,
                    sim_time_s=0.0,
                    tool=ToolState(
                        pose=Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)),
                        twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
                        jaw=0.0,
                        wrench=Vec3(x=0.0, y=0.0, z=0.0),
                        in_contact=False,
                    ),
                    cameras=[],
                    safety=SafetyStatus(
                        motion_enabled=False,
                        command_blocked=False,
                        block_reason=None,
                        cycle_id_echo=0,
                    ),
                ),
                dt=0.0,
                sim_step_index=0,
                is_capture_tick=True,
            )

        def step(self, command: RobotCommand) -> StepResult:
            return self.reset(EnvConfig(seed=0))

        def get_joint_state(self) -> JointState:
            return JointState(positions={}, velocities={})

        def get_contacts(self) -> list[Contact]:
            return []

    assert isinstance(_MinimalEnvironment(), Environment)
    assert isinstance(_SyntheticSimEnvironment(), EnvironmentWithSensors)


class _TestBackend:
    """Simple backend used for protocol contract tests."""

    def __init__(
        self,
        scene_path: str,
        *,
        tool_wrench: Vec3 | None = None,
        in_contact: bool = False,
    ) -> None:
        self._frame_index = 0
        self._control_rate_hz = 250.0
        self._frame_decimation = 1
        self._jaw = 0.0
        self._tool_wrench = Vec3(
            x=0.0 if tool_wrench is None else float(tool_wrench.x),
            y=0.0 if tool_wrench is None else float(tool_wrench.y),
            z=0.0 if tool_wrench is None else float(tool_wrench.z),
        )
        self._in_contact = bool(in_contact)
        self._scene = SceneGraph(
            frame_index=0,
            slots=[SlotRecord(slot_id="tool", pose={"x": 0.0, "y": 0.0, "z": 0.0})],
            events=[{"phase": "init", "scene_path": scene_path}],
        )
        self._sensors = SensorBundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            tool=ToolState(
                pose=Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)),
                twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
                jaw=0.0,
                wrench=self._tool_wrench,
                in_contact=self._in_contact,
            ),
            cameras=[
                CameraView(
                    camera_id="test_cam",
                    timestamp_ns=0,
                    extrinsics=Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)),
                    intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=640, height=480),
                )
            ],
            safety=SafetyStatus(
                motion_enabled=True,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=-1,
            ),
        )

    def reset(self, config: EnvConfig) -> StepResult:
        self._frame_index = 0
        self._control_rate_hz = config.control_rate_hz
        self._frame_decimation = max(1, round(self._control_rate_hz / config.frame_rate_hz))
        self._jaw = float(config.scene.tool.initial_jaw)
        self._scene = self._scene.model_copy(
            update={"frame_index": 0, "events": [{"reset_seed": config.seed}]}
        )
        tool_with_jaw = self._sensors.tool.model_copy(update={"jaw": self._jaw})
        self._sensors = self._sensors.model_copy(
            deep=True,
            update={
                "timestamp_ns": 0,
                "sim_time_s": 0.0,
                "tool": tool_with_jaw,
                "cameras": [self._sensors.cameras[0].model_copy(update={"frame_rgb": b"reset"})],
                "safety": SafetyStatus(
                    motion_enabled=False,
                    command_blocked=False,
                    block_reason=None,
                    cycle_id_echo=-1,
                ),
            },
        )
        return StepResult(
            sensors=self._sensors,
            dt=0.0,
            sim_step_index=0,
            is_capture_tick=True,
        )

    def step(self, action: RobotCommand) -> StepResult:
        self._frame_index += 1
        dt = 1.0 / self._control_rate_hz
        self._jaw = float(action.tool_jaw_target) if action.tool_jaw_target is not None else self._jaw
        self._sensors = SensorBundle(
            timestamp_ns=action.timestamp_ns,
            sim_time_s=self._frame_index / self._control_rate_hz,
            tool=self._sensors.tool.model_copy(update={"jaw": self._jaw, "wrench": self._tool_wrench, "in_contact": self._in_contact}),
            cameras=[
                self._sensors.cameras[0].model_copy(
                    update={"frame_rgb": f"frame_{self._frame_index}".encode("utf-8")}
                )
            ],
            safety=SafetyStatus(
                motion_enabled=action.enable,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=action.cycle_id,
            ),
        )
        self._scene = self._scene.model_copy(
            update={
                "frame_index": self._frame_index,
                "events": [{"step": self._frame_index}],
            }
        )
        return StepResult(
            sensors=self._sensors,
            dt=dt,
            sim_step_index=self._frame_index,
            is_capture_tick=(self._frame_index % self._frame_decimation == 0),
        )

    def get_sensors(self) -> SensorBundle:
        return self._sensors

    def get_scene(self) -> SceneGraph:
        return self._scene

    def set_control_rate_hz(self, control_rate_hz: float) -> None:
        self._control_rate_hz = control_rate_hz
        self._frame_decimation = max(1, round(self._control_rate_hz / 25.0))

    def set_initial_jaw(self, jaw: float) -> None:
        self._jaw = jaw

    def set_tool_jaw_target(self, jaw: float) -> None:
        self._jaw = jaw

    def get_joint_state(self) -> JointState:
        return JointState(positions={}, velocities={})

    def get_contacts(self) -> list[Contact]:
        return []

    def set_tool_state(self, *, wrench: Vec3, in_contact: bool) -> None:
        self._tool_wrench = wrench
        self._in_contact = in_contact


class _CommandCaptureBackend(_TestBackend):
    """Capture the action payload after dispatch and gating."""

    def __init__(
        self,
        scene_path: str,
        *,
        tool_wrench: Vec3 | None = None,
        in_contact: bool = False,
    ) -> None:
        super().__init__(scene_path, tool_wrench=tool_wrench, in_contact=in_contact)
        self.last_command: RobotCommand | None = None

    def step(self, action: RobotCommand) -> StepResult:
        self.last_command = action
        return super().step(action)


class _Data:
    def __init__(self, value: list[float]) -> None:
        self.value = value


class _FakeForcepsDof:
    def __init__(self) -> None:
        self.position = _Data([0.25, 0.5, 0.75, 0.0, 0.0, 0.0, 1.0])
        self.v = _Data([0.01, -0.02, 0.03])
        self.w = _Data([-0.01, 0.02, -0.03])
        self.wrench = _Data([0.4, 0.5, 0.6])
        self.in_contact = True


class _FakeSofaRuntimeModule:
    """Runtime shim used to exercise capture fallback behavior without SOFA bindings."""

    class _Scene:
        def __init__(self, include_capture_camera: bool) -> None:
            self._include_capture_camera = include_capture_camera

        def getObject(self, name: str) -> object | None:
            if name == "auto_surgery_offscreen_camera" and self._include_capture_camera:
                return object()
            return None

    def __init__(self, include_capture_camera: bool) -> None:
        self._scene = self._Scene(include_capture_camera)

    def load_scene(self, scene_path: str) -> "_FakeSofaRuntimeModule._Scene":
        del scene_path
        return self._scene

    def step(self, _scene: Any, _action: RobotCommand) -> None:
        return None

    def reset(self, _scene: Any, _config: EnvConfig) -> None:
        return None


def _twist_command(
    step_index: int, *, enable: bool = True, cycle_id: int | None = None
) -> RobotCommand:
    generator = SurgicalMotionGenerator(
        load_motion_config("configs/motion/default.yaml").model_copy(update={"seed": 7}),
        load_scene_config("configs/scenes/dejavu_brain.yaml"),
    )
    sim_env = _SyntheticSimEnvironment()
    step = sim_env.reset(
        EnvConfig(
            seed=7,
            scene=_scene_config_with_jaw(
                initial_jaw=0.0,
                scene_path="test://scene.json",
            ),
        )
    )
    command = generator.reset(step)
    for _ in range(step_index):
        step = sim_env.step(command)
        command = generator.next_command(step)
    return command.model_copy(
        update={"enable": enable, "cycle_id": step_index if cycle_id is None else cycle_id}
    )


def test_sofa_environment_auto_forceps_applier_shares_jaw_ref_with_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording/training must not pass a bare applier: it must share `tool_state_jaw_ref`."""

    captured: dict[str, Any] = {}

    import auto_surgery.env.sofa_tools as sofa_tools_mod

    real_resolve = sofa_tools_mod.resolve_tool_action_applier_from_spec

    def _capture(tool_spec: ToolSpec, **kwargs: Any) -> Any:
        captured["jaw_ref"] = kwargs.get("jaw_ref")
        captured["camera_pose_provider"] = kwargs.get("camera_pose_provider")
        return real_resolve(tool_spec, **kwargs)

    monkeypatch.setattr(sofa_tools_mod, "resolve_tool_action_applier_from_spec", _capture)

    scene = load_scene_config("configs/scenes/dejavu_brain.yaml")
    env = SofaEnvironment(
        scene_config=scene,
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: _TestBackend(scene_path),
    )
    assert captured.get("jaw_ref") is env._tool_state_jaw_ref
    provider = captured.get("camera_pose_provider")
    assert callable(provider)
    assert provider() == scene.camera_extrinsics_scene
    updated_scene = scene.model_copy(
        update={
            "camera_extrinsics_scene": Pose(
                position=Vec3(x=1.0, y=2.0, z=3.0),
                rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            )
        }
    )
    env.reset(
        EnvConfig(
            seed=9,
            scene=updated_scene,
            frame_rate_hz=env._frame_rate_hz,
            control_rate_hz=env._control_rate_hz,
        )
    )
    assert provider() == updated_scene.camera_extrinsics_scene


def test_sofa_contract_with_injected_backend_round_trip() -> None:
    env = SofaEnvironment(
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: _TestBackend(scene_path),
    )
    scene = env.reset(
        EnvConfig(
            seed=9,
            scene=_scene_config_with_jaw(initial_jaw=0.0),
        )
    )
    assert scene.sim_step_index == 0
    cmd = RobotCommand(
        timestamp_ns=1_000_000,
        cycle_id=0,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=0.05, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
    )
    step = env.step(cmd)
    assert step.sensors.safety.cycle_id_echo == 0
    assert step.sensors.timestamp_ns == cmd.timestamp_ns
    assert env.get_sensors().safety.cycle_id_echo == cmd.cycle_id
    assert env.get_scene().frame_index == 1


def test_sofa_environment_gating_and_capture_ticks() -> None:
    env = SofaEnvironment(
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: _TestBackend(scene_path),
    )
    cfg = EnvConfig(
        seed=11,
        control_rate_hz=100.0,
        frame_rate_hz=25.0,
        scene=_scene_config_with_jaw(initial_jaw=0.42),
    )
    reset_result = env.reset(cfg)
    assert reset_result.dt == 0.0
    assert reset_result.sim_step_index == 0
    assert reset_result.is_capture_tick is True
    assert reset_result.sensors.cameras[0].frame_rgb == b"reset"
    assert reset_result.sensors.tool.jaw == 0.42
    assert env.get_sensors().sim_time_s == 0.0

    step_a = env.step(_joint_command(0))
    assert step_a.sim_step_index == 1
    assert step_a.dt == 1.0 / cfg.control_rate_hz
    assert step_a.sensors.cameras[0].frame_rgb is None
    assert step_a.sensors.safety.motion_enabled
    assert step_a.sensors.safety.command_blocked is False

    stale = env.step(_joint_command(0))
    assert stale.sim_step_index == 2
    assert stale.sensors.cameras[0].frame_rgb is None
    assert stale.sensors.safety.command_blocked is True
    assert stale.sensors.safety.block_reason == "stale_cycle_id"
    assert stale.sensors.safety.motion_enabled is False

    disabled = env.step(_joint_command(1, enable=False))
    assert disabled.sim_step_index == 3
    assert disabled.sensors.safety.command_blocked is True
    assert disabled.sensors.safety.block_reason == "disabled"
    assert disabled.sensors.safety.motion_enabled is False

    accepted = env.step(_joint_command(2))
    assert accepted.sim_step_index == 4
    assert accepted.sensors.safety.command_blocked is False
    assert accepted.sensors.safety.block_reason is None
    assert accepted.sensors.safety.motion_enabled is True
    assert accepted.is_capture_tick is True
    assert accepted.sensors.cameras[0].frame_rgb is not None


def test_sofa_environment_get_sensors_matches_last_step_contract() -> None:
    backend = _CommandCaptureBackend("test://scene.json")
    env = SofaEnvironment(
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: backend,
    )
    env.reset(
        EnvConfig(
            seed=13,
            control_rate_hz=100.0,
            frame_rate_hz=25.0,
            scene=_scene_config_with_jaw(initial_jaw=0.45),
        )
    )

    accepted = env.step(_joint_command(0))
    observed = env.get_sensors()
    assert observed.timestamp_ns == accepted.sensors.timestamp_ns
    assert observed.sim_time_s == accepted.sensors.sim_time_s
    assert observed.safety.cycle_id_echo == accepted.sensors.safety.cycle_id_echo
    assert observed.safety.command_blocked is False

    stale = env.step(_joint_command(0))
    stale_obs = env.get_sensors()
    assert stale_obs.timestamp_ns == stale.sensors.timestamp_ns
    assert stale_obs.safety.command_blocked is True
    assert stale_obs.safety.block_reason == "stale_cycle_id"

    disabled = env.step(_joint_command(1, enable=False))
    disabled_obs = env.get_sensors()
    assert disabled_obs.timestamp_ns == disabled.sensors.timestamp_ns
    assert disabled_obs.safety.command_blocked is True
    assert disabled_obs.safety.block_reason == "disabled"


def test_sofa_environment_returns_contact_and_wrench_from_backend_state() -> None:
    backend = _TestBackend(
        "test://scene.json",
        tool_wrench=Vec3(x=1.5, y=-0.25, z=0.75),
        in_contact=True,
    )
    env = SofaEnvironment(
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: backend,
    )
    cfg = EnvConfig(
        seed=21,
        scene=_scene_config_with_jaw(initial_jaw=0.12),
    )
    env.reset(cfg)
    command = _joint_command(0)
    step = env.step(command)

    assert step.sensors.tool.wrench == Vec3(x=1.5, y=-0.25, z=0.75)
    assert step.sensors.tool.in_contact is True
    observed = env.get_sensors()
    assert observed.tool.wrench == Vec3(x=1.5, y=-0.25, z=0.75)
    assert observed.tool.in_contact is True


def test_sofa_environment_step_outputs_full_forceps_tool_state() -> None:
    expected_wrench = Vec3(x=0.05, y=0.10, z=-0.20)
    backend = _CommandCaptureBackend(
        "test://scene.json",
        tool_wrench=expected_wrench,
        in_contact=True,
    )
    env = SofaEnvironment(
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: backend,
    )
    env.reset(
        EnvConfig(
            seed=22,
            control_rate_hz=125.0,
            frame_rate_hz=25.0,
            scene=_scene_config_with_jaw(initial_jaw=0.25),
        )
    )
    cmd = _joint_command(0, enable=True).model_copy(
        update={"tool_jaw_target": 0.66, "timestamp_ns": 2_000}
    )
    step = env.step(cmd)

    tool = step.sensors.tool
    assert step.sensors.timestamp_ns == 2_000
    assert step.sensors.sim_time_s == pytest.approx(1.0 / 125.0)
    assert step.sensors.safety.cycle_id_echo == cmd.cycle_id
    assert isinstance(tool.pose, Pose)
    assert isinstance(tool.twist, Twist)
    assert tool.jaw == 0.66
    assert tool.wrench == expected_wrench
    assert tool.in_contact is True
    assert tool.pose == Pose(
        position=Vec3(x=0.0, y=0.0, z=0.0),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    assert tool.twist == Twist(
        linear=Vec3(x=0.0, y=0.0, z=0.0),
        angular=Vec3(x=0.0, y=0.0, z=0.0),
    )
    assert backend.last_command is not None
    assert backend.last_command.tool_jaw_target == pytest.approx(0.66)


def test_native_backend_forceps_observer_populates_tool_fields_without_sensor_payload() -> None:
    module = _FakeSofaRuntimeModule(include_capture_camera=False)
    jaw_ref = {"jaw": 0.0}
    observer = build_forceps_observer(dof=_FakeForcepsDof(), jaw_ref=jaw_ref)
    backend = _NativeSofaBackend(
        "fake_runtime",
        module,
        "fake://scene",
        step_dt=0.01,
        tool_state_observer=observer,
        tool_state_jaw_ref=jaw_ref,
    )

    reset = backend.reset(
        EnvConfig(seed=7, scene=_scene_config_with_jaw(initial_jaw=0.21, scene_path="fake://scene.json"))
    )
    tool = reset.sensors.tool
    assert tool.pose.position == Vec3(x=0.25, y=0.5, z=0.75)
    assert tool.twist == Twist(
        linear=Vec3(x=0.01, y=-0.02, z=0.03),
        angular=Vec3(x=-0.01, y=0.02, z=-0.03),
    )
    assert tool.jaw == pytest.approx(0.21)
    assert tool.wrench == Vec3(x=0.4, y=0.5, z=0.6)
    assert tool.in_contact is True

    backend.set_tool_jaw_target(0.88)
    step = backend.step(_joint_command(0))
    assert step.sensors.tool.jaw == pytest.approx(0.88)
    assert step.sensors.tool.in_contact is True
    assert step.sensors.tool.wrench == Vec3(x=0.4, y=0.5, z=0.6)


def test_native_backend_capture_fallback_uses_offscreen_capture_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeSofaRuntimeModule(include_capture_camera=True)
    backend = _NativeSofaBackend("fake_runtime", module, "fake://scene", step_dt=0.01)
    calls: dict[str, int] = {"count": 0}

    def fake_capture(scene_root: Any, *, step_index: int, width: int, height: int) -> bytes | None:
        del scene_root, width, height
        assert step_index >= 0
        calls["count"] += 1
        return b"captured_frame"

    monkeypatch.setattr("auto_surgery.env.sofa_backend._render_camera_frame_to_png", fake_capture)

    reset = backend.reset(
        EnvConfig(
            seed=7,
            scene=_scene_config_with_jaw(initial_jaw=0.0, scene_path="fake://scene.json"),
        )
    )
    assert reset.sensors.cameras[0].frame_rgb == b"captured_frame"
    assert calls["count"] == 1

    step = backend.step(_joint_command(0))
    assert step.sensors.cameras[0].frame_rgb == b"captured_frame"
    assert calls["count"] == 2


def test_native_backend_capture_fallback_no_camera_keeps_frames_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeSofaRuntimeModule(include_capture_camera=False)
    backend = _NativeSofaBackend("fake_runtime", module, "fake://scene", step_dt=0.01)
    calls: dict[str, int] = {"count": 0}

    def fake_capture(scene_root: Any, *, step_index: int, width: int, height: int) -> bytes | None:
        del scene_root, step_index, width, height
        calls["count"] += 1
        return b"captured_frame"

    monkeypatch.setattr("auto_surgery.env.sofa_backend._render_camera_frame_to_png", fake_capture)

    reset = backend.reset(
        EnvConfig(
            seed=7,
            scene=_scene_config_with_jaw(initial_jaw=0.0, scene_path="fake://scene.json"),
        )
    )
    assert reset.sensors.cameras[0].frame_rgb is None
    assert calls["count"] == 0


def test_sofa_environment_mode_dispatch_and_blocking_noop() -> None:
    backend = _CommandCaptureBackend("test://scene.json")
    env = SofaEnvironment(
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: backend,
        scene_config=load_scene_config("configs/scenes/dejavu_brain.yaml"),
    )
    env.reset(EnvConfig(seed=7, scene=_scene_config_with_jaw(initial_jaw=0.0)))

    active = _twist_command(1)
    active_step = env.step(active)
    assert active_step.sensors.safety.command_blocked is False
    assert backend.last_command is not None
    assert backend.last_command.control_mode == ControlMode.CARTESIAN_TWIST
    assert backend.last_command.cartesian_twist is not None
    twist = backend.last_command.cartesian_twist
    assert any(
        v != 0.0
        for v in (
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        )
    )

    blocked = env.step(active)
    assert blocked.sensors.safety.command_blocked is True
    assert blocked.sensors.safety.block_reason == "stale_cycle_id"
    assert blocked.sensors.cameras[0].frame_rgb is None
    assert backend.last_command is not None
    assert backend.last_command.control_mode == ControlMode.CARTESIAN_TWIST
    assert backend.last_command.cartesian_twist is not None
    assert backend.last_command.cartesian_twist.linear.x == 0.0
    assert backend.last_command.cartesian_twist.linear.y == 0.0
    assert backend.last_command.cartesian_twist.linear.z == 0.0

    legacy = _joint_command(3)
    legacy_step = env.step(legacy)
    assert legacy_step.sensors.safety.command_blocked is False
    assert backend.last_command is not None
    assert backend.last_command.control_mode == ControlMode.CARTESIAN_TWIST
    assert backend.last_command.cartesian_twist is not None
    assert legacy_step.sensors.safety.motion_enabled is True

    unsupported = RobotCommand(
        timestamp_ns=3_000,
        cycle_id=4,
        control_mode=ControlMode.JOINT_POSITION,
        joint_positions={"jaw": 0.2},
        enable=True,
        source="test",
    )
    unsupported_step = env.step(unsupported)
    assert unsupported_step.sensors.safety.command_blocked is False
    assert unsupported_step.sensors.safety.motion_enabled is True
    assert backend.last_command is not None
    assert backend.last_command.control_mode == ControlMode.CARTESIAN_TWIST
    assert backend.last_command.cartesian_twist is not None
    assert backend.last_command.cartesian_twist == Twist(
        linear=Vec3(x=0.0, y=0.0, z=0.0),
        angular=Vec3(x=0.0, y=0.0, z=0.0),
    )


def test_robot_command_payload_validation_is_strict() -> None:
    with pytest.raises(ValueError):
        RobotCommand(timestamp_ns=1, cycle_id=0, control_mode=ControlMode.JOINT_POSITION)

    with pytest.raises(ValueError):
        RobotCommand(
            timestamp_ns=2,
            cycle_id=1,
            control_mode=ControlMode.JOINT_POSITION,
            joint_positions={"j0": 0.0},
            cartesian_twist={
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        )


def test_sofa_runtime_discovery_contract_surface() -> None:
    contract = discover_sofa_runtime_contract()
    assert "candidates" in contract
    assert "resolved_module_name" in contract


def test_native_backend_temporary_scene_artifacts_cleanup_on_reset_and_close(tmp_path: Path) -> None:
    module = _FakeSofaRuntimeModule(include_capture_camera=False)
    scene_epoch_one = tmp_path / "auto-surgery-brain-dejavu-epoch-one.scn"
    scene_epoch_one.write_text("<Node />", encoding="utf-8")
    scene_epoch_two = tmp_path / "auto-surgery-brain-dejavu-epoch-two.scn"
    scene_epoch_two.write_text("<Node />", encoding="utf-8")
    scene_epoch_three = tmp_path / "regular-scene.scn"
    scene_epoch_three.write_text("<Node />", encoding="utf-8")

    backend = _NativeSofaBackend(
        "fake_runtime",
        module,
        str(scene_epoch_one),
        step_dt=0.01,
    )

    first_reset = backend.reset(
        EnvConfig(
            seed=1,
            scene=_scene_config_with_jaw(initial_jaw=0.0, scene_path=scene_epoch_one),
        )
    )
    assert first_reset.sensors.cameras[0].frame_rgb is None
    assert scene_epoch_one.exists()

    second_reset = backend.reset(
        EnvConfig(
            seed=2,
            scene=_scene_config_with_jaw(initial_jaw=0.0, scene_path=scene_epoch_two),
        )
    )
    assert second_reset.sensors.cameras[0].frame_rgb is None
    assert not scene_epoch_one.exists()
    assert scene_epoch_two.exists()

    backend.close()
    assert not scene_epoch_two.exists()

    baseline = _NativeSofaBackend(
        "fake_runtime",
        module,
        str(scene_epoch_three),
        step_dt=0.01,
    )
    baseline.reset(EnvConfig(seed=3, scene=_scene_config_with_jaw(initial_jaw=0.0, scene_path=scene_epoch_three)))
    baseline.close()
    assert scene_epoch_three.exists()


def test_sofa_non_stub_raises_when_runtime_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _empty_candidates() -> tuple[str, ...]:
        return ("definitely-not-present-sofa",)

    monkeypatch.setattr(
        "auto_surgery.env.sofa_discovery._module_candidates_for_runtime",
        _empty_candidates,
    )

    module_name, module = resolve_sofa_runtime_import_candidates()
    assert module_name is None
    assert module is None

    with pytest.raises(SofaNotIntegratedError):
        SofaEnvironment(
            sofa_scene_path="test://scene.json",
            sofa_backend_factory=None,
        )
