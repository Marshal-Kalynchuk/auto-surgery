from __future__ import annotations

import pytest
from typing import Any

from auto_surgery.env.action_generators import build_sine_twist_command
from auto_surgery.env.protocol import Environment
from auto_surgery.env.protocol import EnvironmentWithSensors
from auto_surgery.env.real import RealEnvironment
from auto_surgery.env.sofa_backend import _NativeSofaBackend
from auto_surgery.env.sofa import (
    SofaEnvironment,
    SofaNotIntegratedError,
    discover_sofa_runtime_contract,
    resolve_sofa_runtime_import_candidates,
)
from auto_surgery.schemas.commands import ControlMode, Pose, Quaternion, RobotCommand
from auto_surgery.schemas.manifests import DomainRandomizationConfig, EnvConfig, SceneConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph, SlotRecord
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
    state = env.reset(EnvConfig(seed=7))
    assert state.sim_step_index == 0
    cmd = RobotCommand(
        timestamp_ns=100,
        cycle_id=0,
        control_mode=ControlMode.JOINT_POSITION,
        joint_positions={"j0": 0.1},
        enable=True,
    )
    result = env.step(cmd)
    assert result.sensors.timestamp_ns == cmd.timestamp_ns
    if isinstance(env, EnvironmentWithSensors):
        obs = env.get_sensors()
        assert obs.safety.cycle_id_echo == 0
    return 1


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
                {
                    "type": "reset",
                    "seed": config.seed,
                    "domain_randomization": config.domain_randomization.spatial_variation,
                }
            ],
        )
        self._last_accepted_cycle_id = -1
        self._sim_step_index = 0
        self._sim_time_s = 0.0
        self._jaw_target = float(config.scene.initial_jaw)
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
        control_mode=ControlMode.JOINT_POSITION,
        joint_positions={"j0": float(step_index) * 0.01},
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

    def __init__(self, scene_path: str) -> None:
        self._frame_index = 0
        self._control_rate_hz = 250.0
        self._frame_decimation = 1
        self._jaw = 0.0
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
                wrench=Vec3(x=0.0, y=0.0, z=0.0),
                in_contact=False,
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
        self._jaw = float(config.scene.initial_jaw)
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
            tool=self._sensors.tool.model_copy(update={"jaw": self._jaw}),
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


class _CommandCaptureBackend(_TestBackend):
    """Capture the action payload after dispatch and gating."""

    def __init__(self, scene_path: str) -> None:
        super().__init__(scene_path)
        self.last_command: RobotCommand | None = None

    def step(self, action: RobotCommand) -> StepResult:
        self.last_command = action
        return super().step(action)


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


def _twist_command(step_index: int, *, enable: bool = True, cycle_id: int | None = None) -> RobotCommand:
    return build_sine_twist_command(
        step_index,
        base_ns=100_000,
        linear_amplitude=0.05,
        phase_scale=0.1,
    ).model_copy(
        update={
            "enable": enable,
            "cycle_id": step_index if cycle_id is None else cycle_id,
        }
    )


def test_sofa_contract_with_injected_backend_round_trip() -> None:
    env = SofaEnvironment(
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: _TestBackend(scene_path),
    )
    scene = env.reset(
        EnvConfig(
            seed=9,
            domain_randomization=DomainRandomizationConfig(spatial_variation={"scene": "mock"}),
        )
    )
    assert scene.sim_step_index == 0
    cmd = RobotCommand(
        timestamp_ns=1_000_000,
        cycle_id=0,
        control_mode=ControlMode.JOINT_POSITION,
        joint_positions={"j0": 0.05},
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
        scene=SceneConfig(initial_jaw=0.42),
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
            scene=SceneConfig(initial_jaw=0.45),
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

    reset = backend.reset(EnvConfig(seed=7))
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

    reset = backend.reset(EnvConfig(seed=7))
    assert reset.sensors.cameras[0].frame_rgb is None
    assert calls["count"] == 0


def test_sofa_environment_mode_dispatch_and_blocking_noop() -> None:
    backend = _CommandCaptureBackend("test://scene.json")
    env = SofaEnvironment(
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: backend,
    )
    env.reset(EnvConfig(seed=7))

    active = _twist_command(1)
    active_step = env.step(active)
    assert active_step.sensors.safety.command_blocked is False
    assert backend.last_command is not None
    assert backend.last_command.control_mode == ControlMode.CARTESIAN_TWIST
    assert backend.last_command.cartesian_twist is not None
    assert backend.last_command.cartesian_twist.linear.x != 0.0

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
    assert backend.last_command.control_mode == ControlMode.JOINT_POSITION
    assert backend.last_command.joint_positions == {"j0": 0.03}


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
