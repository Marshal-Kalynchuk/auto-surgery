"""Environment-level orchestration over SOFA runtime backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto_surgery.env.protocol import Environment
from auto_surgery.schemas.commands import ControlMode, RobotCommand, Twist, Vec3
from auto_surgery.schemas.manifests import EnvConfig, SceneConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import Contact, JointState, SafetyStatus, SensorBundle
from auto_surgery.env.sofa_backend import (
    SofaRuntimeBackend,
    SofaRuntimeBackendFactory,
    SofaRuntimeContractError,
    SofaNotIntegratedError,
    SofaSceneFactory,
    build_sofa_runtime_backend,
)


_ZERO_TWIST = Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0))


class SofaEnvironment(Environment):
    """Adapter implementing the `Environment` protocol for SOFA backends."""

    def __init__(
        self,
        *,
        sofa_scene_path: str | None = None,
        sofa_scene_factory: SofaSceneFactory | None = None,
        sofa_import_hint: str = "pip install sofa-python3 (or your vendor's SOFA bindings)",
        sofa_backend_factory: SofaRuntimeBackendFactory | None = None,
        step_dt: float = 0.01,
        action_applier: Callable[[Any, RobotCommand], None] | None = None,
        pre_init_hooks: list[Callable[[Any, EnvConfig], None]] | None = None,
        scene_config: SceneConfig | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        extra_runtime = extra or {}
        resolved_path = sofa_scene_path
        resolved_factory = sofa_scene_factory
        resolved_applier = action_applier

        if scene_config is not None:
            if scene_config.scene_xml_path:
                resolved_path = scene_config.scene_xml_path
                resolved_factory = None
            elif resolved_factory is None and resolved_path is None:
                from auto_surgery.env.sofa_registry import resolve_scene_factory

                resolved_factory = resolve_scene_factory(scene_config.scene_id)
            if resolved_applier is None:
                from auto_surgery.env.sofa_tools import resolve_tool_action_applier

                resolved_applier = resolve_tool_action_applier(scene_config.tool_id)

        if resolved_path is None and resolved_factory is None:
            raise SofaNotIntegratedError(
                "SofaEnvironment requires sofa_scene_path, sofa_scene_factory, "
                "or a SceneConfig that resolves to one of those."
            )
        if resolved_path is not None and resolved_factory is not None:
            raise SofaNotIntegratedError(
                "Provide only one of sofa_scene_path / scene_xml_path or sofa_scene_factory."
            )
        if resolved_factory is not None and sofa_backend_factory is not None:
            raise SofaNotIntegratedError(
                "When using sofa_scene_factory, do not pass sofa_backend_factory "
                "(it only accepts a scene_path)."
            )
        if pre_init_hooks and sofa_backend_factory is not None:
            raise SofaNotIntegratedError(
                "pre_init_hooks cannot be combined with sofa_backend_factory; the custom backend "
                "owns init order."
            )

        try:
            if sofa_backend_factory is not None:
                if resolved_path is None:
                    raise SofaNotIntegratedError(
                        "sofa_backend_factory requires a concrete XML scene path."
                    )
                self._backend = sofa_backend_factory(resolved_path, extra_runtime)
            else:
                self._backend = build_sofa_runtime_backend(
                    resolved_path,
                    sofa_import_hint,
                    step_dt,
                    resolved_applier,
                    extra=extra_runtime,
                    sofa_scene_factory=resolved_factory,
                    pre_init_hooks=pre_init_hooks,
                )
        except (SofaRuntimeContractError, SofaNotIntegratedError) as exc:
            raise SofaNotIntegratedError(
                "Failed to initialize SOFA backend. Install/point a compatible runtime or pass "
                "sofa_backend_factory for your adapter contract."
            ) from exc
        self._sim_step_index = 0
        self._sim_time_s = 0.0
        self._last_accepted_cycle_id = -1
        self._control_rate_hz = 250.0
        self._frame_rate_hz = 30.0
        self._frame_decimation = max(1, round(self._control_rate_hz / self._frame_rate_hz))
        self._latest_step_sensors: SensorBundle | None = None

    def _dt(self) -> float:
        return 1.0 / self._control_rate_hz

    @staticmethod
    def _noop_command(command: RobotCommand) -> RobotCommand:
        """Convert blocked command into deterministic no-motion command."""
        return command.model_copy(
            update={
                "control_mode": ControlMode.CARTESIAN_TWIST,
                "enable": False,
                "tool_jaw_target": None,
                "cartesian_twist": _ZERO_TWIST,
                "cartesian_pose_target": None,
                "joint_positions": None,
                "joint_velocities": None,
            }
        )

    def _dispatch_command(self, command: RobotCommand) -> RobotCommand:
        if command.control_mode == ControlMode.CARTESIAN_TWIST:
            return command
        if command.control_mode == ControlMode.CARTESIAN_POSE:
            return command
        if command.control_mode == ControlMode.JOINT_POSITION:
            return command
        if command.control_mode == ControlMode.JOINT_VELOCITY:
            return command
        raise ValueError(
            f"Unsupported control mode for SOFA dispatch: {command.control_mode.value}"
        )

    def _mask_camera_frames(self, sensors: SensorBundle, *, is_capture_tick: bool) -> SensorBundle:
        if is_capture_tick:
            return sensors
        masked_cameras = [camera.model_copy(update={"frame_rgb": None}) for camera in sensors.cameras]
        return sensors.model_copy(deep=True, update={"cameras": masked_cameras})

    def _apply_backend_control_rate(self) -> None:
        if hasattr(self._backend, "set_control_rate_hz"):
            self._backend.set_control_rate_hz(self._control_rate_hz)

    def _apply_initial_jaw(self, jaw: float) -> None:
        if hasattr(self._backend, "set_initial_jaw"):
            self._backend.set_initial_jaw(jaw)

    def _apply_tool_jaw(self, command: RobotCommand) -> None:
        if command.tool_jaw_target is None:
            return
        if hasattr(self._backend, "set_tool_jaw_target"):
            self._backend.set_tool_jaw_target(command.tool_jaw_target)

    def _command_block_reason(self, command: RobotCommand) -> tuple[bool, str | None]:
        if command.cycle_id <= self._last_accepted_cycle_id:
            return True, "stale_cycle_id"
        if not command.enable:
            return True, "disabled"
        return False, None

    def _safety(self, command: RobotCommand, blocked: bool, reason: str | None) -> SafetyStatus:
        return SafetyStatus(
            motion_enabled=command.enable and not blocked,
            command_blocked=blocked,
            block_reason=reason,
            cycle_id_echo=command.cycle_id,
        )

    def _step_sensors(self) -> SensorBundle:
        return self._backend.get_sensors()

    def reset(self, config: EnvConfig) -> StepResult:
        self._sim_step_index = 0
        self._sim_time_s = 0.0
        self._last_accepted_cycle_id = -1
        self._control_rate_hz = float(config.control_rate_hz)
        self._frame_rate_hz = float(config.frame_rate_hz)
        self._frame_decimation = max(1, round(self._control_rate_hz / self._frame_rate_hz))
        self._apply_backend_control_rate()
        self._apply_initial_jaw(config.scene.initial_jaw)

        self._backend.reset(config)
        sensors = self._step_sensors().model_copy(
            update={
                "timestamp_ns": 0,
                "sim_time_s": 0.0,
                "safety": SafetyStatus(
                    motion_enabled=False,
                    command_blocked=False,
                    block_reason=None,
                    cycle_id_echo=-1,
                ),
            }
        )
        self._latest_step_sensors = sensors
        return StepResult(
            sensors=sensors,
            dt=0.0,
            sim_step_index=0,
            is_capture_tick=True,
        )

    @property
    def sofa_scene_root(self) -> Any:
        """Return the live `Sofa.Core.Node` root for native SOFA runs (valid after `reset()`)."""

        backend = self._backend
        scene = getattr(backend, "_scene_handle", None)
        if scene is None:
            raise SofaNotIntegratedError(
                "No native SOFA scene root is available. Call reset() before requesting it."
            )
        return scene

    def step(self, action: RobotCommand) -> StepResult:
        blocked, reason = self._command_block_reason(action)
        command = self._dispatch_command(self._noop_command(action) if blocked else action)
        self._apply_tool_jaw(command)
        self._backend.step(command)
        self._sim_step_index += 1
        self._sim_time_s += self._dt()
        if not blocked:
            self._last_accepted_cycle_id = action.cycle_id
        is_capture_tick = self._sim_step_index % self._frame_decimation == 0
        step_sensors = self._mask_camera_frames(self._step_sensors(), is_capture_tick=is_capture_tick)
        step_sensors = step_sensors.model_copy(
            update={
                "safety": self._safety(action, blocked=blocked, reason=reason),
                "timestamp_ns": action.timestamp_ns,
                "sim_time_s": self._sim_time_s,
            },
        )
        self._latest_step_sensors = step_sensors
        return StepResult(
            sensors=step_sensors,
            dt=self._dt(),
            sim_step_index=self._sim_step_index,
            is_capture_tick=is_capture_tick,
        )

    def get_sensors(self) -> SensorBundle:
        if self._latest_step_sensors is None:
            return self._backend.get_sensors()
        return self._latest_step_sensors.model_copy(deep=True)

    def get_joint_state(self) -> JointState:
        return self._backend.get_joint_state()

    def get_contacts(self) -> list[Contact]:
        return self._backend.get_contacts()

    def get_scene(self) -> SceneGraph:
        return self._backend.get_scene()
