"""Environment-level orchestration over SOFA runtime backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto_surgery.env.protocol import Environment
from auto_surgery.env.sofa_scenes.forceps_assets import _twist_camera_to_scene
from auto_surgery.schemas.commands import (
    ControlFrame,
    ControlMode,
    Pose,
    Quaternion,
    RobotCommand,
    SafetyMetadata,
    Twist,
    Vec3,
)
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
_IDENTITY_POSE = Pose(
    position=Vec3(x=0.0, y=0.0, z=0.0),
    rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
)
_SUPPORTED_CONTROL_MODES = frozenset({ControlMode.CARTESIAN_TWIST, ControlMode.CARTESIAN_POSE})
_FORCEPS_CONTROL_MODES = frozenset({ControlMode.CARTESIAN_TWIST})
_FORCEPS_TOOL_IDS = frozenset({"forceps", "dejavu_forceps"})


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
        self._tool_state_jaw_ref: dict[str, float] = {"jaw": 0.0}
        self._camera_pose: dict[str, Pose] = {"pose": _IDENTITY_POSE}
        self._supported_control_modes: frozenset[ControlMode] = _SUPPORTED_CONTROL_MODES

        extra_runtime = dict(extra or {})
        resolved_path, resolved_factory, resolved_applier, resolved_tool_id = self._resolve_from_scene_config(
            scene_config=scene_config,
            sofa_scene_path=sofa_scene_path,
            sofa_scene_factory=sofa_scene_factory,
            action_applier=action_applier,
        )
        if resolved_tool_id in _FORCEPS_TOOL_IDS:
            self._supported_control_modes = _FORCEPS_CONTROL_MODES
        self._action_applier = resolved_applier

        self._validate_backend_inputs(
            resolved_path=resolved_path,
            resolved_factory=resolved_factory,
            sofa_backend_factory=sofa_backend_factory,
            pre_init_hooks=pre_init_hooks,
        )
        if resolved_tool_id is not None:
            extra_runtime.setdefault("tool_id", resolved_tool_id)

        self._backend = self._build_backend(
            resolved_path=resolved_path,
            resolved_factory=resolved_factory,
            sofa_backend_factory=sofa_backend_factory,
            sofa_import_hint=sofa_import_hint,
            step_dt=step_dt,
            resolved_applier=resolved_applier,
            extra_runtime=extra_runtime,
            pre_init_hooks=pre_init_hooks,
        )

        self._sim_step_index = 0
        self._sim_time_s = 0.0
        self._last_accepted_cycle_id = -1
        self._control_rate_hz = 250.0
        self._frame_rate_hz = 30.0
        self._frame_decimation = max(1, round(self._control_rate_hz / self._frame_rate_hz))
        self._latest_step_sensors: SensorBundle | None = None
        self._workspace_envelope = scene_config.tool.workspace_envelope if scene_config is not None else None

    def _resolve_from_scene_config(
        self,
        *,
        scene_config: SceneConfig | None,
        sofa_scene_path: str | None,
        sofa_scene_factory: SofaSceneFactory | None,
        action_applier: Callable[[Any, RobotCommand], None] | None,
    ) -> tuple[str | None, SofaSceneFactory | None, Callable[[Any, RobotCommand], None] | None, str | None]:
        """Derive scene path, factory, applier, and tool id from optional ``scene_config``.

        Without ``scene_config`` we fall through to the explicit args. With one,
        we fill in any unset slot from ``scene_config`` and seed the
        camera-pose handle used by the action applier.
        """

        resolved_path = sofa_scene_path
        resolved_factory = sofa_scene_factory
        resolved_applier = action_applier
        resolved_tool_id: str | None = None

        if scene_config is None:
            return resolved_path, resolved_factory, resolved_applier, resolved_tool_id

        if resolved_factory is None and resolved_path is None:
            from auto_surgery.env.sofa_registry import resolve_scene_factory

            try:
                resolved_factory = resolve_scene_factory(scene_config.scene_id)
            except KeyError:
                resolved_path = str(scene_config.tissue_scene_path.resolve())

        if resolved_applier is None:
            from auto_surgery.env.sofa_tools import resolve_tool_action_applier_from_spec

            resolved_applier = resolve_tool_action_applier_from_spec(
                scene_config.tool,
                jaw_ref=self._tool_state_jaw_ref,
                camera_pose_provider=lambda: self._camera_pose["pose"],
            )
        resolved_tool_id = str(scene_config.tool.tool_id).strip().lower()
        self._camera_pose["pose"] = scene_config.camera_extrinsics_scene.model_copy(deep=True)
        return resolved_path, resolved_factory, resolved_applier, resolved_tool_id

    @staticmethod
    def _validate_backend_inputs(
        *,
        resolved_path: str | None,
        resolved_factory: SofaSceneFactory | None,
        sofa_backend_factory: SofaRuntimeBackendFactory | None,
        pre_init_hooks: list[Callable[[Any, EnvConfig], None]] | None,
    ) -> None:
        if resolved_path is None and resolved_factory is None:
            raise SofaNotIntegratedError(
                "SofaEnvironment requires sofa_scene_path, sofa_scene_factory, "
                "or a SceneConfig that resolves to one of those."
            )
        if resolved_path is not None and resolved_factory is not None:
            raise SofaNotIntegratedError(
                "Provide only one of sofa_scene_path or sofa_scene_factory."
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

    def _build_backend(
        self,
        *,
        resolved_path: str | None,
        resolved_factory: SofaSceneFactory | None,
        sofa_backend_factory: SofaRuntimeBackendFactory | None,
        sofa_import_hint: str,
        step_dt: float,
        resolved_applier: Callable[[Any, RobotCommand], None] | None,
        extra_runtime: dict[str, Any],
        pre_init_hooks: list[Callable[[Any, EnvConfig], None]] | None,
    ) -> SofaRuntimeBackend:
        try:
            if sofa_backend_factory is not None:
                if resolved_path is None:
                    raise SofaNotIntegratedError(
                        "sofa_backend_factory requires a concrete XML scene path."
                    )
                return sofa_backend_factory(resolved_path, extra_runtime)
            return build_sofa_runtime_backend(
                resolved_path,
                sofa_import_hint,
                step_dt,
                resolved_applier,
                extra=extra_runtime,
                sofa_scene_factory=resolved_factory,
                pre_init_hooks=pre_init_hooks,
                tool_state_jaw_ref=self._tool_state_jaw_ref,
            )
        except (SofaRuntimeContractError, SofaNotIntegratedError) as exc:
            raise SofaNotIntegratedError(
                "Failed to initialize SOFA backend. Install/point a compatible runtime or pass "
                "sofa_backend_factory for your adapter contract."
            ) from exc

    def _dt(self) -> float:
        return 1.0 / self._control_rate_hz

    @staticmethod
    def _noop_command(
        command: RobotCommand,
        *,
        enable: bool | None = None,
    ) -> RobotCommand:
        """Convert blocked or unsupported commands into deterministic no-motion commands."""
        if enable is None:
            enable = command.enable
        return command.model_copy(
            update={
                "control_mode": ControlMode.CARTESIAN_TWIST,
                "enable": enable,
                "tool_jaw_target": None,
                "cartesian_twist": _ZERO_TWIST,
                "cartesian_pose_target": None,
                "joint_positions": None,
                "joint_velocities": None,
            }
        )

    def _dispatch_command(self, command: RobotCommand) -> RobotCommand:
        if command.control_mode in self._supported_control_modes:
            return command
        if self._action_applier is not None:
            return self._noop_command(command)
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
        if (
            command.control_mode != ControlMode.CARTESIAN_TWIST
            or command.cartesian_twist is None
            or command.frame != ControlFrame.CAMERA
            or self._workspace_envelope is None
        ):
            return False, None

        last_sensors = self._latest_step_sensors
        if last_sensors is None:
            last_sensors = self._backend.get_sensors()
        tip_pose_scene = last_sensors.tool.pose
        tip_linear_velocity = _twist_camera_to_scene(command.cartesian_twist, self._camera_pose["pose"])
        dt = self._dt()
        tip_next_scene = Vec3(
            x=float(tip_pose_scene.position.x) + float(tip_linear_velocity[0]) * dt,
            y=float(tip_pose_scene.position.y) + float(tip_linear_velocity[1]) * dt,
            z=float(tip_pose_scene.position.z) + float(tip_linear_velocity[2]) * dt,
        )

        distance_next = float(
            self._workspace_envelope.signed_distance_to_envelope(tip_next_scene)
        )
        near_boundary = 0.5 * float(self._workspace_envelope.outer_margin_mm)

        if distance_next < 0.0:
            command.safety = SafetyMetadata(
                clamped_linear=False,
                clamped_angular=False,
                biased_linear=False,
                biased_angular=False,
                scaled_by=None,
                signed_distance_to_envelope_mm=distance_next,
                signed_distance_to_surface_mm=None,
            )
            return True, "tip_outside_workspace_envelope"

        if 0.0 <= distance_next < near_boundary and near_boundary > 0.0:
            current_distance = float(
                self._workspace_envelope.signed_distance_to_envelope(tip_pose_scene.position)
            )
            # Scale only when moving toward the boundary (distance decreases with time).
            if distance_next < current_distance and current_distance > near_boundary:
                scale = (near_boundary - current_distance) / (distance_next - current_distance)
                scale = max(0.0, min(1.0, float(scale)))
                if scale < 1.0:
                    scaled_linear = Vec3(
                        x=float(command.cartesian_twist.linear.x) * scale,
                        y=float(command.cartesian_twist.linear.y) * scale,
                        z=float(command.cartesian_twist.linear.z) * scale,
                    )
                    command.cartesian_twist = command.cartesian_twist.model_copy(
                        update={"linear": scaled_linear}
                    )
                    scaled_tip_next_scene = Vec3(
                        x=float(tip_pose_scene.position.x) + float(tip_linear_velocity[0] * scale) * dt,
                        y=float(tip_pose_scene.position.y) + float(tip_linear_velocity[1] * scale) * dt,
                        z=float(tip_pose_scene.position.z) + float(tip_linear_velocity[2] * scale) * dt,
                    )
                    command.safety = SafetyMetadata(
                        clamped_linear=True,
                        clamped_angular=False,
                        biased_linear=False,
                        biased_angular=False,
                        scaled_by=scale,
                        signed_distance_to_envelope_mm=float(
                            self._workspace_envelope.signed_distance_to_envelope(scaled_tip_next_scene)
                        ),
                        signed_distance_to_surface_mm=None,
                    )
                    return False, None

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
        self._workspace_envelope = config.scene.tool.workspace_envelope
        self._camera_pose["pose"] = config.scene.camera_extrinsics_scene.model_copy(deep=True)
        self._apply_backend_control_rate()
        self._apply_initial_jaw(config.scene.tool.initial_jaw)

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
        command = self._dispatch_command(
            self._noop_command(action, enable=False) if blocked else action
        )
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

    def close(self) -> None:
        if hasattr(self._backend, "close"):
            self._backend.close()
