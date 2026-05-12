"""SOFA runtime backend adapters."""

from __future__ import annotations

from collections.abc import Callable
import io
from typing import Any, Protocol

from auto_surgery.schemas.commands import Pose, Quaternion, RobotCommand, Twist, Vec3
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import (
    CameraIntrinsics,
    CameraView,
    Contact,
    SafetyStatus,
    SensorBundle,
    ToolState,
    JointState,
)
from auto_surgery.env.sofa_discovery import (
    _module_candidates_for_runtime,
    resolve_sofa_runtime_import_candidates,
)
from auto_surgery.env.sofa_rgb_native import render_frame_to_rgb


class SofaRuntimeContractError(RuntimeError):
    """Raised when discovered SOFA runtime is missing expected API methods."""


class SofaNotIntegratedError(RuntimeError):
    """Raised when SOFA is requested without a usable runtime adapter."""


class SofaRuntimeBackend(Protocol):
    """Minimal runtime backend contract used by `SofaEnvironment`."""

    def reset(self, config: EnvConfig) -> StepResult: ...

    def step(self, action: RobotCommand) -> StepResult: ...

    def set_control_rate_hz(self, control_rate_hz: float) -> None: ...

    def set_initial_jaw(self, jaw: float) -> None: ...

    def set_tool_jaw_target(self, jaw: float) -> None: ...

    def get_sensors(self) -> SensorBundle: ...

    def get_scene(self) -> SceneGraph: ...

    def get_joint_state(self) -> JointState: ...

    def get_contacts(self) -> list[Contact]: ...


SofaRuntimeBackendFactory = Callable[[str, dict[str, Any] | None], SofaRuntimeBackend]
SofaSceneFactory = Callable[[Any, EnvConfig], Any]


def _first_callable(objects: list[Any], names: tuple[str, ...]) -> Callable[..., Any] | None:
    for name in names:
        for obj in objects:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn
    return None


def _call_runtime_callable(fn: Callable[..., Any], *args: Any) -> Any:
    """Best-effort invocation helper for vendor bindings with inconsistent signatures."""

    if fn is None:
        return None
    try:
        return fn(*args)
    except TypeError:
        if len(args) <= 0:
            raise
        try:
            return fn(args[0])
        except TypeError:
            try:
                return fn()
            except TypeError:
                raise


_OFFSCREEN_CAMERA_NAME = "auto_surgery_offscreen_camera"
_CAPTURE_WIDTH = 950
_CAPTURE_HEIGHT = 700


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _render_camera_frame_to_png(
    scene_root: Any,
    *,
    step_index: int,
    width: int = _CAPTURE_WIDTH,
    height: int = _CAPTURE_HEIGHT,
) -> bytes | None:
    """Render one capture frame to PNG bytes when an SOFA offscreen camera is available."""
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        rgb = render_frame_to_rgb(
            scene_root,
            step_index=step_index,
            width=width,
            height=height,
        )
        with io.BytesIO() as buffer:
            Image.fromarray(rgb).save(buffer, format="PNG")
            return buffer.getvalue()
    except (AttributeError, OSError, TypeError, ValueError):
        return None


class _NativeSofaBackend:
    """Thin adapter around a discovered SOFA runtime module."""

    _LOADER_NAMES = ("load", "load_scene", "loadScene", "load_from_file")
    _STEP_NAMES = ("step", "animate", "simulate", "advance", "tick")
    _RESET_NAMES = ("init", "initialize", "reset")
    _SCENE_NAMES = ("snapshot", "get_scene_graph", "getScene", "scene_graph")
    _SENSOR_NAMES = ("snapshot_sensors", "get_sensors", "read_sensors", "sensors")

    def __init__(
        self,
        module_name: str,
        module: Any,
        scene_path: str | None,
        *,
        scene_factory: SofaSceneFactory | None = None,
        step_dt: float = 0.01,
        action_applier: Callable[[Any, RobotCommand], None] | None = None,
        pre_init_hooks: list[Callable[[Any, EnvConfig], None]] | None = None,
    ) -> None:
        self._module = module
        self._module_name = module_name
        self._scene_path = scene_path
        self._scene_factory = scene_factory
        self._step_dt = float(step_dt)
        self._action_applier = action_applier
        self._pre_init_hooks = tuple(pre_init_hooks or ())
        self._frame_index = 0
        self._jaw_target = 0.0
        self._capture_camera: Any | None = None
        self._capture_enabled = False
        self._scene_handle: Any | None = None
        self._latest_scene = SceneGraph(
            frame_index=0, events=[{"phase": "created", "backend": module_name}]
        )
        self._latest_sensors = SensorBundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            tool=ToolState(
                pose=Pose(
                    position=Vec3(x=0.0, y=0.0, z=0.0),
                    rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
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
                    camera_id="sofa_default",
                    timestamp_ns=0,
                    extrinsics=Pose(
                        position=Vec3(x=0.0, y=0.0, z=0.0),
                        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                    ),
                    intrinsics=CameraIntrinsics(
                        fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1
                    ),
                )
            ],
            safety=SafetyStatus(
                motion_enabled=False,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=0,
            ),
        )
        self._loader, self._stepper, self._resetter, self._snapshotter, self._sensor_reader = (
            self._resolve_runtime_functions(module)
        )

    @classmethod
    def _resolve_runtime_functions(
        cls,
        module: Any,
    ) -> tuple[
        Callable[..., Any] | None,
        Callable[..., Any] | None,
        Callable[..., Any] | None,
        Callable[..., Any] | None,
        Callable[..., Any] | None,
    ]:
        candidate_objects = [module]
        nested = getattr(module, "Simulation", None)
        if nested is not None:
            candidate_objects.append(nested)
        nested = getattr(module, "Core", None)
        if nested is not None:
            candidate_objects.append(nested)
        loader = _first_callable(candidate_objects, cls._LOADER_NAMES)
        stepper = _first_callable(candidate_objects, cls._STEP_NAMES)
        resetter = _first_callable(candidate_objects, cls._RESET_NAMES)
        snapshotter = _first_callable(candidate_objects, cls._SCENE_NAMES)
        sensor_reader = _first_callable(candidate_objects, cls._SENSOR_NAMES)
        return loader, stepper, resetter, snapshotter, sensor_reader

    def reset(self, config: EnvConfig) -> StepResult:
        self.set_control_rate_hz(config.control_rate_hz)
        self.set_initial_jaw(float(config.scene.initial_jaw))
        if self._scene_factory is None and self._loader is None:
            raise SofaRuntimeContractError(
                "SOFA runtime loaded but no scene loader found. "
                f"Expected one of {self._LOADER_NAMES}."
            )
        if self._resetter is None:
            raise SofaRuntimeContractError(
                "SOFA runtime loaded but no reset method found. "
                f"Expected one of {self._RESET_NAMES}."
            )
        self._frame_index = 0
        if self._scene_factory is not None:
            # Build the scene graph in-process via the provided Python factory.
            core = getattr(self._module, "Core", None)
            node_ctor = getattr(core, "Node", None) if core is not None else None
            if node_ctor is None:
                raise SofaNotIntegratedError(
                    "Sofa runtime missing Sofa.Core.Node; cannot construct in-process scene."
                )

            self._scene_handle = node_ctor("root")
            _call_runtime_callable(self._scene_factory, self._scene_handle, config)
        else:
            if self._loader is None or self._scene_path is None:
                raise SofaRuntimeContractError("Scene loading requires scene_path.")
            self._scene_handle = _call_runtime_callable(self._loader, self._scene_path)

        for hook in self._pre_init_hooks:
            hook(self._scene_handle, config)
        _call_runtime_callable(self._resetter, self._scene_handle, config)
        self._resolve_capture_camera()
        self._latest_scene = self._extract_scene()
        self._latest_sensors = self._extract_sensors()
        return StepResult(
            sensors=self._latest_sensors,
            dt=0.0,
            sim_step_index=0,
            is_capture_tick=True,
        )

    def set_control_rate_hz(self, control_rate_hz: float) -> None:
        if control_rate_hz <= 0:
            raise ValueError("control_rate_hz must be positive.")
        self._step_dt = 1.0 / float(control_rate_hz)

    def set_initial_jaw(self, jaw: float) -> None:
        self._jaw_target = float(jaw)
        self._latest_sensors = self._latest_sensors.model_copy(
            update={"tool": self._latest_sensors.tool.model_copy(update={"jaw": self._jaw_target})}
        )

    def set_tool_jaw_target(self, jaw: float) -> None:
        self._jaw_target = float(jaw)

    def _resolve_capture_camera(self) -> Any | None:
        if self._scene_handle is None:
            self._capture_camera = None
            self._capture_enabled = False
            return None
        get_object = getattr(self._scene_handle, "getObject", None)
        if callable(get_object):
            camera = get_object(_OFFSCREEN_CAMERA_NAME)
            if camera is not None:
                self._capture_camera = camera
                self._capture_enabled = True
                return camera
        self._capture_camera = None
        self._capture_enabled = False
        return None

    def _capture_camera_frame(self) -> bytes | None:
        if self._scene_handle is None:
            self._capture_enabled = False
            return None
        if self._capture_camera is None:
            self._resolve_capture_camera()
        if not self._capture_enabled:
            return None
        width = _as_int(
            getattr(self._capture_camera, "widthViewport", _CAPTURE_WIDTH),
            _CAPTURE_WIDTH,
        )
        height = _as_int(
            getattr(self._capture_camera, "heightViewport", _CAPTURE_HEIGHT),
            _CAPTURE_HEIGHT,
        )
        frame_rgb = _render_camera_frame_to_png(
            self._scene_handle,
            step_index=self._frame_index,
            width=width,
            height=height,
        )
        if frame_rgb is None:
            self._capture_enabled = False
            return None
        return frame_rgb

    def step(self, action: RobotCommand) -> StepResult:
        if self._scene_handle is None:
            raise SofaRuntimeContractError("step() called before reset().")
        if self._stepper is None:
            raise SofaRuntimeContractError(
                "SOFA runtime loaded but no stepping method found. "
                f"Expected one of {self._STEP_NAMES}."
            )
        if self._action_applier is not None:
            self._action_applier(self._scene_handle, action)
        try:
            self._stepper(self._scene_handle, action)
        except TypeError:
            self._stepper(self._scene_handle, self._step_dt)
        self._frame_index += 1
        self._latest_scene = self._extract_scene()
        self._latest_sensors = self._extract_sensors()
        return StepResult(
            sensors=self._latest_sensors,
            dt=self._step_dt,
            sim_step_index=self._frame_index,
            is_capture_tick=True,
        )

    def get_sensors(self) -> SensorBundle:
        return self._latest_sensors.model_copy(deep=True)

    def get_scene(self) -> SceneGraph:
        return self._latest_scene.model_copy(deep=True)

    def get_joint_state(self) -> JointState:
        return JointState(positions={}, velocities={})

    def get_contacts(self) -> list[Contact]:
        return []

    def _extract_sensors(self) -> SensorBundle:
        if self._sensor_reader is None:
            frame_rgb = self._capture_camera_frame()
            return SensorBundle(
                timestamp_ns=0,
                sim_time_s=self._frame_index * self._step_dt,
                tool=ToolState(
                    pose=Pose(
                        position=Vec3(x=0.0, y=0.0, z=0.0),
                        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                    ),
                    twist=Twist(
                        linear=Vec3(x=0.0, y=0.0, z=0.0),
                        angular=Vec3(x=0.0, y=0.0, z=0.0),
                    ),
                    jaw=self._jaw_target,
                    wrench=Vec3(x=0.0, y=0.0, z=0.0),
                    in_contact=False,
                ),
                cameras=[
                    CameraView(
                        camera_id="sofa_default",
                        timestamp_ns=0,
                        extrinsics=Pose(
                            position=Vec3(x=0.0, y=0.0, z=0.0),
                            rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                        ),
                        intrinsics=CameraIntrinsics(
                            fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1
                        ),
                        frame_rgb=frame_rgb,
                    )
                ],
                safety=SafetyStatus(
                    motion_enabled=False,
                    command_blocked=False,
                    block_reason=None,
                    cycle_id_echo=-1,
                ),
            )
        payload = _call_runtime_callable(self._sensor_reader, self._scene_handle)
        if isinstance(payload, SensorBundle):
            return payload
        if isinstance(payload, dict):
            timestamp = int(payload.get("timestamp_ns", 0))
            sim_time_s = float(payload.get("sim_time_s", self._frame_index * self._step_dt))
            raw_tool = payload.get("tool")
            if isinstance(raw_tool, dict):
                tool = ToolState.model_validate(raw_tool)
            else:
                tool = ToolState(
                    pose=Pose(
                        position=Vec3(x=0.0, y=0.0, z=0.0),
                        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                    ),
                    twist=Twist(
                        linear=Vec3(x=0.0, y=0.0, z=0.0),
                        angular=Vec3(x=0.0, y=0.0, z=0.0),
                    ),
                    jaw=self._jaw_target,
                    wrench=Vec3(x=0.0, y=0.0, z=0.0),
                    in_contact=False,
                )

            raw_cameras = payload.get("cameras", [])
            if isinstance(raw_cameras, list) and raw_cameras:
                parsed_cameras: list[CameraView] = []
                for item in raw_cameras:
                    if isinstance(item, CameraView):
                        parsed_cameras.append(item)
                    elif isinstance(item, dict):
                        parsed_cameras.append(CameraView.model_validate(item))
                cameras = parsed_cameras if parsed_cameras else [
                    CameraView(
                        camera_id="sofa_default",
                        timestamp_ns=timestamp,
                        extrinsics=Pose(
                            position=Vec3(x=0.0, y=0.0, z=0.0),
                            rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                        ),
                        intrinsics=CameraIntrinsics(
                            fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1
                        ),
                        frame_rgb=None,
                    )
                ]
            else:
                cameras = [
                    CameraView(
                        camera_id="sofa_default",
                        timestamp_ns=timestamp,
                        extrinsics=Pose(
                            position=Vec3(x=0.0, y=0.0, z=0.0),
                            rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                        ),
                        intrinsics=CameraIntrinsics(
                            fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1
                        ),
                        frame_rgb=None,
                    )
                ]
            raw_safety = payload.get("safety")
            if isinstance(raw_safety, dict):
                safety = SafetyStatus.model_validate(raw_safety)
            else:
                safety = SafetyStatus(
                    motion_enabled=False,
                    command_blocked=False,
                    block_reason=None,
                    cycle_id_echo=-1,
                )
            return SensorBundle(
                timestamp_ns=timestamp,
                sim_time_s=sim_time_s,
                tool=tool,
                cameras=cameras,
                safety=safety,
            )
        frame_rgb = self._capture_camera_frame()
        return SensorBundle(
            timestamp_ns=0,
            sim_time_s=self._frame_index * self._step_dt,
            tool=ToolState(
                pose=Pose(
                    position=Vec3(x=0.0, y=0.0, z=0.0),
                    rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
                twist=Twist(
                    linear=Vec3(x=0.0, y=0.0, z=0.0),
                    angular=Vec3(x=0.0, y=0.0, z=0.0),
                ),
                jaw=self._jaw_target,
                wrench=Vec3(x=0.0, y=0.0, z=0.0),
                in_contact=False,
            ),
            cameras=[
                CameraView(
                    camera_id="sofa_default",
                    timestamp_ns=0,
                    extrinsics=Pose(
                        position=Vec3(x=0.0, y=0.0, z=0.0),
                        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                    ),
                    intrinsics=CameraIntrinsics(
                        fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1
                    ),
                    frame_rgb=frame_rgb,
                )
            ],
            safety=SafetyStatus(
                motion_enabled=False,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=-1,
            ),
        )

    def _extract_scene(self) -> SceneGraph:
        if self._snapshotter is not None and self._scene_handle is not None:
            payload = _call_runtime_callable(self._snapshotter, self._scene_handle)
            if isinstance(payload, SceneGraph):
                return payload.model_copy(deep=True)
            if isinstance(payload, dict):
                return SceneGraph(**payload)

        return SceneGraph(
            frame_index=self._frame_index,
            events=[{"backend": self._module_name, "scene_path": self._scene_path}],
        )


def build_sofa_runtime_backend(
    sofa_scene_path: str | None,
    sofa_import_hint: str = "pip install sofa-python3 (or your vendor's SOFA bindings)",
    step_dt: float = 0.01,
    action_applier: Callable[[Any, RobotCommand], None] | None = None,
    *,
    sofa_scene_factory: SofaSceneFactory | None = None,
    candidates: tuple[str, ...] | None = None,
    extra: dict[str, Any] | None = None,
    pre_init_hooks: list[Callable[[Any, EnvConfig], None]] | None = None,
) -> SofaRuntimeBackend:
    """Instantiate a runtime backend from discovered SOFA module."""

    # extra is accepted to preserve explicit factory hooks and to centralize
    # extension points for future backend adapters.
    del extra

    module_name, module = resolve_sofa_runtime_import_candidates(candidates=candidates)
    if module is None:
        candidates_text = ", ".join(candidates or _module_candidates_for_runtime())
        raise SofaNotIntegratedError(
            "SOFA runtime module was not importable in non-stub mode. "
            f"Checked: {candidates_text}. Hint: {sofa_import_hint}"
        )
    if sofa_scene_path is None and sofa_scene_factory is None:
        raise SofaNotIntegratedError(
            "Non-stub execution requires either sofa_scene_path or sofa_scene_factory."
        )
    if sofa_scene_path is not None and (
        not isinstance(sofa_scene_path, str) or not sofa_scene_path.strip()
    ):
        raise SofaNotIntegratedError("sofa_scene_path must be a non-empty string.")
    if module_name is None:
        raise SofaNotIntegratedError("Resolved SOFA module name was unexpectedly empty.")

    # Verify we can reach the SofaPython3 python bindings (Sofa.Core.Node construction).
    core = getattr(module, "Core", None)
    if core is None or not hasattr(core, "Node"):
        raise SofaNotIntegratedError(
            f"Sofa runtime appears incomplete: missing Sofa.Core.Node. Hint: {sofa_import_hint}"
        )

    return _NativeSofaBackend(
        module_name,
        module,
        sofa_scene_path,
        scene_factory=sofa_scene_factory,
        step_dt=step_dt,
        action_applier=action_applier,
        pre_init_hooks=pre_init_hooks,
    )

