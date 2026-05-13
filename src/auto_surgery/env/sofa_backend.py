"""SOFA runtime backend adapters."""

from __future__ import annotations

from contextlib import suppress
from collections.abc import Callable
import io
import re
from typing import Any, Protocol, Sequence
from pathlib import Path
import numpy as np

from auto_surgery.schemas.commands import Pose, Quaternion, RobotCommand, Twist, Vec3
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneConfig, SceneGraph, VisualToneAugmentation
from auto_surgery.schemas.sensors import (
    CameraIntrinsics,
    CameraView,
    Contact,
    SafetyStatus,
    SensorBundle,
    ToolState,
    JointState,
)
from auto_surgery.env.sofa_tools import build_forceps_observer
from auto_surgery.env.sofa_discovery import (
    _module_candidates_for_runtime,
    resolve_sofa_runtime_import_candidates,
)
from auto_surgery.env.sofa_rgb_native import compensate_principal_point, render_frame_to_rgb
from auto_surgery.env.sofa_scenes.dejavu_paths import (
    DEJAVU_ROOT_PLACEHOLDER,
    resolve_dejavu_root,
    render_dejavu_scene_template,
)
from auto_surgery.randomization.scn_template import render_scene_template

_AUTO_SCENE_PREFIX = "auto-surgery-brain-dejavu-"
_AUTO_WARP_PREFIX = "auto-surgery-tissue-warp-"


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


def _resolve_tool_state_observer(
    *,
    extra: dict[str, Any] | None,
    jaw_ref: dict[str, float],
) -> Callable[[Any], ToolState] | None:
    if not isinstance(extra, dict):
        return None
    tool_id = str(extra.get("tool_id", "")).strip().lower()
    if tool_id not in {"forceps", "dejavu_forceps"}:
        return None
    provided = extra.get("tool_state_observer")
    if provided is not None and callable(provided):
        return provided
    return build_forceps_observer(dof=None, jaw_ref=jaw_ref)


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


def _apply_tone_augmentation(
    frame_bytes: bytes,
    tone: VisualToneAugmentation,
) -> bytes:
    if tone.is_identity():
        return frame_bytes

    try:
        from PIL import Image
    except ImportError:
        return frame_bytes

    try:
        with io.BytesIO(frame_bytes) as input_buffer:
            with Image.open(input_buffer) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return frame_bytes

    rgb = np.clip(rgb * float(tone.brightness_scale), 0.0, 1.0)
    rgb = np.clip(
        (rgb - 0.5) * float(tone.contrast_scale) + 0.5,
        0.0,
        1.0,
    )

    gamma = float(tone.gamma)
    if gamma > 0.0:
        rgb = np.power(rgb, 1.0 / gamma)

    luma = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    rgb = np.stack(
        (
            luma + (rgb[:, :, 0] - luma) * float(tone.saturation_scale),
            luma + (rgb[:, :, 1] - luma) * float(tone.saturation_scale),
            luma + (rgb[:, :, 2] - luma) * float(tone.saturation_scale),
        ),
        axis=2,
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    with io.BytesIO() as output_buffer:
        Image.fromarray((rgb * 255.0).astype(np.uint8)).save(output_buffer, format="PNG")
        return output_buffer.getvalue()


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
        tool_state_observer: Callable[[Any], ToolState] | None = None,
        tool_state_jaw_ref: dict[str, float] | None = None,
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
        self._tool_state_jaw_ref = tool_state_jaw_ref or {"jaw": 0.0}
        self._tool_state_jaw_ref["jaw"] = self._jaw_target
        self._tool_state_observer = tool_state_observer
        self._capture_camera: Any | None = None
        self._capture_enabled = False
        self._scene_handle: Any | None = None
        self._rendered_scene_paths: set[Path] = set()
        self._scene_camera_extrinsics = self._default_camera_extrinsics()
        self._scene_camera_intrinsics = self._default_camera_intrinsics()
        self._scene_background_rgb = (0.0, 0.0, 0.0)
        self._scene_tone_augmentation = VisualToneAugmentation()
        self._latest_scene = SceneGraph(
            frame_index=0, events=[{"phase": "created", "backend": module_name}]
        )
        default_tool_state = self._default_tool_state()
        self._latest_sensors = SensorBundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            tool=default_tool_state,
            cameras=self._camera_default_view(timestamp_ns=0, frame_rgb=None),
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

    @staticmethod
    def _default_camera_extrinsics() -> Pose:
        return Pose(
            position=Vec3(x=0.0, y=0.0, z=0.0),
            rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        )

    @staticmethod
    def _default_camera_intrinsics() -> CameraIntrinsics:
        return CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1)

    @staticmethod
    def _scene_camera_path_from_config(config: EnvConfig) -> str | None:
        tissue_scene_path = getattr(config.scene, "tissue_scene_path", None)
        if tissue_scene_path is None:
            return None
        return str(tissue_scene_path.resolve())

    @staticmethod
    def _coerce_background_rgb(rgb: Sequence[float] | tuple[float, float, float]) -> tuple[float, float, float]:
        values = list(rgb or (0.0, 0.0, 0.0))
        if len(values) < 3:
            values = values + [0.0] * (3 - len(values))
        return (float(values[0]), float(values[1]), float(values[2]))

    @staticmethod
    def _scene_template_candidate(raw_xml: str, candidate: Path) -> bool:
        has_placeholders = "{{" in raw_xml and "}}" in raw_xml
        return candidate.suffix == ".template" or has_placeholders

    @staticmethod
    def _is_owned_temporary_path(path: Path) -> bool:
        name = path.name
        return name.startswith(_AUTO_SCENE_PREFIX) or name.startswith(_AUTO_WARP_PREFIX)

    def _register_rendered_path(self, path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            resolved = path.expanduser()
        self._rendered_scene_paths.add(resolved)

    def _scene_temporary_artifacts(self, rendered_scene_path: Path) -> set[Path]:
        try:
            scene_text = rendered_scene_path.read_text(encoding="utf-8")
        except OSError:
            return set()

        mesh_paths: set[Path] = set()
        scene_parent = rendered_scene_path.parent
        for match in re.finditer(r'filename="([^"]+)"', scene_text):
            raw_path = match.group(1).strip()
            if not raw_path:
                continue
            mesh_path = Path(raw_path).expanduser()
            if not mesh_path.is_absolute():
                mesh_path = (scene_parent / mesh_path).resolve()
            if (
                mesh_path.suffix.lower() == ".obj"
                and self._is_owned_temporary_path(mesh_path)
                and mesh_path.exists()
            ):
                mesh_paths.add(mesh_path)
        return mesh_paths

    @staticmethod
    def _coerce_intrinsics(cfg: CameraIntrinsics) -> CameraIntrinsics:
        if cfg.fx <= 1.0 and cfg.fy <= 1.0 and cfg.cx == 0.0 and cfg.cy == 0.0:
            return CameraIntrinsics(
                fx=1.0,
                fy=1.0,
                cx=0.0,
                cy=0.0,
                width=_as_int(cfg.width, 1),
                height=_as_int(cfg.height, 1),
            )
        return cfg

    def _default_tool_state(self) -> ToolState:
        return ToolState(
            pose=Pose(
                position=Vec3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
            twist=Twist(
                linear=Vec3(x=0.0, y=0.0, z=0.0),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            jaw=float(self._tool_state_jaw_ref.get("jaw", 0.0)),
            wrench=Vec3(x=0.0, y=0.0, z=0.0),
            in_contact=False,
        )

    def _observed_tool_state(self) -> ToolState:
        if self._tool_state_observer is None:
            return self._default_tool_state()
        try:
            tool = self._tool_state_observer(self._scene_handle)
        except Exception:
            return self._default_tool_state()
        if isinstance(tool, ToolState):
            return tool
        return self._default_tool_state()

    def _sync_tool_jaw(self, jaw: float) -> None:
        self._tool_state_jaw_ref["jaw"] = float(jaw)

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

    def _cleanup_rendered_scene_paths(self) -> None:
        for path in tuple(self._rendered_scene_paths):
            with suppress(OSError):
                path.unlink()
            self._rendered_scene_paths.discard(path)

    def _resolve_scene_path_for_runtime(self, configured_path: str, scene: SceneConfig) -> str:
        candidate = Path(configured_path).expanduser()
        raw_xml: str
        try:
            raw_xml = candidate.read_text(encoding="utf-8")
        except (OSError, TypeError):
            return str(candidate)
        if self._scene_template_candidate(raw_xml, candidate):
            return str(
                render_scene_template(
                    scene,
                    dejavu_root=resolve_dejavu_root(),
                    template_path=candidate,
                )
            )
        if DEJAVU_ROOT_PLACEHOLDER not in raw_xml:
            return str(candidate)
        return render_dejavu_scene_template(candidate)

    def _resolve_scene_path(self, config: EnvConfig) -> str | None:
        requested = self._scene_camera_path_from_config(config)
        if requested is None:
            return self._scene_path
        requested_path = Path(requested).expanduser()
        if requested_path.exists():
            resolved = self._resolve_scene_path_for_runtime(requested, config.scene)
        else:
            resolved = str(requested_path)
        if resolved != requested:
            resolved_path = Path(resolved)
            if self._is_owned_temporary_path(resolved_path):
                self._register_rendered_path(resolved_path)
                self._rendered_scene_paths.update(self._scene_temporary_artifacts(resolved_path))
            else:
                self._register_rendered_path(resolved_path)
        elif self._is_owned_temporary_path(requested_path):
            self._register_rendered_path(requested_path)
            self._rendered_scene_paths.update(self._scene_temporary_artifacts(requested_path))
        return resolved

    def reset(self, config: EnvConfig) -> StepResult:
        self.set_control_rate_hz(config.control_rate_hz)
        self.set_initial_jaw(float(config.scene.tool.initial_jaw))
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
        self._cleanup_rendered_scene_paths()
        self._scene_camera_extrinsics = config.scene.camera_extrinsics_scene
        self._scene_camera_intrinsics = self._coerce_intrinsics(config.scene.camera_intrinsics)
        self._scene_background_rgb = self._coerce_background_rgb(config.scene.lighting.background_rgb)
        self._scene_tone_augmentation = config.scene.tone_augmentation
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
            scene_path = self._resolve_scene_path(config)
            if self._loader is None or scene_path is None:
                raise SofaRuntimeContractError("Scene loading requires scene_path.")
            self._scene_path = scene_path
            self._scene_handle = _call_runtime_callable(self._loader, scene_path)

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
        self._sync_tool_jaw(self._jaw_target)
        self._latest_sensors = self._latest_sensors.model_copy(
            update={"tool": self._latest_sensors.tool.model_copy(update={"jaw": self._jaw_target})}
        )

    def set_tool_jaw_target(self, jaw: float) -> None:
        self._jaw_target = float(jaw)
        self._sync_tool_jaw(self._jaw_target)

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
        if self._scene_camera_intrinsics is not None:
            frame_rgb = compensate_principal_point(
                frame_rgb,
                width=width,
                height=height,
                camera_intrinsics=self._scene_camera_intrinsics,
                background_rgb=self._scene_background_rgb,
            )
        frame_rgb = _apply_tone_augmentation(frame_rgb, tone=self._scene_tone_augmentation)
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

    def close(self) -> None:
        self._cleanup_rendered_scene_paths()

    def _camera_default_view(
        self,
        *,
        timestamp_ns: int,
        frame_rgb: bytes | None,
    ) -> list[CameraView]:
        return [
            CameraView(
                camera_id="sofa_default",
                timestamp_ns=timestamp_ns,
                extrinsics=self._scene_camera_extrinsics,
                intrinsics=self._scene_camera_intrinsics,
                frame_rgb=frame_rgb,
            )
        ]

    def _extract_sensors(self) -> SensorBundle:
        if self._sensor_reader is None:
            frame_rgb = self._capture_camera_frame()
            tool = self._observed_tool_state()
            return SensorBundle(
                timestamp_ns=0,
                sim_time_s=self._frame_index * self._step_dt,
                tool=tool,
                cameras=self._camera_default_view(timestamp_ns=0, frame_rgb=frame_rgb),
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
                tool = self._observed_tool_state()

            raw_cameras = payload.get("cameras", [])
            if isinstance(raw_cameras, list) and raw_cameras:
                parsed_cameras: list[CameraView] = []
                for item in raw_cameras:
                    if isinstance(item, CameraView):
                        parsed_cameras.append(item)
                    elif isinstance(item, dict):
                        parsed_cameras.append(CameraView.model_validate(item))
                cameras = parsed_cameras if parsed_cameras else self._camera_default_view(
                    timestamp_ns=timestamp,
                    frame_rgb=None,
                )
            else:
                cameras = self._camera_default_view(timestamp_ns=timestamp, frame_rgb=None)
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
            tool=self._observed_tool_state(),
            cameras=self._camera_default_view(timestamp_ns=0, frame_rgb=frame_rgb),
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
    tool_state_jaw_ref: dict[str, float] | None = None,
) -> SofaRuntimeBackend:
    """Instantiate a runtime backend from discovered SOFA module."""

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

    tool_state_jaw_ref = tool_state_jaw_ref or {"jaw": 0.0}
    tool_state_observer = _resolve_tool_state_observer(
        extra=extra,
        jaw_ref=tool_state_jaw_ref,
    )

    return _NativeSofaBackend(
        module_name,
        module,
        sofa_scene_path,
        scene_factory=sofa_scene_factory,
        step_dt=step_dt,
        action_applier=action_applier,
        pre_init_hooks=pre_init_hooks,
        tool_state_observer=tool_state_observer,
        tool_state_jaw_ref=tool_state_jaw_ref,
    )

