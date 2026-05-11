"""SOFA environment adapter with staged runtime integration."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any, Protocol

from auto_surgery.env.protocol import Environment
from auto_surgery.env.sim import StubSimEnvironment
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import SensorBundle

DEFAULT_SOFA_MODULE_CANDIDATES = ("sofa", "Sofa", "sofa.core", "Sofa.Core")


class SofaRuntimeContractError(RuntimeError):
    """Raised when discovered SOFA runtime is missing expected API methods."""


class SofaNotIntegratedError(RuntimeError):
    """Raised when SOFA is requested without a usable runtime adapter."""


class SofaRuntimeBackend(Protocol):
    """Minimal runtime backend contract used by `SofaEnvironment`."""

    def reset(self, config: EnvConfig) -> SceneGraph: ...

    def step(self, action: RobotCommand) -> StepResult: ...

    def get_sensors(self) -> SensorBundle: ...

    def get_scene(self) -> SceneGraph: ...


SofaRuntimeBackendFactory = Callable[[str, dict[str, Any] | None], SofaRuntimeBackend]


def _module_candidates_for_runtime() -> tuple[str, ...]:
    """Return ordered SOFA module import candidates."""

    return DEFAULT_SOFA_MODULE_CANDIDATES


def resolve_sofa_runtime_import_candidates(
    *, candidates: tuple[str, ...] | None = None
) -> tuple[str, Any] | tuple[None, None]:
    """Resolve the first importable SOFA module candidate."""

    for candidate in candidates or _module_candidates_for_runtime():
        try:
            return candidate, importlib.import_module(candidate)
        except ImportError:
            continue
    return None, None


def discover_sofa_runtime_contract() -> dict[str, Any]:
    """Collect module discovery metadata for docs/tests."""

    module_name, module = resolve_sofa_runtime_import_candidates()
    return {
        "candidates": tuple(_module_candidates_for_runtime()),
        "resolved_module_name": module_name,
        "resolved_module_path": getattr(module, "__file__", None) if module else None,
    }


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


class _StubRuntimeBackend:
    """Protocol adapter that forwards to `StubSimEnvironment`."""

    def __init__(self, stub: StubSimEnvironment) -> None:
        self._stub = stub

    def reset(self, config: EnvConfig) -> SceneGraph:
        return self._stub.reset(config)

    def step(self, action: RobotCommand) -> StepResult:
        return self._stub.step(action)

    def get_sensors(self) -> SensorBundle:
        return self._stub.get_sensors()

    def get_scene(self) -> SceneGraph:
        return self._stub.get_scene()


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
        scene_path: str,
    ) -> None:
        self._module_name = module_name
        self._scene_path = scene_path
        self._frame_index = 0
        self._scene_handle: Any | None = None
        self._latest_scene = SceneGraph(
            frame_index=0, events=[{"phase": "created", "backend": module_name}]
        )
        self._latest_sensors = SensorBundle(
            timestamp_ns=0,
            clock_source="sofa",
            modalities={"backend": module_name},
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

    def reset(self, config: EnvConfig) -> SceneGraph:
        if self._loader is None:
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
        self._scene_handle = _call_runtime_callable(self._loader, self._scene_path)
        _call_runtime_callable(self._resetter, self._scene_handle, config)
        self._latest_scene = self._extract_scene()
        return self._latest_scene

    def step(self, action: RobotCommand) -> StepResult:
        if self._scene_handle is None:
            raise SofaRuntimeContractError("step() called before reset().")
        if self._stepper is None:
            raise SofaRuntimeContractError(
                "SOFA runtime loaded but no stepping method found. "
                f"Expected one of {self._STEP_NAMES}."
            )
        _call_runtime_callable(self._stepper, self._scene_handle, action)
        self._frame_index += 1
        self._latest_scene = self._extract_scene()
        self._latest_sensors = self._extract_sensors()
        return StepResult(
            next_scene=self._latest_scene,
            sensor_observation=self._latest_sensors,
            info={"backend": self._module_name, "frame_index": self._frame_index},
        )

    def get_sensors(self) -> SensorBundle:
        return self._latest_sensors.model_copy(deep=True)

    def get_scene(self) -> SceneGraph:
        return self._latest_scene.model_copy(deep=True)

    def _extract_sensors(self) -> SensorBundle:
        if self._sensor_reader is None:
            return SensorBundle(
                timestamp_ns=0,
                clock_source="sofa",
                modalities={"backend": self._module_name, "frame_index": self._frame_index},
            )
        payload = _call_runtime_callable(self._sensor_reader, self._scene_handle)
        if isinstance(payload, SensorBundle):
            return payload
        if isinstance(payload, dict):
            timestamp = int(payload.get("timestamp_ns", 0))
            clock_source = str(payload.get("clock_source", "sofa"))
            modalities = dict(payload.get("modalities", {}))
            return SensorBundle(
                timestamp_ns=timestamp,
                clock_source=clock_source,
                modalities=modalities,
            )
        return SensorBundle(
            timestamp_ns=0,
            clock_source="sofa",
            modalities={"backend": self._module_name, "frame_index": self._frame_index},
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
    sofa_scene_path: str,
    sofa_import_hint: str = "pip install sofa-python3 (or your vendor's SOFA bindings)",
    *,
    candidates: tuple[str, ...] | None = None,
    extra: dict[str, Any] | None = None,
    ) -> SofaRuntimeBackend:
    """Instantiate a runtime backend from discovered SOFA module."""

    module_name, module = resolve_sofa_runtime_import_candidates(candidates=candidates)
    if module is None:
        candidates_text = ", ".join(candidates or _module_candidates_for_runtime())
        raise SofaNotIntegratedError(
            "SOFA runtime module was not importable in non-stub mode. "
            f"Checked: {candidates_text}. Hint: {sofa_import_hint}"
        )
    if not isinstance(sofa_scene_path, str) or not sofa_scene_path.strip():
        raise SofaNotIntegratedError("sofa_scene_path is required for non-stub execution.")
    if module_name is None:
        raise SofaNotIntegratedError("Resolved SOFA module name was unexpectedly empty.")
    return _NativeSofaBackend(module_name, module, sofa_scene_path)


class SofaEnvironment(Environment):
    """Adapter implementing the `Environment` protocol for SOFA or stub fallback."""

    def __init__(
        self,
        *,
        sofa_scene_path: str | None = None,
        fallback_to_stub: bool = True,
        sofa_import_hint: str = "pip install sofa-python3 (or your vendor's SOFA bindings)",
        sofa_backend_factory: SofaRuntimeBackendFactory | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        extra_runtime = extra or {}

        if fallback_to_stub:
            warnings.warn(
                "SofaEnvironment fallback_to_stub=True: using StubSimEnvironment backend. "
                "This is expected in Stage-0 until vendor integration is wired.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._backend: SofaRuntimeBackend = _StubRuntimeBackend(StubSimEnvironment())
            return

        if sofa_scene_path is None:
            raise SofaNotIntegratedError(
                "fallback_to_stub=False requires a valid sofa_scene_path for SOFA scene loading."
            )

        try:
            if sofa_backend_factory is not None:
                self._backend = sofa_backend_factory(sofa_scene_path, extra_runtime)
            else:
                self._backend = build_sofa_runtime_backend(
                    sofa_scene_path,
                    sofa_import_hint,
                    extra=extra_runtime,
                )
        except (SofaRuntimeContractError, SofaNotIntegratedError) as exc:
            raise SofaNotIntegratedError(
                "Failed to initialize SOFA backend. Install/point a compatible runtime or pass "
                "sofa_backend_factory for your adapter contract."
            ) from exc

    def reset(self, config: EnvConfig) -> SceneGraph:
        return self._backend.reset(config)

    def step(self, action: RobotCommand) -> StepResult:
        result = self._backend.step(action)
        modalities = dict(result.sensor_observation.modalities)
        modalities["command_echo"] = action.model_dump()
        return StepResult(
            next_scene=result.next_scene,
            sensor_observation=result.sensor_observation.model_copy(
                update={
                    "timestamp_ns": action.timestamp_ns,
                    "clock_source": result.sensor_observation.clock_source,
                    "modalities": modalities,
                }
            ),
            info=result.info,
        )

    def get_sensors(self) -> SensorBundle:
        return self._backend.get_sensors()

    def get_scene(self) -> SceneGraph:
        return self._backend.get_scene()
