"""SOFA runtime backend adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from auto_surgery.env.sim import StubSimEnvironment
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import SensorBundle
from auto_surgery.env.sofa_discovery import (
    _module_candidates_for_runtime,
    resolve_sofa_runtime_import_candidates,
)


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

