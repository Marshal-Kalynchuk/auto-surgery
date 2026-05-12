"""Environment-level orchestration over SOFA runtime backends."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from auto_surgery.env.protocol import Environment
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig, SceneConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import SensorBundle
from auto_surgery.env.sofa_backend import (
    SofaRuntimeBackend,
    SofaRuntimeBackendFactory,
    SofaRuntimeContractError,
    SofaNotIntegratedError,
    SofaSceneFactory,
    _StubRuntimeBackend,
    build_sofa_runtime_backend,
)
from auto_surgery.env.sim import StubSimEnvironment


class SofaEnvironment(Environment):
    """Adapter implementing the `Environment` protocol for SOFA or stub fallback."""

    def __init__(
        self,
        *,
        sofa_scene_path: str | None = None,
        sofa_scene_factory: SofaSceneFactory | None = None,
        fallback_to_stub: bool = True,
        sofa_import_hint: str = "pip install sofa-python3 (or your vendor's SOFA bindings)",
        sofa_backend_factory: SofaRuntimeBackendFactory | None = None,
        step_dt: float = 0.01,
        action_applier: Callable[[Any, RobotCommand], None] | None = None,
        pre_init_hooks: list[Callable[[Any, EnvConfig], None]] | None = None,
        scene_config: SceneConfig | None = None,
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
                "fallback_to_stub=False requires sofa_scene_path, sofa_scene_factory, "
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

    def reset(self, config: EnvConfig) -> SceneGraph:
        return self._backend.reset(config)

    @property
    def sofa_scene_root(self) -> Any:
        """Return the live `Sofa.Core.Node` root for native SOFA runs (valid after `reset()`)."""

        backend = self._backend
        scene = getattr(backend, "_scene_handle", None)
        if scene is None:
            raise SofaNotIntegratedError(
                "No native SOFA scene root is available (stub backend, or reset() was not called)."
            )
        return scene

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
