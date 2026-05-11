"""SOFA environment adapter scaffold implementing the protocol."""

from __future__ import annotations

import warnings
from typing import Any

from auto_surgery.env.protocol import Environment
from auto_surgery.env.sim import StubSimEnvironment
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import SensorBundle


class SofaNotIntegratedError(RuntimeError):
    """Raised when SOFA is requested but the adapter has no real bindings."""


class SofaEnvironment(Environment):
    """Adapter implementing the `Environment` protocol for SOFA (scaffold)."""

    def __init__(
        self,
        *,
        sofa_scene_path: str | None = None,
        fallback_to_stub: bool = True,
        sofa_import_hint: str = "pip install sofa-python3 (or your vendor's SOFA bindings)",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._sofa_scene_path = sofa_scene_path
        self._fallback_to_stub = fallback_to_stub
        self._sofa_import_hint = sofa_import_hint
        self._extra = extra or {}
        self._stub = StubSimEnvironment() if fallback_to_stub else None

        if not fallback_to_stub:
            raise SofaNotIntegratedError(
                "SofaEnvironment is currently a scaffold. Real SOFA runtime integration is not wired in "
                "this repo yet. Either enable `fallback_to_stub=True` for Stage-0 development, or "
                f"integrate SOFA bindings first. Hint: {sofa_import_hint}"
            )

        warnings.warn(
            "SofaEnvironment fallback_to_stub=True: using StubSimEnvironment backend. "
            "This is expected until SOFA runtime integration is implemented.",
            RuntimeWarning,
            stacklevel=2,
        )

    def reset(self, config: EnvConfig) -> SceneGraph:
        assert self._stub is not None
        return self._stub.reset(config)

    def step(self, action: RobotCommand) -> StepResult:
        assert self._stub is not None
        return self._stub.step(action)

    def get_sensors(self) -> SensorBundle:
        assert self._stub is not None
        return self._stub.get_sensors()

    def get_scene(self) -> SceneGraph:
        assert self._stub is not None
        return self._stub.get_scene()

