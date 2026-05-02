"""Real robot environment — delegates to injected implementation when wired."""

from __future__ import annotations

from typing import TYPE_CHECKING

from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import SensorBundle

if TYPE_CHECKING:
    from auto_surgery.env.protocol import Environment


class RealEnvironmentNotWiredError(RuntimeError):
    """Raised when neuroArm hooks are not configured."""


class RealEnvironment:
    """Delegates to an injected `Environment` (vendor SDK or test double)."""

    def __init__(self, impl: Environment | None = None) -> None:
        self._impl = impl

    def _require_impl(self) -> Environment:
        if self._impl is None:
            raise RealEnvironmentNotWiredError(
                "RealEnvironment has no implementation. Pass impl=... with neuroArm "
                "teleop + safety boundary integration, or use StubSimEnvironment in simulation."
            )
        return self._impl

    def reset(self, config: EnvConfig) -> SceneGraph:
        return self._require_impl().reset(config)

    def step(self, action: RobotCommand) -> StepResult:
        return self._require_impl().step(action)

    def get_sensors(self) -> SensorBundle:
        return self._require_impl().get_sensors()

    def get_scene(self) -> SceneGraph:
        return self._require_impl().get_scene()
