"""Environment protocol — sim/real parity boundary."""

from typing import Protocol, runtime_checkable

from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import SensorBundle


@runtime_checkable
class Environment(Protocol):
    """Same contract for SimEnvironment and RealEnvironment (architecture §8.3.2)."""

    def reset(self, config: EnvConfig) -> SceneGraph: ...

    def step(self, action: RobotCommand) -> StepResult: ...

    def get_sensors(self) -> SensorBundle: ...

    def get_scene(self) -> SceneGraph: ...
