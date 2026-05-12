"""Environment protocol — sim/real parity boundary."""

from typing import Protocol, runtime_checkable

from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import Contact, JointState, SensorBundle
from auto_surgery.schemas.scene import SceneGraph


@runtime_checkable
class Environment(Protocol):
    """Typed sim / real parity boundary."""

    def reset(self, config: EnvConfig) -> StepResult: ...

    def step(self, command: RobotCommand) -> StepResult: ...

    def get_joint_state(self) -> JointState: ...

    def get_contacts(self) -> list[Contact]: ...


@runtime_checkable
class EnvironmentWithSensors(Protocol):
    """Optional sensory-readback convenience surface.

    For SOFA environments, this is expected to mirror the most recent
    contract-safe `step()` output after capture masking and safety updates.
    """

    def get_sensors(self) -> SensorBundle: ...


@runtime_checkable
class EnvironmentWithScene(Protocol):
    """Optional scene snapshot surface, not part of the strict contract."""

    def get_scene(self) -> SceneGraph: ...
