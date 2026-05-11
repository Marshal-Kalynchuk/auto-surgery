from __future__ import annotations

import pytest

from auto_surgery.env.protocol import Environment
from auto_surgery.env.real import RealEnvironment
from auto_surgery.env.sim import StubSimEnvironment
from auto_surgery.env.sofa import (
    SofaEnvironment,
    SofaNotIntegratedError,
    discover_sofa_runtime_contract,
    resolve_sofa_runtime_import_candidates,
)
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph, SlotRecord
from auto_surgery.schemas.sensors import SensorBundle


def _rollout(env: Environment) -> int:
    scene = env.reset(EnvConfig(seed=7))
    assert scene.slots
    cmd = RobotCommand(timestamp_ns=100, joint_positions={"j0": 0.1})
    result = env.step(cmd)
    assert result.sensor_observation.timestamp_ns == cmd.timestamp_ns
    obs = env.get_sensors()
    assert obs.clock_source
    return env.get_scene().frame_index


def test_stub_sim_matches_environment_protocol() -> None:
    sim = StubSimEnvironment()
    assert isinstance(sim, Environment)
    assert _rollout(sim) >= 1


def test_real_environment_delegates_to_impl() -> None:
    inner = StubSimEnvironment()
    real = RealEnvironment(impl=inner)
    assert isinstance(real, Environment)
    assert _rollout(real) >= 1


class _TestBackend:
    """Simple backend used for protocol contract tests."""

    def __init__(self, scene_path: str) -> None:
        self._frame_index = 0
        self._scene = SceneGraph(
            frame_index=0,
            slots=[SlotRecord(slot_id="tool", pose={"x": 0.0, "y": 0.0, "z": 0.0})],
            events=[{"phase": "init", "scene_path": scene_path}],
        )
        self._sensors = SensorBundle(
            timestamp_ns=0, clock_source="test", modalities={"backend": "test"}
        )

    def reset(self, config: EnvConfig) -> SceneGraph:
        self._frame_index = 0
        self._scene = self._scene.model_copy(
            update={"frame_index": 0, "events": [{"reset_seed": config.seed}]}
        )
        return self._scene

    def step(self, action: RobotCommand) -> StepResult:
        self._frame_index += 1
        self._sensors = SensorBundle(
            timestamp_ns=action.timestamp_ns,
            clock_source="test",
            modalities={"backend": "test", "frame_index": self._frame_index},
        )
        self._scene = self._scene.model_copy(
            update={
                "frame_index": self._frame_index,
                "events": [{"step": self._frame_index}],
            }
        )
        return StepResult(
            next_scene=self._scene,
            sensor_observation=self._sensors,
            info={"frame": self._frame_index},
        )

    def get_sensors(self) -> SensorBundle:
        return self._sensors

    def get_scene(self) -> SceneGraph:
        return self._scene


def test_sofa_contract_with_injected_backend_round_trip() -> None:
    env = SofaEnvironment(
        sofa_scene_path="test://scene.json",
        fallback_to_stub=False,
        sofa_backend_factory=lambda scene_path, _extra: _TestBackend(scene_path),
    )
    scene = env.reset(EnvConfig(seed=9, domain_randomization={"scene": "mock"}))
    assert scene.frame_index == 0
    cmd = RobotCommand(timestamp_ns=1_000_000, joint_positions={"j0": 0.05})
    step = env.step(cmd)
    assert step.sensor_observation.modalities["command_echo"]["timestamp_ns"] == 1_000_000
    assert step.sensor_observation.timestamp_ns == cmd.timestamp_ns
    assert env.get_scene().frame_index == 1
    assert env.get_sensors().clock_source == "test"


def test_sofa_runtime_discovery_contract_surface() -> None:
    contract = discover_sofa_runtime_contract()
    assert "candidates" in contract
    assert "resolved_module_name" in contract


def test_sofa_non_stub_raises_when_runtime_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _empty_candidates() -> tuple[str, ...]:
        return ("definitely-not-present-sofa",)

    monkeypatch.setattr(
        "auto_surgery.env.sofa._module_candidates_for_runtime",
        _empty_candidates,
    )

    module_name, module = resolve_sofa_runtime_import_candidates()
    assert module_name is None
    assert module is None

    with pytest.raises(SofaNotIntegratedError):
        SofaEnvironment(
            sofa_scene_path="test://scene.json",
            fallback_to_stub=False,
            sofa_backend_factory=None,
        )
