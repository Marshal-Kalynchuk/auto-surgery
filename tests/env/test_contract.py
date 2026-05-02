from __future__ import annotations

from auto_surgery.env.protocol import Environment
from auto_surgery.env.real import RealEnvironment
from auto_surgery.env.sim import StubSimEnvironment
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig


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
