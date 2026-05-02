"""Deterministic stub simulator — SOFA-compatible API surface, interim physics."""

from __future__ import annotations

import time

from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.logging import SafetyDecision
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph, SlotRecord
from auto_surgery.schemas.sensors import SensorBundle


class StubSimEnvironment:
    """Deterministic stand-in until SOFA is wrapped behind the same API."""

    def __init__(self) -> None:
        self._frame_index: int = 0
        self._scene: SceneGraph = SceneGraph(slots=[])

    def reset(self, config: EnvConfig) -> SceneGraph:
        self._frame_index = 0
        tool_slot = SlotRecord(slot_id="tool_0", pose={"x": 0.0, "y": 0.0, "z": 0.1})
        self._scene = SceneGraph(
            frame_index=self._frame_index,
            slots=[tool_slot],
            events=[{"type": "reset", "seed": config.seed, **config.domain_randomization}],
        )
        return self._scene.model_copy(deep=True)

    def get_sensors(self) -> SensorBundle:
        now_ns = time.time_ns()
        return SensorBundle(
            timestamp_ns=now_ns,
            clock_source="monotonic",
            modalities={
                "kinematics": {"joint_0": 0.0},
                "stereo_rgb_shape": [480, 640, 3],
            },
        )

    def get_scene(self) -> SceneGraph:
        return self._scene.model_copy(deep=True)

    def step(self, action: RobotCommand) -> StepResult:
        self._frame_index += 1
        self._scene = self._scene.model_copy(
            update={"frame_index": self._frame_index},
            deep=True,
        )
        obs = SensorBundle(
            timestamp_ns=action.timestamp_ns,
            clock_source="monotonic",
            modalities={"command_echo": action.model_dump()},
        )
        return StepResult(
            next_scene=self._scene.model_copy(deep=True),
            sensor_observation=obs,
            info={},
        )

    def gate_command(self, cmd: RobotCommand) -> SafetyDecision:
        veto = bool(cmd.mode_flags.get("force_veto"))
        return SafetyDecision(ok=not veto, gate_action="veto" if veto else "pass", reason_codes=[])
