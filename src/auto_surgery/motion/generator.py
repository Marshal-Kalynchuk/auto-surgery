"""Public motion generator runtime for surgical trajectory playback."""

from __future__ import annotations

from dataclasses import dataclass

from auto_surgery.motion.fsm import _Fsm
from auto_surgery.motion.primitives import Primitive, _evaluate
from auto_surgery.motion.sequencer import _Sequencer
from auto_surgery.schemas.commands import ControlFrame, ControlMode, RobotCommand
from auto_surgery.schemas.motion import MotionGeneratorConfig
from auto_surgery.schemas.results import StepResult


@dataclass(frozen=True)
class RealisedPrimitive:
    primitive: Primitive
    started_at_tick: int
    ended_at_tick: int
    early_terminated: bool


class SurgicalMotionGenerator:
    def __init__(self, motion_config: MotionGeneratorConfig, scene_config: object) -> None:
        self._motion_config = motion_config
        self._scene_config = scene_config
        self._sequencer = _Sequencer(motion_config, scene_config)
        self._fsm = _Fsm(self._sequencer)
        self._cycle_id = -1
        self._reset_called = False
        self._last_jaw_commanded = self._initial_jaw()
        self._finalized = False

    def _initial_jaw(self) -> float:
        tool = getattr(self._scene_config, "tool", None)
        if tool is None:
            return 0.0
        return float(getattr(tool, "initial_jaw", 0.0))

    def reset(self, initial_step: StepResult) -> RobotCommand:
        self._sequencer.reset(initial_step)
        self._fsm.reset()
        self._cycle_id = -1
        self._last_jaw_commanded = self._initial_jaw()
        self._reset_called = True
        self._finalized = False
        return self.next_command(initial_step)

    def next_command(self, last_step: StepResult) -> RobotCommand:
        if not self._reset_called:
            raise RuntimeError("reset() must be called before next_command()")

        active = self._fsm.step(last_step, self._last_jaw_commanded)
        output = _evaluate(active, last_step)
        self._last_jaw_commanded = output.jaw_target
        self._cycle_id += 1
        return RobotCommand(
            timestamp_ns=last_step.sensors.timestamp_ns + int(last_step.dt * 1_000_000_000),
            cycle_id=self._cycle_id,
            control_mode=ControlMode.CARTESIAN_TWIST,
            cartesian_twist=output.twist_camera,
            frame=ControlFrame.CAMERA,
            tool_jaw_target=output.jaw_target,
            enable=True,
            source="scripted",
        )

    def finalize(self, last_step: StepResult) -> None:
        if self._finalized:
            return
        self._fsm.finalize(last_step)
        self._finalized = True

    @property
    def realised_sequence(self) -> tuple[RealisedPrimitive, ...]:
        return tuple(
            RealisedPrimitive(
                primitive=primitive,
                started_at_tick=started_at,
                ended_at_tick=ended_at,
                early_terminated=early_terminated,
            )
            for primitive, started_at, ended_at, early_terminated in self._fsm.completed
        )
