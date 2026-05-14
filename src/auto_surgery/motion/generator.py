"""Public motion generator runtime for surgical trajectory playback."""

from __future__ import annotations

from dataclasses import dataclass

from auto_surgery.motion.fsm import _Fsm
from auto_surgery.motion.primitives import (
    ActivePrimitive,
    tip_desired_pose_scene,
    _jaw_target,
    _time_to_fraction,
)
from auto_surgery.motion.sequencer import _Sequencer
from auto_surgery.schemas.commands import ControlFrame, ControlMode, RobotCommand, SafetyMetadata
from auto_surgery.schemas.motion import MotionGeneratorConfig
from auto_surgery.schemas.results import StepResult


@dataclass(frozen=True)
class RealisedPrimitive:
    primitive: object
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

        active_fsm = self._fsm.step(last_step, self._last_jaw_commanded)
        scene_geometry = getattr(self._sequencer, "_scene_geometry", None)

        active = ActivePrimitive(
            primitive=active_fsm.primitive,
            started_at_pose_scene=active_fsm.started_at_pose_scene,
            started_at_jaw=active_fsm.started_at_jaw,
            duration_s=float(active_fsm.duration_s),
            elapsed_s=float(active_fsm.elapsed_s),
            contact_was_in=bool(active_fsm.contact_was_in),
        )

        next_pose_scene = tip_desired_pose_scene(
            active,
            last_step,
            scene_geometry=scene_geometry,
        )

        primitive = active.primitive
        duration_s = float(active.duration_s)
        elapsed_s = float(active.elapsed_s)

        tau = _time_to_fraction(elapsed_s, duration_s)
        jaw_target = _jaw_target(primitive=primitive, active=active, tau=tau)

        motion_shaping = getattr(self._motion_config, "motion_shaping", None)
        motion_shaping_enabled = bool(
            bool(getattr(self._motion_config, "motion_shaping_enabled", False)) or motion_shaping is not None
        )
        motion_shaping = motion_shaping if motion_shaping_enabled else None

        self._last_jaw_commanded = jaw_target
        self._cycle_id += 1

        return RobotCommand(
            timestamp_ns=last_step.sensors.timestamp_ns + int(last_step.dt * 1_000_000_000),
            cycle_id=self._cycle_id,
            control_mode=ControlMode.CARTESIAN_POSE,
            cartesian_pose_target=next_pose_scene,
            frame=ControlFrame.SCENE,
            tool_jaw_target=jaw_target,
            enable=True,
            source="scripted",
            motion_shaping=motion_shaping,
            motion_shaping_enabled=motion_shaping_enabled,
            safety=SafetyMetadata(
                clamped_linear=False,
                clamped_angular=False,
                biased_linear=False,
                biased_angular=False,
                scaled_by=None,
                signed_distance_to_envelope_mm=None,
                signed_distance_to_surface_mm=None,
                pose_error_norm_mm=None,
                pose_error_norm_rad=None,
            ),
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
