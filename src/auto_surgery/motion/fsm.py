"""Finite-state machine for per-tick primitive execution."""

from __future__ import annotations

from dataclasses import dataclass

from auto_surgery.schemas.commands import Pose
from auto_surgery.schemas.results import StepResult
from auto_surgery.motion.primitives import Approach, Dwell, Probe, Primitive, _probe_retract_time
from auto_surgery.motion.sequencer import _Sequencer


@dataclass
class _ActivePrimitive:
    primitive: Primitive
    started_at_tick: int
    started_at_pose_scene: Pose
    started_at_jaw: float
    duration_s: float
    elapsed_s: float = 0.0
    contact_was_in: bool = False
    in_post_contact_phase: bool = False
    post_contact_started_at_s: float = 0.0


_REALISED_RECORD = tuple[Primitive, int, int, bool]


class _Fsm:
    def __init__(self, sequencer: _Sequencer) -> None:
        self._sequencer = sequencer
        self._active: _ActivePrimitive | None = None
        self._realised: list[_REALISED_RECORD] = []
        self._last_finished_by_contact = False

    def reset(self) -> None:
        self._active = None
        self._realised = []
        self._last_finished_by_contact = False

    def step(self, last_step: StepResult, last_jaw: float) -> _ActivePrimitive:
        if self._active is None or self._active_finished(last_step):
            self._record_finished_if_any(last_step.sim_step_index)
            next_primitive = self._sequencer.next_primitive(last_step, last_jaw)
            if next_primitive is None:
                next_primitive = Dwell(duration_s=1e9, jaw_target_start=None, jaw_target_end=None)
            self._active = _ActivePrimitive(
                primitive=next_primitive,
                started_at_tick=last_step.sim_step_index,
                started_at_pose_scene=last_step.sensors.tool.pose,
                started_at_jaw=last_jaw,
                duration_s=float(next_primitive.duration_s),
                contact_was_in=bool(last_step.sensors.tool.in_contact),
            )
        self._active.elapsed_s += float(last_step.dt)
        return self._active

    def _active_finished(self, last_step: StepResult) -> bool:
        active = self._active
        if active is None:
            return False

        self._last_finished_by_contact = False
        contact_now = bool(last_step.sensors.tool.in_contact)
        contact_rising = contact_now and not active.contact_was_in
        active.contact_was_in = contact_now

        match active.primitive:
            case Probe() if active.in_post_contact_phase:
                elapsed_since_contact = active.elapsed_s - active.post_contact_started_at_s
                hold = float(active.primitive.hold_after_contact_s)
                retract_time = _probe_retract_time(
                    active.primitive.retract_distance_m,
                    peak_retract_speed_m_per_s=float(active.primitive.retract_peak_speed_m_per_s),
                )
                return elapsed_since_contact >= hold + retract_time
            case Probe() if not active.in_post_contact_phase and contact_rising:
                active.in_post_contact_phase = True
                active.post_contact_started_at_s = active.elapsed_s
                return False
            case Approach() if active.primitive.end_on_contact and contact_rising:
                self._last_finished_by_contact = True
                return True
            case _:
                return active.elapsed_s >= active.duration_s

    def _record_finished_if_any(self, tick: int) -> None:
        if self._active is None:
            return
        self._realised.append(
            (
                self._active.primitive,
                self._active.started_at_tick,
                int(tick),
                bool(self._last_finished_by_contact),
            ),
        )
        self._active = None
        self._last_finished_by_contact = False

    def finalize(self, last_step: StepResult) -> None:
        self._record_finished_if_any(last_step.sim_step_index)

    @property
    def completed(self) -> tuple[_REALISED_RECORD, ...]:
        return tuple(self._realised)
