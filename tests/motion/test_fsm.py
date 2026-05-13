from __future__ import annotations

from typing import Any

from auto_surgery.motion.fsm import _ActivePrimitive, _Fsm
from auto_surgery.motion.primitives import Approach, Dwell, Probe
from auto_surgery.schemas.commands import Quaternion, Pose, Twist, Vec3
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState


class _StubSequencer:
    def __init__(self, primitives: list[Any]) -> None:
        self._primitives = list(primitives)
        self._index = 0

    def reset(self, initial_step: StepResult) -> None:
        _ = initial_step
        self._index = 0

    def next_primitive(self, last_step: StepResult, last_jaw: float) -> Any | None:
        _ = last_step, last_jaw
        if self._index >= len(self._primitives):
            return None
        primitive = self._primitives[self._index]
        self._index += 1
        return primitive


def _identity_pose() -> Pose:
    return Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _step(*, sim_step_index: int, dt: float, in_contact: bool = False) -> StepResult:
    cam_pose = _identity_pose()
    bundle = SensorBundle(
        timestamp_ns=sim_step_index,
        sim_time_s=sim_step_index * dt,
        tool=ToolState(
            pose=_identity_pose(),
            twist=Twist(
                linear=Vec3(x=0.0, y=0.0, z=0.0),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            jaw=0.0,
            wrench=Vec3(x=0.0, y=0.0, z=0.0),
            in_contact=in_contact,
        ),
        cameras=[
            CameraView(
                camera_id="cam",
                timestamp_ns=sim_step_index,
                extrinsics=cam_pose,
                intrinsics=CameraIntrinsics(fx=1, fy=1, cx=0.0, cy=0.0, width=640, height=480),
            ),
        ],
        safety=SafetyStatus(
            motion_enabled=True,
            command_blocked=False,
            block_reason=None,
            cycle_id_echo=sim_step_index,
        ),
    )
    return StepResult(sensors=bundle, dt=dt, sim_step_index=sim_step_index, is_capture_tick=True)


def test_probe_post_contact_takes_priority_over_duration() -> None:
    fsm = _Fsm(
        _StubSequencer(
            [
                Probe(
                    target_pose_scene=_identity_pose(),
                    duration_s=0.1,
                    hold_after_contact_s=10.0,
                    jaw_target_start=None,
                    jaw_target_end=None,
                )
            ]
        )
    )
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0))
    fsm.reset()

    active = fsm.step(_step(sim_step_index=0, dt=0.0), last_jaw=0.0)
    assert isinstance(active.primitive, Probe)

    # Contact occurs and flips phase; duration fallback must not finish this tick.
    active = fsm.step(_step(sim_step_index=1, dt=0.2, in_contact=True), last_jaw=0.0)
    assert active.in_post_contact_phase is True
    assert fsm.completed == ()

    # Duration would have elapsed already, but we remain in post-contact hold phase.
    active = fsm.step(_step(sim_step_index=2, dt=0.2, in_contact=True), last_jaw=0.0)
    assert isinstance(active, _ActivePrimitive)
    assert fsm.completed == ()


def test_approach_ends_on_rising_contact() -> None:
    fsm = _Fsm(_StubSequencer([Approach(target_pose_scene=_identity_pose(), duration_s=10.0, end_on_contact=True, jaw_target_start=None, jaw_target_end=None)]))
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0))
    fsm.reset()
    _ = fsm.step(_step(sim_step_index=0, dt=0.0), last_jaw=0.0)

    active = fsm.step(_step(sim_step_index=1, dt=0.05, in_contact=True), last_jaw=0.0)
    assert isinstance(active.primitive, Dwell)
    assert len(fsm.completed) == 1


def test_continuous_contact_not_false_positive_across_primitive_boundary() -> None:
    fsm = _Fsm(
        _StubSequencer(
            [
                Approach(target_pose_scene=_identity_pose(), duration_s=0.05, end_on_contact=True, jaw_target_start=None, jaw_target_end=None),
                Approach(target_pose_scene=_identity_pose(), duration_s=10.0, end_on_contact=True, jaw_target_start=None, jaw_target_end=None),
            ]
        )
    )
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0, in_contact=True))
    fsm.reset()

    fsm.step(_step(sim_step_index=0, dt=0.0, in_contact=True), last_jaw=0.0)
    fsm.step(_step(sim_step_index=1, dt=0.1, in_contact=True), last_jaw=0.0)
    active = fsm.step(_step(sim_step_index=2, dt=0.1, in_contact=True), last_jaw=0.0)
    assert isinstance(active.primitive, Approach)
    assert len(fsm.completed) == 1

    active = fsm.step(_step(sim_step_index=3, dt=0.1, in_contact=True), last_jaw=0.0)
    assert len(fsm.completed) == 1
    assert not isinstance(active.primitive, Dwell)


def test_false_contact_rising_edge_isolated_to_transition() -> None:
    fsm = _Fsm(
        _StubSequencer(
            [
                Approach(target_pose_scene=_identity_pose(), duration_s=0.05, end_on_contact=True, jaw_target_start=None, jaw_target_end=None),
                Approach(target_pose_scene=_identity_pose(), duration_s=10.0, end_on_contact=True, jaw_target_start=None, jaw_target_end=None),
            ]
        )
    )
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0, in_contact=False))
    fsm.reset()

    fsm.step(_step(sim_step_index=0, dt=0.0, in_contact=False), last_jaw=0.0)
    active = fsm.step(_step(sim_step_index=1, dt=0.1, in_contact=True), last_jaw=0.0)
    assert len(fsm.completed) == 1
    assert isinstance(active.primitive, Approach)

    active = fsm.step(_step(sim_step_index=2, dt=0.1, in_contact=True), last_jaw=0.0)
    assert len(fsm.completed) == 1
    assert isinstance(active.primitive, Approach)


def test_probe_falls_back_to_duration_without_contact() -> None:
    fsm = _Fsm(
        _StubSequencer(
            [
                Probe(
                    target_pose_scene=_identity_pose(),
                    duration_s=0.1,
                    hold_after_contact_s=0.0,
                    jaw_target_start=None,
                    jaw_target_end=None,
                )
            ]
        )
    )
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0))
    fsm.reset()
    _ = fsm.step(_step(sim_step_index=0, dt=0.0), last_jaw=0.0)

    active = fsm.step(_step(sim_step_index=1, dt=0.2, in_contact=False), last_jaw=0.0)
    assert isinstance(active.primitive, Probe)
    assert len(fsm.completed) == 0

    active = fsm.step(_step(sim_step_index=2, dt=0.0), last_jaw=0.0)
    assert isinstance(active.primitive, Dwell)
    assert len(fsm.completed) == 1


def test_probe_post_contact_uses_configured_retract_speed() -> None:
    fsm = _Fsm(
        _StubSequencer(
            [
                Probe(
                    target_pose_scene=_identity_pose(),
                    duration_s=1.0,
                    hold_after_contact_s=0.0,
                    retract_distance_m=0.004,
                    retract_peak_speed_m_per_s=0.4,
                    jaw_target_start=None,
                    jaw_target_end=None,
                )
            ]
        )
    )
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0))
    fsm.reset()

    _ = fsm.step(_step(sim_step_index=0, dt=0.0), last_jaw=0.0)
    _ = fsm.step(_step(sim_step_index=1, dt=0.02, in_contact=True), last_jaw=0.0)

    # At configured speed, retract should complete at this modest post-contact delta.
    active = fsm.step(_step(sim_step_index=2, dt=0.0, in_contact=True), last_jaw=0.0)
    assert isinstance(active.primitive, Dwell)
    assert len(fsm.completed) == 1
