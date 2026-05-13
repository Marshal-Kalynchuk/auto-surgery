from __future__ import annotations

from typing import Any

from auto_surgery.motion.fsm import _ActivePrimitive, _Fsm
from auto_surgery.motion.primitives import Hold, Reach
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


def test_reach_finishes_on_rising_contact() -> None:
    fsm = _Fsm(
        _StubSequencer(
            [
                Reach(
                    target_pose_scene=_identity_pose(),
                    duration_s=10.0,
                    end_on_contact=True,
                    jaw_target_start=None,
                    jaw_target_end=None,
                )
            ]
        )
    )
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0))
    fsm.reset()

    active = fsm.step(_step(sim_step_index=0, dt=0.0), last_jaw=0.0)
    assert isinstance(active.primitive, Reach)

    active = fsm.step(_step(sim_step_index=1, dt=0.05, in_contact=True), last_jaw=0.0)
    assert isinstance(active.primitive, Hold)
    assert len(fsm.completed) == 1
    assert fsm.completed[0][3] is True

    # No second completion for this single-primitive sequence.
    active = fsm.step(_step(sim_step_index=2, dt=0.0), last_jaw=0.0)
    assert isinstance(active, _ActivePrimitive)
    assert len(fsm.completed) == 1


def test_reach_finishes_by_duration_with_continuous_contact() -> None:
    fsm = _Fsm(
        _StubSequencer(
            [
                Reach(target_pose_scene=_identity_pose(), duration_s=0.05, end_on_contact=True, jaw_target_start=None, jaw_target_end=None),
                Reach(target_pose_scene=_identity_pose(), duration_s=10.0, end_on_contact=True, jaw_target_start=None, jaw_target_end=None),
            ]
        )
    )
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0, in_contact=True))
    fsm.reset()

    fsm.step(_step(sim_step_index=0, dt=0.0, in_contact=True), last_jaw=0.0)
    fsm.step(_step(sim_step_index=1, dt=0.05, in_contact=True), last_jaw=0.0)
    active = fsm.step(_step(sim_step_index=2, dt=0.05, in_contact=True), last_jaw=0.0)
    assert len(fsm.completed) == 1
    assert isinstance(active.primitive, Reach)
    assert fsm.completed[0][3] is False
    assert not isinstance(active.primitive, Hold)


def test_rising_contact_only_finishes_when_not_already_in_contact() -> None:
    fsm = _Fsm(
        _StubSequencer(
            [
                Reach(target_pose_scene=_identity_pose(), duration_s=0.05, end_on_contact=True, jaw_target_start=None, jaw_target_end=None),
                Reach(target_pose_scene=_identity_pose(), duration_s=10.0, end_on_contact=True, jaw_target_start=None, jaw_target_end=None),
            ]
        )
    )
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0, in_contact=False))
    fsm.reset()

    active = fsm.step(_step(sim_step_index=0, dt=0.0, in_contact=False), last_jaw=0.0)
    assert isinstance(active.primitive, Reach)

    active = fsm.step(_step(sim_step_index=1, dt=0.0, in_contact=True), last_jaw=0.0)
    assert len(fsm.completed) == 1
    assert isinstance(active.primitive, Reach)
    assert fsm.completed[0][3] is True

    active = fsm.step(_step(sim_step_index=2, dt=0.1, in_contact=True), last_jaw=0.0)
    assert len(fsm.completed) == 1
    assert isinstance(active.primitive, Reach)


def test_hold_finishes_by_duration_only() -> None:
    fsm = _Fsm(_StubSequencer([Hold(duration_s=0.2, jaw_target_start=None, jaw_target_end=None)]))
    fsm._sequencer.reset(_step(sim_step_index=0, dt=0.0, in_contact=True))
    fsm.reset()

    _ = fsm.step(_step(sim_step_index=0, dt=0.0, in_contact=True), last_jaw=0.0)
    active = fsm.step(_step(sim_step_index=1, dt=0.2, in_contact=True), last_jaw=0.0)
    assert len(fsm.completed) == 0
    assert isinstance(active.primitive, Hold)

    active = fsm.step(_step(sim_step_index=2, dt=0.0, in_contact=False), last_jaw=0.0)
    assert isinstance(active.primitive, Hold)
    assert len(fsm.completed) == 1
    assert fsm.completed[0][3] is False
