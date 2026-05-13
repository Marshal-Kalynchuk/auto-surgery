from __future__ import annotations

import numpy as np

from auto_surgery.env.scene_geometry import SceneGeometry, SurfacePoint
from auto_surgery.motion.sequencer import MotionGeneratorConfig, Sequencer
from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3
from auto_surgery.schemas.scene import SceneConfig, SphereEnvelope, TargetVolume
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import CameraIntrinsics, CameraView, SafetyStatus, SensorBundle, ToolState


class _MockSceneGeometry(SceneGeometry):
    def closest_surface_point(self, p_scene: Vec3) -> SurfacePoint:
        return SurfacePoint(position=p_scene, normal=Vec3(x=0.0, y=0.0, z=1.0), signed_distance=0.0)

    def signed_distance(self, p_scene: Vec3) -> float:
        del p_scene
        return 0.0

    def ray_cast(self, origin: Vec3, direction: Vec3, max_distance_m: float) -> None:
        del origin
        del direction
        del max_distance_m
        return None

    def bounds(self) -> tuple[Vec3, Vec3]:
        return Vec3(x=-1.0, y=-1.0, z=-1.0), Vec3(x=1.0, y=1.0, z=1.0)


def _identity_pose() -> Pose:
    return Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _step() -> StepResult:
    return StepResult(
        sensors=SensorBundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            tool=ToolState(
                pose=_identity_pose(),
                twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
                jaw=0.0,
                wrench=Vec3(x=0.0, y=0.0, z=0.0),
                in_contact=False,
            ),
            cameras=[
                CameraView(
                    camera_id="cam",
                    timestamp_ns=0,
                    extrinsics=_identity_pose(),
                    intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=640, height=480),
                )
            ],
            safety=SafetyStatus(motion_enabled=True, command_blocked=False, block_reason=None, cycle_id_echo=0),
        ),
        dt=0.0,
        sim_step_index=0,
        is_capture_tick=True,
    )


def _sequencer_config(seed: int = 42) -> MotionGeneratorConfig:
    return MotionGeneratorConfig(
        seed=seed,
        primitive_count_min=1,
        primitive_count_max=1,
        approach_duration_range_s=(0.2, 0.2),
        dwell_duration_range_s=(0.2, 0.2),
        retract_duration_range_s=(0.2, 0.2),
        retract_distance_range_m=(0.0, 0.0),
        sweep_duration_range_s=(0.2, 0.2),
        sweep_arc_range_rad=(0.2, 0.2),
        rotate_duration_range_s=(0.2, 0.2),
        rotate_angle_range_rad=(0.2, 0.2),
        probe_duration_range_s=(0.2, 0.2),
        probe_hold_range_s=(0.1, 0.1),
        target_orientation_jitter_rad=0.1,
        jaw_value_range=(0.0, 1.0),
        jaw_change_probability=0.0,
        weight_approach=1.0,
        weight_dwell=1.0,
        weight_retract=1.0,
        weight_sweep=1.0,
        weight_rotate=1.0,
        weight_probe=1.0,
    )


def _scene() -> SceneConfig:
    return SceneConfig(
        tissue_scene_path="",
        target_volumes=[
            TargetVolume(
                label="general",
                center_scene=Vec3(x=0.0, y=0.0, z=0.0),
                half_extents_scene=Vec3(x=0.15, y=0.15, z=0.15),
            )
        ],
    )


def _inner_envelope() -> SphereEnvelope:
    return SphereEnvelope(
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        radius_m=1.0,
        outer_margin_m=1.0,
        inner_margin_m=0.2,
    )


def test_build_reach_samples_within_inner_envelope() -> None:
    cfg = _sequencer_config(seed=7)
    scene = _scene()
    step = _step()
    envelope = _inner_envelope()
    seq = Sequencer(cfg, scene, _MockSceneGeometry(), envelope)
    seq.reset(step)
    primitive = seq._build_reach(step)

    target = primitive.target_pose_scene.position
    signed_distance = envelope.signed_distance_to_envelope(target)
    assert float(signed_distance) >= float(envelope.inner_margin_m)


def test_named_sub_rngs_are_seed_deterministic() -> None:
    cfg = _sequencer_config(seed=88)
    scene = _scene()
    step = _step()
    envelope = _inner_envelope()

    left = Sequencer(cfg, scene, _MockSceneGeometry(), envelope)
    right = Sequencer(cfg, scene, _MockSceneGeometry(), envelope)
    left.reset(step)
    right.reset(step)

    left_draws = (
        left._rng_targets.uniform(size=4),
        left._rng_noise.uniform(size=4),
        left._rng_blend.uniform(size=4),
    )
    right_draws = (
        right._rng_targets.uniform(size=4),
        right._rng_noise.uniform(size=4),
        right._rng_blend.uniform(size=4),
    )

    for lhs, rhs in zip(left_draws, right_draws):
        assert np.array_equal(lhs, rhs)

