from __future__ import annotations

from pathlib import Path

from auto_surgery.config import load_motion_config
from auto_surgery.motion.generator import SurgicalMotionGenerator
from auto_surgery.schemas.commands import Pose, Quaternion, Twist, Vec3
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneConfig, TargetVolume
from auto_surgery.schemas.sensors import (
    CameraIntrinsics,
    CameraView,
    SafetyStatus,
    SensorBundle,
    ToolState,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION_CONFIG_PATH = REPO_ROOT / "configs" / "motion" / "default.yaml"


def _minimal_step() -> StepResult:
    pose = Pose(
        position=Vec3(x=0.0, y=0.0, z=0.0),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    return StepResult(
        sensors=SensorBundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            tool=ToolState(
                pose=pose,
                twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
                jaw=0.0,
                wrench=Vec3(x=0.0, y=0.0, z=0.0),
                in_contact=False,
            ),
            cameras=[
                CameraView(
                    camera_id="cam",
                    timestamp_ns=0,
                    extrinsics=pose,
                    intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=640, height=480),
                )
            ],
            safety=SafetyStatus(
                motion_enabled=True,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=0,
            ),
        ),
        dt=1.0 / 30.0,
        sim_step_index=0,
        is_capture_tick=True,
    )


def test_surgical_motion_generator_emits_motion_shaping_after_defaults() -> None:
    motion = load_motion_config(MOTION_CONFIG_PATH).with_default_motion_shaping("dejavu_brain")
    assert motion.motion_shaping is not None
    assert motion.motion_shaping_enabled is True

    scene = SceneConfig(
        tissue_scene_path=Path("/tmp/tissue_placeholder.scn"),
        target_volumes=[
            TargetVolume(
                label="general",
                center_scene=Vec3(x=0.0, y=0.0, z=0.0),
                half_extents_scene=Vec3(x=10.0, y=10.0, z=10.0),
            )
        ],
    )
    gen = SurgicalMotionGenerator(motion, scene)
    cmd = gen.reset(_minimal_step())
    assert cmd.motion_shaping is not None
    assert cmd.motion_shaping_enabled is True
