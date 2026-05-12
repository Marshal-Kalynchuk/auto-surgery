from __future__ import annotations

from pathlib import Path

from auto_surgery.logging.case_log import CaseCatalog
from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import (
    SessionWriter,
    load_segment_frames,
    load_session_manifest,
)
from auto_surgery.schemas.commands import ControlMode, Pose, Quaternion, RobotCommand, Twist, Vec3
from auto_surgery.schemas.logging import LoggedFrame, SafetyDecision
from auto_surgery.schemas.manifests import DataClassification
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import (
    CameraIntrinsics,
    CameraView,
    SafetyStatus,
    SensorBundle,
    ToolState,
)


def test_session_writer_append_only_segments(tmp_path: Path) -> None:
    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    writer = SessionWriter(
        root_uri,
        "c1",
        "s1",
        capture_rig_id="rig-a",
        clock_source="ntp",
        software_git_sha="deadbeef",
        data_classification=DataClassification.SYNTHETIC,
        sensor_list=["kinematics"],
        segment_max_frames=2,
    )
    for i in range(5):
        lf = LoggedFrame(
            frame_index=i,
            timestamp_ns=i,
            sensor_payload=SensorBundle(
                timestamp_ns=i,
                sim_time_s=0.0,
                tool=ToolState(
                    pose=Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)),
                    twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
                    jaw=0.0,
                    wrench=Vec3(x=0.0, y=0.0, z=0.0),
                    in_contact=False,
                ),
                cameras=[
                    CameraView(
                        camera_id="test_cam",
                        timestamp_ns=i,
                        extrinsics=Pose(
                            position=Vec3(x=0.0, y=0.0, z=0.0),
                            rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                        ),
                        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1),
                    )
                ],
                safety=SafetyStatus(
                    motion_enabled=True,
                    command_blocked=False,
                    block_reason=None,
                    cycle_id_echo=0,
                ),
            ),
            scene_snapshot=SceneGraph(frame_index=i),
            commanded_action=RobotCommand(
                timestamp_ns=i,
                cycle_id=i,
                control_mode=ControlMode.CARTESIAN_TWIST,
                cartesian_twist={"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}},
            ),
            safety_decision=SafetyDecision(ok=True),
        )
        writer.write_frame(lf)
    manifest = writer.finalize()
    assert manifest.partition_checksums
    assert load_session_manifest(root_uri, "c1", "s1").session_id == "s1"
    seg0 = load_segment_frames(root_uri, "c1", "s1", 0)
    assert len(seg0) == 2
    catalog = CaseCatalog(root_uri)
    catalog.append(manifest, manifest_relative_path=session_manifest_path("c1", "s1"))
