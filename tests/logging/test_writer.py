from __future__ import annotations

from pathlib import Path

from auto_surgery.logging.case_log import CaseCatalog
from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import (
    SessionWriter,
    load_segment_frames,
    load_session_manifest,
)
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.logging import LoggedFrame, SafetyDecision
from auto_surgery.schemas.manifests import DataClassification
from auto_surgery.schemas.scene import SceneGraph
from auto_surgery.schemas.sensors import SensorBundle


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
            sensor_payload=SensorBundle(timestamp_ns=i, clock_source="ntp", modalities={}),
            scene_snapshot=SceneGraph(frame_index=i),
            commanded_action=RobotCommand(timestamp_ns=i),
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
