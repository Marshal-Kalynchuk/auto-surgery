from __future__ import annotations

from pathlib import Path

from auto_surgery.env.sim import StubSimEnvironment
from auto_surgery.logging.case_log import CaseCatalog
from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import SessionWriter
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.logging import LoggedFrame
from auto_surgery.schemas.manifests import (
    CheckpointManifest,
    DataClassification,
    DatasetManifest,
    EnvConfig,
)
from auto_surgery.training.checkpoints import load_torch_checkpoint, save_torch_checkpoint_atomic
from auto_surgery.training.datasets import frame_count_estimate, iter_logged_frames


def test_sim_rollout_to_dataset_loader_and_checkpoint(tmp_path: Path) -> None:
    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    sim = StubSimEnvironment()
    sim.reset(EnvConfig(seed=3, domain_randomization={"lighting": "bright"}))
    writer = SessionWriter(
        root_uri,
        "case_x",
        "sess_y",
        capture_rig_id="rig",
        clock_source="monotonic",
        software_git_sha="abc123",
        data_classification=DataClassification.SIMULATION,
        sensor_list=["kinematics"],
        segment_max_frames=4,
    )
    for i in range(10):
        cmd = RobotCommand(timestamp_ns=1_000 + i, joint_positions={"j0": float(i) * 0.01})
        gate = sim.gate_command(cmd)
        step = sim.step(cmd)
        lf = LoggedFrame(
            frame_index=i,
            timestamp_ns=cmd.timestamp_ns,
            sensor_payload=step.sensor_observation,
            scene_snapshot=step.next_scene,
            commanded_action=cmd,
            executed_action=cmd,
            safety_decision=gate,
        )
        writer.write_frame(lf)
    manifest = writer.finalize()
    catalog = CaseCatalog(root_uri)
    catalog.append(manifest, manifest_relative_path=session_manifest_path("case_x", "sess_y"))

    manifest_uri = f"{root_uri}{session_manifest_path('case_x', 'sess_y')}"
    ds = DatasetManifest(
        dataset_id="ds1",
        session_manifest_paths=[manifest_uri],
        data_classification=DataClassification.SIMULATION,
    )
    assert frame_count_estimate(ds) == 10
    frames = list(iter_logged_frames(ds))
    assert len(frames) == 10
    assert frames[0].commanded_action is not None

    ckpt_uri = (tmp_path / "checkpoint.pt").as_uri()
    save_torch_checkpoint_atomic(
        ckpt_uri,
        {"metrics": {"loss": 0.01}},
        CheckpointManifest(
            checkpoint_id="ckpt1",
            dataset_manifest_path=manifest_uri,
            training_config_path="n/a",
            git_sha="abc123",
        ),
    )
    loaded = load_torch_checkpoint(ckpt_uri)
    assert loaded["metrics"]["loss"] == 0.01
