from __future__ import annotations

from pathlib import Path

import pytest

from auto_surgery.env.sim import StubSimEnvironment
from auto_surgery.logging.case_log import CaseCatalog
from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import SessionWriter
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.logging import LoggedFrame
from auto_surgery.schemas.manifests import DataClassification, DatasetManifest, EnvConfig
from auto_surgery.training.datasets import frame_count_estimate, iter_logged_frames
from auto_surgery.training.extract_pseudo_actions import extract_pseudo_actions
from auto_surgery.training.idm_train import train_idm
from auto_surgery.training.sofa_smoke import run_sofa_smoke_pipeline


def _ensure_torch_available() -> None:
    pytest.importorskip("torch", reason="torch extra is required for Stage-0 IDM training")


def test_idm_stage0_stub_pipeline(tmp_path: Path) -> None:
    _ensure_torch_available()

    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    derived_root_uri = (tmp_path / "derived").as_uri().rstrip("/") + "/"

    sim = StubSimEnvironment()
    sim.reset(EnvConfig(seed=3, domain_randomization={"lighting": "bright"}))

    writer = SessionWriter(
        root_uri,
        "case_id_stage0",
        "session_id_stage0",
        capture_rig_id="rig",
        clock_source="monotonic",
        software_git_sha="abc123",
        data_classification=DataClassification.SIMULATION,
        sensor_list=["kinematics"],
        segment_max_frames=8,
    )

    n_frames = 64
    for i in range(n_frames):
        cmd = RobotCommand(timestamp_ns=1_000_000 + i, joint_positions={"j0": float(i) * 0.01})
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
    CaseCatalog(root_uri).append(
        manifest,
        manifest_relative_path=session_manifest_path("case_id_stage0", "session_id_stage0"),
    )
    manifest_uri = f"{root_uri}{session_manifest_path('case_id_stage0', 'session_id_stage0')}"

    ds = DatasetManifest(
        dataset_id="ds_stage0",
        session_manifest_paths=[manifest_uri],
        data_classification=DataClassification.SIMULATION,
    )
    assert frame_count_estimate(ds) == n_frames

    ckpt_uri = (tmp_path / "idm_stage0.pt").as_uri()
    train_idm(
        ds,
        out_ckpt_uri=ckpt_uri,
        dataset_manifest_path=manifest_uri,
        steps=300,
        lr=5e-3,
        hidden_dim=32,
        device=None,
        git_sha="stage0-test",
    )

    out_ds = extract_pseudo_actions(
        ds,
        idm_ckpt_uri=ckpt_uri,
        out_root_uri=derived_root_uri,
        out_case_id="case_id_stage0_derived",
        out_session_id="session_id_stage0_derived",
        capture_rig_id="rig",
        clock_source="monotonic",
        software_git_sha="stage0-test",
    )

    orig_frames = list(iter_logged_frames(ds))
    pred_frames = list(iter_logged_frames(out_ds))
    assert len(pred_frames) == len(orig_frames) == n_frames

    mse_sum = 0.0
    n = 0
    for orig, pred in zip(orig_frames, pred_frames, strict=True):
        assert orig.frame_index == pred.frame_index
        assert orig.executed_action is not None
        assert pred.executed_action is not None
        assert orig.executed_action.joint_positions is not None
        assert pred.executed_action.joint_positions is not None
        diff = (
            orig.executed_action.joint_positions["j0"] - pred.executed_action.joint_positions["j0"]
        )
        mse_sum += diff * diff
        n += 1

    mse = mse_sum / max(n, 1)
    assert mse < 1e-4


def test_sofa_smoke_pipeline(tmp_path: Path) -> None:
    _ensure_torch_available()

    root_uri = (tmp_path / "sofa_smoke").as_uri().rstrip("/") + "/"
    source, stats, derived = run_sofa_smoke_pipeline(
        out_root_uri=root_uri,
        case_id="sofa_case_stage0",
        session_id="sofa_session_stage0",
        derived_case_id="sofa_case_stage0_derived",
        derived_session_id="sofa_session_stage0_derived",
        fallback_to_stub=True,
        steps=24,
        train_steps=64,
        train_lr=5e-3,
        hidden_dim=16,
    )

    assert "train_mse" in stats
    assert frame_count_estimate(source) == 24
    src_frames = list(iter_logged_frames(source))
    pred_frames = list(iter_logged_frames(derived))
    assert len(src_frames) == len(pred_frames) == 24
