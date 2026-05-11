"""SOFA rollout and IDM smoke helpers for Stage-0 execution tests."""

from __future__ import annotations

import math
from typing import Any

from auto_surgery.env.sofa import SofaEnvironment, SofaRuntimeBackendFactory
from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import SessionWriter
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.logging import LoggedFrame
from auto_surgery.schemas.manifests import (
    DataClassification,
    DatasetManifest,
    EnvConfig,
    RetentionTier,
)
from auto_surgery.training.datasets import frame_count_estimate
from auto_surgery.training.extract_pseudo_actions import extract_pseudo_actions
from auto_surgery.training.idm_train import train_idm


def build_lite_command(
    step_index: int, *, base_ns: int = 1_000_000, amplitude: float = 0.05
) -> RobotCommand:
    """Generate a deterministic 1-joint command for smoke trajectories."""

    t = step_index * 0.1
    joint = amplitude * math.sin(t)
    return RobotCommand(
        timestamp_ns=base_ns + step_index,
        joint_positions={"j0": float(joint)},
        representation="joint",
    )


def run_sofa_rollout_dataset(
    *,
    storage_root_uri: str,
    case_id: str,
    session_id: str,
    sofa_scene_path: str | None = None,
    fallback_to_stub: bool = True,
    steps: int = 64,
    segment_max_frames: int = 128,
    seed: int = 7,
    domain_randomization: dict[str, Any] | None = None,
    sofa_backend_factory: SofaRuntimeBackendFactory | None = None,
    capture_rig_id: str = "sofa_rig",
) -> DatasetManifest:
    """Run a bounded rollout and persist frames into a manifest-consumable dataset."""

    env = SofaEnvironment(
        sofa_scene_path=sofa_scene_path,
        fallback_to_stub=fallback_to_stub,
        sofa_backend_factory=sofa_backend_factory,
    )
    env.reset(EnvConfig(seed=seed, domain_randomization=domain_randomization or {}))

    root = storage_root_uri.rstrip("/") + "/"
    writer = SessionWriter(
        root,
        case_id,
        session_id,
        capture_rig_id=capture_rig_id,
        clock_source="monotonic",
        software_git_sha="stage0",
        data_classification=DataClassification.SIMULATION,
        retention_tier=RetentionTier.CURATED_TRAINING,
        sensor_list=["command_echo"],
        segment_max_frames=segment_max_frames,
    )

    for i in range(steps):
        cmd = build_lite_command(i)
        step = env.step(cmd)
        lf = LoggedFrame(
            frame_index=i,
            timestamp_ns=cmd.timestamp_ns,
            sensor_payload=step.sensor_observation,
            scene_snapshot=step.next_scene,
            commanded_action=cmd,
            executed_action=cmd,
            safety_decision=None,
            skill_state=None,
            surgeon_input=None,
            outcome_label=None,
        )
        writer.write_frame(lf)

    writer.finalize()
    session_manifest_uri = f"{root}{session_manifest_path(case_id, session_id)}"
    return DatasetManifest(
        dataset_id=f"sofa_stage0_{case_id}_{session_id}",
        session_manifest_paths=[session_manifest_uri],
        frame_filters={"start_step": 0, "num_steps": steps},
        data_classification=DataClassification.SIMULATION,
        retention_tier=RetentionTier.CURATED_TRAINING,
    )


def run_sofa_smoke_pipeline(
    *,
    out_root_uri: str,
    case_id: str,
    session_id: str,
    derived_case_id: str,
    derived_session_id: str,
    fallback_to_stub: bool = True,
    steps: int = 64,
    train_steps: int = 64,
    train_lr: float = 5e-3,
    hidden_dim: int = 32,
) -> tuple[DatasetManifest, dict[str, float], DatasetManifest]:
    """Execute end-to-end: rollout -> train_idm -> extract pseudo actions."""

    base_root = out_root_uri.rstrip("/") + "/"
    src_ds = run_sofa_rollout_dataset(
        storage_root_uri=base_root,
        case_id=case_id,
        session_id=session_id,
        fallback_to_stub=fallback_to_stub,
        steps=steps,
    )
    assert frame_count_estimate(src_ds) == steps

    ckpt_uri = f"{base_root}sofa_idm_stage0.pt"
    train_stats = train_idm(
        src_ds,
        out_ckpt_uri=ckpt_uri,
        checkpoint_id="sofa_stage0_ckpt",
        dataset_manifest_path=f"{base_root}{session_manifest_path(case_id, session_id)}",
        steps=train_steps,
        lr=train_lr,
        hidden_dim=hidden_dim,
        device=None,
        git_sha="stage0-smoke",
    )

    pseudo_ds = extract_pseudo_actions(
        src_ds,
        idm_ckpt_uri=ckpt_uri,
        out_root_uri=base_root,
        out_case_id=derived_case_id,
        out_session_id=derived_session_id,
        capture_rig_id="sofa_rig",
        clock_source="monotonic",
        software_git_sha="stage0-smoke",
        device=None,
    )

    return src_ds, train_stats, pseudo_ds
