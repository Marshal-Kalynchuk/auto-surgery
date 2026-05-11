"""Pseudo-action extraction from LoggedFrame datasets."""

from __future__ import annotations

from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import SessionWriter
from auto_surgery.models.idm import IDMConfig, build_idm_mlp
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.logging import LoggedFrame
from auto_surgery.schemas.manifests import DatasetManifest
from auto_surgery.schemas.sensors import SensorBundle
from auto_surgery.training.checkpoints import load_torch_checkpoint
from auto_surgery.training.datasets import iter_logged_frames


def _vectorize_sensor_obs(sensors: SensorBundle, *, joint_keys: list[str]) -> list[float]:
    modalities = sensors.modalities
    cmd_echo = modalities.get("command_echo")
    if not isinstance(cmd_echo, dict):
        raise ValueError("Stage-0 extract expects sensors.modalities['command_echo'] to be a dict.")
    joint_positions = cmd_echo.get("joint_positions")
    if not isinstance(joint_positions, dict):
        raise ValueError("Stage-0 extract expects command_echo['joint_positions'] to be a dict.")
    missing = [k for k in joint_keys if k not in joint_positions]
    if missing:
        raise ValueError(f"command_echo joint_positions missing keys: {missing}")
    return [float(joint_positions[k]) for k in joint_keys]


def _vectorize_action_from_vector(
    action_vec: list[float], *, joint_keys: list[str]
) -> RobotCommand:
    if len(action_vec) != len(joint_keys):
        raise ValueError("Action vector length does not match joint_keys.")
    joint_positions = {k: float(action_vec[i]) for i, k in enumerate(joint_keys)}
    return RobotCommand(timestamp_ns=0, joint_positions=joint_positions, representation="joint")


def extract_pseudo_actions(
    manifest: DatasetManifest,
    *,
    idm_ckpt_uri: str,
    out_root_uri: str,
    out_case_id: str,
    out_session_id: str,
    capture_rig_id: str = "rig",
    clock_source: str = "monotonic",
    software_git_sha: str = "stage0",
    device: str | None = None,
) -> DatasetManifest:
    """Run IDM over frames and write derived session outputs."""
    try:
        import torch
    except ImportError as e:
        raise ImportError("Extracting pseudo actions requires torch (train extra).") from e

    ckpt = load_torch_checkpoint(idm_ckpt_uri)
    vectorizer = ckpt["vectorizer"]
    joint_keys = list(vectorizer["joint_keys"])
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    model_state = ckpt["model_state"]

    cfg = IDMConfig(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    model = build_idm_mlp(cfg)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.load_state_dict(model_state)
    model.eval()

    writer = SessionWriter(
        out_root_uri,
        out_case_id,
        out_session_id,
        capture_rig_id=capture_rig_id,
        clock_source=clock_source,
        software_git_sha=software_git_sha,
        data_classification=manifest.data_classification,
        retention_tier=manifest.retention_tier,
        sensor_list=[],
        segment_max_frames=256,
    )

    for frame in iter_logged_frames(manifest, phi_training_allowed=False):
        obs_vec = _vectorize_sensor_obs(frame.sensor_payload, joint_keys=joint_keys)
        x = torch.tensor([obs_vec], dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred_vec = model(x)[0].detach().cpu().tolist()
        predicted = _vectorize_action_from_vector(pred_vec, joint_keys=joint_keys)
        predicted.timestamp_ns = frame.timestamp_ns

        lf = LoggedFrame(
            frame_index=frame.frame_index,
            timestamp_ns=frame.timestamp_ns,
            sensor_payload=frame.sensor_payload,
            scene_snapshot=frame.scene_snapshot,
            commanded_action=frame.commanded_action,
            executed_action=predicted,
            safety_decision=frame.safety_decision,
            skill_state=frame.skill_state,
            surgeon_input=frame.surgeon_input,
            outcome_label=frame.outcome_label,
        )
        writer.write_frame(lf)

    writer.finalize()

    out_ds = DatasetManifest(
        dataset_id=f"{manifest.dataset_id}_pseudo",
        session_manifest_paths=[
            f"{out_root_uri}{session_manifest_path(out_case_id, out_session_id)}",
        ],
        frame_filters={},
        data_classification=manifest.data_classification,
        retention_tier=manifest.retention_tier,
    )
    return out_ds


def main() -> None:
    raise SystemExit(
        "This module is intended to be called from the Stage-0 test harness. "
        "Wire a CLI only after the Stage-0 loop is stable."
    )


if __name__ == "__main__":
    main()

