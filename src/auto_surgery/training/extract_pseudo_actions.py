"""Pseudo-action extraction from LoggedFrame datasets."""

from __future__ import annotations

import math

from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import SessionWriter
from auto_surgery.models.idm import IDMConfig, build_idm_mlp
from auto_surgery.schemas.commands import ControlFrame, ControlMode, Pose, Quaternion, RobotCommand, Twist, Vec3
from auto_surgery.schemas.logging import LoggedFrame
from auto_surgery.schemas.manifests import DatasetManifest
from auto_surgery.training.checkpoints import load_torch_checkpoint
from auto_surgery.training.datasets import iter_logged_frames
from auto_surgery.training.idm_train import vectorize_action_features


def _normalized_quaternion(*, w: float, x: float, y: float, z: float) -> Quaternion:
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    inv = 1.0 / norm
    return Quaternion(w=w * inv, x=x * inv, y=y * inv, z=z * inv)


def _vectorize_sensor_obs(command: RobotCommand, *, twist_keys: list[str]) -> list[float]:
    return vectorize_action_features(command, feature_keys=twist_keys)


def _vectorize_action_from_vector(
    action_vec: list[float], *, cycle_id: int, twist_keys: list[str]
) -> RobotCommand:
    if len(action_vec) != len(twist_keys):
        raise ValueError("Action vector length does not match twist_keys.")
    key_by_index = dict(zip(twist_keys, action_vec))
    if twist_keys and twist_keys[0].startswith("cartesian_pose_target"):
        return RobotCommand(
            timestamp_ns=0,
            cycle_id=cycle_id,
            control_mode=ControlMode.CARTESIAN_POSE,
            frame=ControlFrame.SCENE,
            cartesian_pose_target=Pose(
                position=Vec3(
                    x=float(key_by_index["cartesian_pose_target.position.x"]),
                    y=float(key_by_index["cartesian_pose_target.position.y"]),
                    z=float(key_by_index["cartesian_pose_target.position.z"]),
                ),
                rotation=_normalized_quaternion(
                    w=float(key_by_index["cartesian_pose_target.rotation.w"]),
                    x=float(key_by_index["cartesian_pose_target.rotation.x"]),
                    y=float(key_by_index["cartesian_pose_target.rotation.y"]),
                    z=float(key_by_index["cartesian_pose_target.rotation.z"]),
                ),
            ),
            enable=True,
            source="idm",
        )
    linear = Vec3(
        x=key_by_index.get("cartesian_twist.linear.x", 0.0),
        y=key_by_index.get("cartesian_twist.linear.y", 0.0),
        z=key_by_index.get("cartesian_twist.linear.z", 0.0),
    )
    angular = Vec3(
        x=key_by_index.get("cartesian_twist.angular.x", 0.0),
        y=key_by_index.get("cartesian_twist.angular.y", 0.0),
        z=key_by_index.get("cartesian_twist.angular.z", 0.0),
    )
    twist = Twist(linear=linear, angular=angular)
    return RobotCommand(
        timestamp_ns=0,
        cycle_id=cycle_id,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=twist,
        enable=True,
        source="idm",
    )


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
    if "twist_keys" not in vectorizer:
        raise RuntimeError(
            "IDM checkpoint is missing twist_keys vectorizer metadata; "
            "expected `vectorizer.twist_keys` for CARTESIAN_TWIST models."
        )
    twist_keys = list(vectorizer["twist_keys"])
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
        action_for_obs = frame.executed_action or frame.commanded_action
        if action_for_obs is None:
            raise ValueError(
                f"Frame {frame.frame_index} has no commanded/executed action for stage-0 extraction."
            )
        obs_vec = _vectorize_sensor_obs(action_for_obs, twist_keys=twist_keys)
        x = torch.tensor([obs_vec], dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred_vec = model(x)[0].detach().cpu().tolist()
        predicted = _vectorize_action_from_vector(
            pred_vec, cycle_id=frame.frame_index, twist_keys=twist_keys
        )
        predicted.timestamp_ns = frame.timestamp_ns

        lf = LoggedFrame(
            frame_index=frame.frame_index,
            timestamp_ns=frame.timestamp_ns,
            sensor_payload=frame.sensor_payload,
            scene_snapshot=frame.scene_snapshot,
            commanded_action=frame.commanded_action,
            executed_action=predicted,
            safety_decision=frame.safety_decision,
            entity_state=frame.entity_state,
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
    raise SystemExit(
        "Direct execution of this module is deprecated. "
        "Use: uv run auto-surgery extract-pseudo-actions."
    )
