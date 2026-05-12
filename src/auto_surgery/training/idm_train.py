"""Train an Inverse Dynamics Model (IDM) on logged environment frames.

Stage-0 vectorizes `command_echo` joint positions. When `rgb` blobs exist under a
session's `blobs/` prefix, a future encoder can join them by `frame_index` before this
function is extended beyond the MLP joint-space baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fsspec

from auto_surgery.models.idm import IDMConfig, build_idm_mlp
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import CheckpointManifest, DatasetManifest
from auto_surgery.schemas.sensors import SensorBundle
from auto_surgery.training.checkpoints import save_torch_checkpoint_atomic
from auto_surgery.training.datasets import iter_logged_frames


@dataclass(frozen=True)
class VectorizerConfig:
    """Vectorization contract stored in the IDM checkpoint."""

    joint_keys: list[str]


def _extract_command_echo(sensors: SensorBundle) -> dict[str, Any] | None:
    """Stage-0 expects a sensor modality with joint command echo."""
    modalities = sensors.modalities
    cmd_echo = modalities.get("command_echo")
    if not isinstance(cmd_echo, dict):
        return None
    return cmd_echo


def vectorize_command_joint_positions(
    cmd: RobotCommand, *, joint_keys: list[str]
) -> list[float]:
    if cmd.joint_positions is None:
        raise ValueError("RobotCommand.joint_positions is required for Stage-0 vectorization.")
    missing = [k for k in joint_keys if k not in cmd.joint_positions]
    if missing:
        raise ValueError(f"RobotCommand missing joint keys: {missing}")
    return [float(cmd.joint_positions[k]) for k in joint_keys]


def vectorize_sensor_to_obs(
    sensors: SensorBundle, *, joint_keys: list[str]
) -> list[float]:
    cmd_echo = _extract_command_echo(sensors)
    if cmd_echo is None:
        raise ValueError("Stage-0 vectorizer requires sensors.modalities['command_echo'].")
    joint_positions = cmd_echo.get("joint_positions")
    if not isinstance(joint_positions, dict):
        raise ValueError("command_echo must include a joint_positions dict.")
    missing = [k for k in joint_keys if k not in joint_positions]
    if missing:
        raise ValueError(f"command_echo missing joint keys: {missing}")
    return [float(joint_positions[k]) for k in joint_keys]


def infer_joint_keys_from_manifest(manifest: DatasetManifest) -> list[str]:
    for frame in iter_logged_frames(manifest, phi_training_allowed=False):
        action = frame.executed_action or frame.commanded_action
        if action is None or action.joint_positions is None:
            continue
        return sorted(action.joint_positions.keys())
    raise RuntimeError("Could not infer joint_keys from dataset frames.")


def train_idm(
    manifest: DatasetManifest,
    *,
    out_ckpt_uri: str,
    checkpoint_id: str = "idm_stage0_ckpt",
    dataset_manifest_path: str = "n/a",
    steps: int = 300,
    lr: float = 2e-3,
    hidden_dim: int = 256,
    device: str | None = None,
    git_sha: str = "unknown",
) -> dict[str, float]:
    try:
        import torch
    except ImportError as e:
        raise ImportError("Training requires torch: uv sync --extra train") from e

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    joint_keys = infer_joint_keys_from_manifest(manifest)

    obs_vecs: list[list[float]] = []
    act_vecs: list[list[float]] = []
    for frame in iter_logged_frames(manifest, phi_training_allowed=False):
        action = frame.executed_action or frame.commanded_action
        if action is None or action.joint_positions is None:
            continue
        obs_vec = vectorize_sensor_to_obs(frame.sensor_payload, joint_keys=joint_keys)
        act_vec = vectorize_command_joint_positions(action, joint_keys=joint_keys)
        obs_vecs.append(obs_vec)
        act_vecs.append(act_vec)

    if not obs_vecs:
        raise RuntimeError("No (obs, action) pairs found in dataset for IDM training.")

    obs_dim = len(obs_vecs[0])
    act_dim = len(act_vecs[0])
    if any(len(v) != obs_dim for v in obs_vecs):
        raise ValueError("Inconsistent obs vector dimensions.")
    if any(len(v) != act_dim for v in act_vecs):
        raise ValueError("Inconsistent action vector dimensions.")

    cfg = IDMConfig(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    model = build_idm_mlp(cfg).to(dev)

    x = torch.tensor(obs_vecs, dtype=torch.float32, device=dev)
    y = torch.tensor(act_vecs, dtype=torch.float32, device=dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last_loss = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    state = {
        "model_state": model.state_dict(),
        "vectorizer": VectorizerConfig(joint_keys=joint_keys).__dict__,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dim": hidden_dim,
        "metrics": {"train_mse": float(last_loss or 0.0), "steps": float(steps)},
    }
    manifest_obj = CheckpointManifest(
        checkpoint_id=checkpoint_id,
        dataset_manifest_path=dataset_manifest_path,
        training_config_path="n/a",
        git_sha=git_sha,
        mlflow_run_id=None,
        metrics={"train_mse": float(last_loss or 0.0)},
    )
    save_torch_checkpoint_atomic(out_ckpt_uri, state, manifest_obj)
    return {"train_mse": float(last_loss or 0.0)}


def _load_dataset_manifest_from_uri(uri: str) -> DatasetManifest:
    """Load a DatasetManifest JSON file (fsspec URI supported)."""
    fs, path = fsspec.url_to_fs(uri)
    with fs.open(path, "rb") as f:
        raw = f.read().decode("utf-8")
    import json

    payload = json.loads(raw)
    return DatasetManifest.model_validate(payload)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train Stage-0 IDM on logged frames.")
    p.add_argument(
        "--dataset-manifest-uri",
        required=True,
        help="fsspec URI to DatasetManifest JSON.",
    )
    p.add_argument(
        "--out-ckpt-uri", required=True, help="fsspec URI to write .pt checkpoint."
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    manifest = _load_dataset_manifest_from_uri(args.dataset_manifest_uri)
    train_idm(
        manifest,
        out_ckpt_uri=args.out_ckpt_uri,
        steps=args.steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
        dataset_manifest_path=args.dataset_manifest_uri,
    )


if __name__ == "__main__":
    main()

