"""Train an Inverse Dynamics Model (IDM) on logged environment frames.

Stage-0 vectorizes either ``cartesian_twist`` (legacy) or ``cartesian_pose_target``
(scene-frame pose commands from the motion generator). When ``rgb`` blobs exist under a
session's ``blobs/`` prefix, a future encoder can join them by ``frame_index`` before this
function is extended beyond the vectorized baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import fsspec

from auto_surgery.models.idm import IDMConfig, build_idm_mlp
from auto_surgery.schemas.commands import ControlMode, RobotCommand
from auto_surgery.schemas.manifests import CheckpointManifest, DatasetManifest
from auto_surgery.training.checkpoints import save_torch_checkpoint_atomic
from auto_surgery.training.datasets import iter_logged_frames


@dataclass(frozen=True)
class VectorizerConfig:
    """Vectorization contract stored in the IDM checkpoint."""

    twist_keys: list[str]


_TWIST_FEATURE_KEYS: list[str] = [
    "cartesian_twist.linear.x",
    "cartesian_twist.linear.y",
    "cartesian_twist.linear.z",
    "cartesian_twist.angular.x",
    "cartesian_twist.angular.y",
    "cartesian_twist.angular.z",
]

_POSE_FEATURE_KEYS: list[str] = [
    "cartesian_pose_target.position.x",
    "cartesian_pose_target.position.y",
    "cartesian_pose_target.position.z",
    "cartesian_pose_target.rotation.w",
    "cartesian_pose_target.rotation.x",
    "cartesian_pose_target.rotation.y",
    "cartesian_pose_target.rotation.z",
]


def _action_scalar_table(command: RobotCommand) -> dict[str, float]:
    out: dict[str, float] = {}
    if command.cartesian_twist is not None:
        twist = command.cartesian_twist
        out.update(
            {
                "cartesian_twist.linear.x": float(twist.linear.x),
                "cartesian_twist.linear.y": float(twist.linear.y),
                "cartesian_twist.linear.z": float(twist.linear.z),
                "cartesian_twist.angular.x": float(twist.angular.x),
                "cartesian_twist.angular.y": float(twist.angular.y),
                "cartesian_twist.angular.z": float(twist.angular.z),
            }
        )
    if command.cartesian_pose_target is not None:
        pose = command.cartesian_pose_target
        out.update(
            {
                "cartesian_pose_target.position.x": float(pose.position.x),
                "cartesian_pose_target.position.y": float(pose.position.y),
                "cartesian_pose_target.position.z": float(pose.position.z),
                "cartesian_pose_target.rotation.w": float(pose.rotation.w),
                "cartesian_pose_target.rotation.x": float(pose.rotation.x),
                "cartesian_pose_target.rotation.y": float(pose.rotation.y),
                "cartesian_pose_target.rotation.z": float(pose.rotation.z),
            }
        )
    return out


def vectorize_action_features(command: RobotCommand, *, feature_keys: list[str]) -> list[float]:
    """Flatten the requested scalar keys from ``command`` in manifest order."""

    table = _action_scalar_table(command)
    missing = [key for key in feature_keys if key not in table]
    if missing:
        raise ValueError(f"Action vectorization missing keys: {missing}")
    return [float(table[key]) for key in feature_keys]


def vectorize_command_twist(cmd: RobotCommand, *, twist_keys: list[str]) -> list[float]:
    return vectorize_action_features(cmd, feature_keys=twist_keys)


def vectorize_sensor_twist(action: RobotCommand, *, twist_keys: list[str]) -> list[float]:
    return vectorize_action_features(action, feature_keys=twist_keys)


def infer_twist_keys_from_manifest(manifest: DatasetManifest) -> list[str]:
    for frame in iter_logged_frames(manifest, phi_training_allowed=False):
        action = frame.executed_action or frame.commanded_action
        if action is None:
            continue
        if action.cartesian_twist is not None:
            return list(_TWIST_FEATURE_KEYS)
        if action.control_mode == ControlMode.CARTESIAN_POSE and action.cartesian_pose_target is not None:
            return list(_POSE_FEATURE_KEYS)
    raise RuntimeError("Could not infer twist or pose feature keys from dataset frames.")


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
    twist_keys = infer_twist_keys_from_manifest(manifest)

    obs_vecs: list[list[float]] = []
    act_vecs: list[list[float]] = []
    for frame in iter_logged_frames(manifest, phi_training_allowed=False):
        action = frame.executed_action or frame.commanded_action
        if action is None:
            continue
        try:
            vec = vectorize_action_features(action, feature_keys=twist_keys)
        except ValueError:
            continue
        obs_vecs.append(vec)
        act_vecs.append(vec)

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
        "vectorizer": VectorizerConfig(twist_keys=twist_keys).__dict__,
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
    p.add_argument("--out-ckpt-uri", required=True, help="fsspec URI to write .pt checkpoint.")
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
    raise SystemExit(
        "Direct execution of this module is deprecated. "
        "Use: uv run auto-surgery train-idm."
    )
