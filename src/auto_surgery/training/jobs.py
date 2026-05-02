"""Job helpers — MLflow optional, Hydra optional."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_bootstrap_m1_job(
    *,
    mlflow_tracking_uri: str | None = None,
    experiment_name: str = "phase0_bootstrap",
) -> float:
    """Run M1 with optional MLflow logging."""

    from auto_surgery.training.bootstrap import run_m1_tiny_overfit

    loss = run_m1_tiny_overfit()
    if mlflow_tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="m1_tiny_overfit"):
            mlflow.log_metric("final_loss", loss)
    return loss


def write_resolved_config_snapshot(output_dir: str | Path, payload: dict[str, Any]) -> None:
    """Persist resolved training config next to artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
