from __future__ import annotations

from typing import Any


def compute_mpc_command(*, observation: Any, plan: Any, **_: Any) -> Any:
    """Compute a model-predictive control command for the next time step."""

    raise NotImplementedError("MPC controller integration is not implemented.")
