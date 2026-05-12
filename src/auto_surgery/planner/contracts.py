from __future__ import annotations

from typing import Any


def evaluate_plan_contracts(*, plan: Any, state: Any | None = None, **_: Any) -> bool:
    """Evaluate hard/soft plan contracts."""

    raise NotImplementedError("Planner contract evaluation is not implemented.")
