from __future__ import annotations

from typing import Any


def evaluate_contracts(*, command: Any, contracts: Any | None = None, **_: Any) -> None:
    """Evaluate hard and soft contracts for a given command."""

    raise NotImplementedError("Contract monitoring is not implemented in Stage-0.")
