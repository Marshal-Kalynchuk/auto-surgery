from __future__ import annotations

from typing import Any


def assess_async_safety(*, command: Any, state: Any | None = None, **_: Any) -> None:
    """Evaluate asynchronous safety conditions for a command/state pair."""

    raise NotImplementedError("Async safety assessment is not implemented in Stage-0.")
