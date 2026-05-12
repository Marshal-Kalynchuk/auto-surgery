from __future__ import annotations

from typing import Any


def enforce_sync_gate(*, command: Any, context: Any | None = None, **_: Any) -> None:
    """Validate a command against synchronous safety constraints."""

    raise NotImplementedError("Sync gate is not implemented in Stage-0.")
