from __future__ import annotations

from typing import Any


def recondition_command(*, command: Any, **_: Any) -> Any:
    """Optionally remap or dampen a command before execution."""

    raise NotImplementedError("Command re-conditioning is not implemented in Stage-0.")
