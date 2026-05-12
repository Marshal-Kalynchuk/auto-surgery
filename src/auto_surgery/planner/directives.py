from __future__ import annotations

from typing import Any


def translate_directives(
    *, raw_directives: Any, context: Any | None = None, **_: Any
) -> list[Any]:
    """Map planner inputs into normalized directive structures."""

    raise NotImplementedError("Planner directive parsing is not implemented.")
