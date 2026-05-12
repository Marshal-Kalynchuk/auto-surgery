from __future__ import annotations

from typing import Any


def search_plans(*, directives: Any, context: Any | None = None, **_: Any) -> list[Any]:
    """Search candidate plans."""

    raise NotImplementedError("Planner search is not implemented.")
