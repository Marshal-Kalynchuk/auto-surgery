from __future__ import annotations

from typing import Any


def detect_out_of_distribution(*, observation: Any, **_: Any) -> None:
    """Return a boolean/signal when a sample is out-of-distribution."""

    raise NotImplementedError("OOD detection is not implemented in Stage-0.")
