"""SOFA module discovery helpers."""

from __future__ import annotations

import importlib
from typing import Any

DEFAULT_SOFA_MODULE_CANDIDATES = ("sofa", "Sofa", "sofa.core", "Sofa.Core")


def _module_candidates_for_runtime() -> tuple[str, ...]:
    """Return ordered SOFA module import candidates."""

    return DEFAULT_SOFA_MODULE_CANDIDATES


def resolve_sofa_runtime_import_candidates(
    *, candidates: tuple[str, ...] | None = None
) -> tuple[str, Any] | tuple[None, None]:
    """Resolve the first importable SOFA module candidate."""

    for candidate in candidates or _module_candidates_for_runtime():
        try:
            return candidate, importlib.import_module(candidate)
        except ImportError:
            continue
    return None, None


def discover_sofa_runtime_contract() -> dict[str, Any]:
    """Collect module discovery metadata for docs/tests."""

    module_name, module = resolve_sofa_runtime_import_candidates()
    return {
        "candidates": tuple(_module_candidates_for_runtime()),
        "resolved_module_name": module_name,
        "resolved_module_path": getattr(module, "__file__", None) if module else None,
    }

