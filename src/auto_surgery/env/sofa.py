"""SOFA environment adapter with staged runtime integration."""

from __future__ import annotations

from auto_surgery.env.sofa_backend import (
    SofaNotIntegratedError,
    SofaRuntimeBackend,
    SofaRuntimeBackendFactory,
    SofaRuntimeContractError,
    SofaSceneFactory,
    build_sofa_runtime_backend,
)
from auto_surgery.env.sofa_discovery import (
    DEFAULT_SOFA_MODULE_CANDIDATES,
    _module_candidates_for_runtime as _discovery_module_candidates_for_runtime,
    resolve_sofa_runtime_import_candidates as _resolve_sofa_runtime_import_candidates,
)
from auto_surgery.env.sofa_orchestration import SofaEnvironment


def _module_candidates_for_runtime() -> tuple[str, ...]:
    """Return the ordered SOFA module import candidates.

    This indirection keeps a stable test patching point on the `auto_surgery.env.sofa`
    module while allowing discovery details to live in `sofa_discovery`.
    """

    return _discovery_module_candidates_for_runtime()


def resolve_sofa_runtime_import_candidates(
    *, candidates: tuple[str, ...] | None = None
) -> tuple[str, object] | tuple[None, None]:
    """Resolve the first importable SOFA module candidate.

    The `candidates` argument is supported for parity with the low-level discovery helper.
    """

    return _resolve_sofa_runtime_import_candidates(
        candidates=candidates or _module_candidates_for_runtime()
    )


def discover_sofa_runtime_contract() -> dict[str, object]:
    """Expose runtime discovery metadata with canonical candidate ordering."""

    module_name, module = _resolve_sofa_runtime_import_candidates(
        candidates=_module_candidates_for_runtime()
    )
    return {
        "candidates": tuple(_module_candidates_for_runtime()),
        "resolved_module_name": module_name,
        "resolved_module_path": getattr(module, "__file__", None) if module else None,
    }


__all__ = [
    "SofaEnvironment",
    "build_sofa_runtime_backend",
    "resolve_sofa_runtime_import_candidates",
    "discover_sofa_runtime_contract",
    "DEFAULT_SOFA_MODULE_CANDIDATES",
    "_module_candidates_for_runtime",
    "SofaRuntimeBackend",
    "SofaRuntimeBackendFactory",
    "SofaSceneFactory",
    "SofaRuntimeContractError",
    "SofaNotIntegratedError",
]
