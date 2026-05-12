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
    discover_sofa_runtime_contract,
    resolve_sofa_runtime_import_candidates,
)
from auto_surgery.env.sofa_orchestration import SofaEnvironment


__all__ = [
    "SofaEnvironment",
    "build_sofa_runtime_backend",
    "resolve_sofa_runtime_import_candidates",
    "discover_sofa_runtime_contract",
    "DEFAULT_SOFA_MODULE_CANDIDATES",
    "SofaRuntimeBackend",
    "SofaRuntimeBackendFactory",
    "SofaSceneFactory",
    "SofaRuntimeContractError",
    "SofaNotIntegratedError",
]
