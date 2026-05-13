"""Public env package exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_surgery.env.protocol import Environment
    from auto_surgery.env.real import RealEnvironment
    from auto_surgery.env.sofa import SofaEnvironment


__all__ = ["Environment", "RealEnvironment", "SofaEnvironment"]

_EXPORTS = {
    "Environment": ("auto_surgery.env.protocol", "Environment"),
    "RealEnvironment": ("auto_surgery.env.real", "RealEnvironment"),
    "SofaEnvironment": ("auto_surgery.env.sofa", "SofaEnvironment"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'auto_surgery.env' has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
