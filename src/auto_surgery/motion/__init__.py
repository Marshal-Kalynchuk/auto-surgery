"""Public motion exports for generators and primitives."""

from __future__ import annotations

from auto_surgery.schemas.scene import TargetVolume
from auto_surgery.schemas.motion import MotionGeneratorConfig

__all__ = [
    "MotionGeneratorConfig",
    "Primitive",
    "PrimitiveKind",
    "RealisedPrimitive",
    "SurgicalMotionGenerator",
    "PrimitiveOutput",
    "Reach",
    "Hold",
    "ContactReach",
    "Grip",
    "Drag",
    "Brush",
    "TargetVolume",
]


def __getattr__(name: str):
    # Lazy-import generator/primitives attributes to avoid circular imports
    if name in {"RealisedPrimitive", "SurgicalMotionGenerator"}:
        from auto_surgery.motion.generator import RealisedPrimitive, SurgicalMotionGenerator

        return {"RealisedPrimitive": RealisedPrimitive, "SurgicalMotionGenerator": SurgicalMotionGenerator}[name]
    if name in {
        "PrimitiveOutput",
        "Reach",
        "Hold",
        "ContactReach",
        "Grip",
        "Drag",
        "Brush",
        "Primitive",
        "PrimitiveKind",
    }:
        from auto_surgery.motion.primitives import (
            PrimitiveOutput,
            Reach,
            Hold,
            ContactReach,
            Grip,
            Drag,
            Brush,
            Primitive,
            PrimitiveKind,
        )

        return {
            "PrimitiveOutput": PrimitiveOutput,
            "Reach": Reach,
            "Hold": Hold,
            "ContactReach": ContactReach,
            "Grip": Grip,
            "Drag": Drag,
            "Brush": Brush,
            "Primitive": Primitive,
            "PrimitiveKind": PrimitiveKind,
        }[name]


def __dir__():
    return __all__ + list(globals().keys())
