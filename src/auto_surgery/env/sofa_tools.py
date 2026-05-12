"""Tool registry hooks for SOFA POC scenes (forceps first; more instruments later)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto_surgery.schemas.commands import RobotCommand


def _set_translation(ogl_model: Any, translation: tuple[float, float, float]) -> bool:
    x, y, z = translation
    translation_vector = [float(x), float(y), float(z)]
    for value in (
        getattr(ogl_model, "translation", None),
        getattr(ogl_model, "findData", lambda _name: None)("translation")
        if hasattr(ogl_model, "findData")
        else None,
    ):
        if value is None:
            continue
        try:
            if hasattr(value, "value"):
                value.value = translation_vector
            else:
                ogl_model.translation = translation_vector
            # Some SOFA bindings require an explicit data-set call instead of
            # overwriting a `.value` attribute.
            if hasattr(value, "setValue"):
                value.setValue(translation_vector)
            return True
        except Exception:
            continue
    return False


def _resolve_forceps_visual(root_node: Any) -> Any | None:
    forceps_node = None
    for key in ("Forceps", "forceps"):
        forceps_node = getattr(root_node, key, None)
        if forceps_node is not None:
            break
    if forceps_node is None and hasattr(root_node, "getChild"):
        for key in ("Forceps", "forceps"):
            try:
                forceps_node = root_node.getChild(key)
                if forceps_node is not None:
                    break
            except Exception:
                forceps_node = None
    if forceps_node is None:
        return None

    for key in ("forcepsVisual", "ForcepsVisual"):
        obj = getattr(forceps_node, key, None)
        if obj is not None:
            return obj
    if hasattr(forceps_node, "getObject"):
        try:
            return forceps_node.getObject("forcepsVisual")
        except Exception:
            return None
    return None


def build_forceps_action_applier(
    *, x_base: float = 0.0, y: float = -30.0, z: float = 40.0, scale: float = 10.0
) -> Callable[[Any, RobotCommand], None]:
    """Return a cached action applier for the named forceps OGL visual."""

    cached_visual: dict[str, Any | None] = {"node": None}

    def _apply_action(scene: Any, action: RobotCommand) -> None:
        if action.joint_positions is None:
            return
        j0 = action.joint_positions.get("j0")
        if j0 is None:
            return
        if cached_visual["node"] is None:
            cached_visual["node"] = _resolve_forceps_visual(scene)
        if cached_visual["node"] is None:
            return
        _set_translation(
            cached_visual["node"],
            (x_base + float(j0) * scale, y, z),
        )

    return _apply_action


def _stub_tool(tool: str) -> Callable[..., Callable[[Any, RobotCommand], None]]:
    def build(**_kwargs: Any) -> Callable[[Any, RobotCommand], None]:
        raise NotImplementedError(
            f"SOFA tool {tool!r} is registered for future work but is not implemented."
        )

    return build


TOOL_REGISTRY: dict[str, Callable[..., Callable[[Any, RobotCommand], None]]] = {
    "forceps": build_forceps_action_applier,
    "scissors": _stub_tool("scissors"),
    "scalpel": _stub_tool("scalpel"),
    "needle": _stub_tool("needle"),
}


def resolve_tool_action_applier(tool_id: str, **kwargs: Any) -> Callable[[Any, RobotCommand], None]:
    """Return an action applier callable for ``tool_id``."""

    key = tool_id.strip().lower()
    builder = TOOL_REGISTRY.get(key)
    if builder is None:
        raise KeyError(f"Unknown SOFA tool id: {tool_id!r}")
    return builder(**kwargs)
