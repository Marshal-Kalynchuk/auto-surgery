"""Read-only probes for deterministic SOFA integration tests (spec §9.3).

Depends on native SofaPython3 bindings and the DejaVu brain scene subtree layout
(brain FEM DOFs historically named ``Brain`` / ``gridDOFs``, matching the
packaged POC XML). These helpers are intentionally narrow — they locate one
MechanicalObject and read ``position``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def _find_named_object(container: Any, names: tuple[str, ...]) -> Any | None:
    """Resolve a SOFA ``BaseObject`` with any of ``names'' (matching order)."""

    if container is None:
        return None
    for name in names:
        direct = getattr(container, name, None)
        if direct is not None:
            return direct
    get_object = getattr(container, "getObject", None)
    if callable(get_object):
        for name in names:
            try:
                obj = get_object(name)
            except Exception:
                obj = None
            if obj is not None:
                return obj
    return None


def _enumerate_child_nodes(node: Any) -> list[Any]:
    children: list[Any] = []

    iterable = getattr(node, "children", None)
    if isinstance(iterable, (list, tuple)):
        return [c for c in iterable if c is not None]

    getter = getattr(node, "getChildren", None)
    if callable(getter):
        try:
            raw = getter()
            if isinstance(raw, (list, tuple)):
                return [c for c in raw if c is not None]
        except Exception:
            pass

    nb = getattr(node, "getNbChildren", None)
    gc = getattr(node, "getChild", None)
    if callable(nb) and callable(gc):
        try:
            count = int(nb())
            for idx in range(count):
                try:
                    ch = gc(idx)
                except Exception:
                    continue
                if ch is not None:
                    children.append(ch)
            if children:
                return children
        except Exception:
            pass

    gn = getattr(node, "getChildNames", None)
    ggc = getattr(node, "getChild", None)
    if callable(gn) and callable(ggc):
        try:
            for name in gn():
                try:
                    ch = ggc(name)
                except Exception:
                    continue
                if ch is not None:
                    children.append(ch)
        except Exception:
            pass

    return children


def iter_scene_nodes_dfs(root: Any) -> list[Any]:
    """DFS over SOFA nodes (handles multiple child APIs)."""
    out: list[Any] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        out.append(node)
        for child in reversed(_enumerate_child_nodes(node)):
            stack.append(child)
    return out


def find_brain_volume_mechanical_object(scene_root: Any) -> Any | None:
    """Return the coarse brain FEM MechanicalObject if discoverable."""

    brain_node = None
    for node in iter_scene_nodes_dfs(scene_root):
        name_attr = getattr(node, "name", None)
        name = ""
        try:
            if hasattr(name_attr, "value"):
                name = str(name_attr.value).strip().lower()
            elif name_attr is not None:
                name = str(name_attr).strip().lower()
        except Exception:
            name = ""
        if name == "brain":
            brain_node = node
            break

    dof_names = (
        "gridDOFs",
        "grid_dofs",
        "GridDOFs",
        "dof",
        "mechObj",
        "grid",
    )

    roots: tuple[Any, ...] = (brain_node,) if brain_node is not None else (scene_root,)
    mo: Any | None = None
    for sr in roots:
        mo = _find_named_object(sr, dof_names)
        if mo is not None:
            return mo

    for node in iter_scene_nodes_dfs(scene_root):
        get_object = getattr(node, "getObject", None)
        if not callable(get_object):
            continue
        for nm in dof_names:
            candidate: Any | None = None
            try:
                candidate = get_object(nm)
            except Exception:
                candidate = None
            if candidate is None:
                continue
            fd = getattr(candidate, "findData", None)
            if not callable(fd) or fd("position") is None:
                continue
            tmpl = getattr(candidate, "template_name", getattr(candidate, "template", ""))
            try:
                t = str(tmpl).lower()
                if not t or "vec3" in t:
                    return candidate
            except Exception:
                return candidate
    return None


def flatten_vec3_positions(mech_object: Any) -> tuple[float, ...]:
    """Return ``position`` as a flattened ``(x0,y0,z0,…)`` tuple."""

    fd = getattr(mech_object, "findData", None)
    blob: Any | None = None
    if callable(fd):
        try:
            blob = fd("position")
        except Exception:
            blob = None
    if blob is None:
        blob = getattr(mech_object, "position", None)
    raw: Sequence[float] | str | Any = ()
    try:
        if hasattr(blob, "value"):
            raw = blob.value
        else:
            raw = blob
    except Exception as exc:
        raise RuntimeError("Could not read SOFA MechanicalObject.position") from exc

    if isinstance(raw, str):
        vals = tuple(float(tok) for tok in raw.strip().split())
        if len(vals) < 3 or len(vals) % 3:
            raise RuntimeError(f"Unexpected position vector length ({len(vals)}).")
        return vals
    if hasattr(raw, "__iter__"):
        vals = tuple(float(v) for v in raw)
    else:
        raise RuntimeError(f"Unreadable position payload: {type(raw).__name__}")
    if len(vals) < 3 or len(vals) % 3:
        raise RuntimeError(f"Unexpected position vector length ({len(vals)}).")
    return vals


def peak_vertex_displacement_magnitude(
    reference_xyz: tuple[float, ...], current_xyz: tuple[float, ...]
) -> float:
    """Return max Euclidean vertex displacement magnitude between snapshots."""

    if len(reference_xyz) != len(current_xyz):
        raise ValueError(
            "Position arrays differ in length "
            f"({len(reference_xyz)} vs {len(current_xyz)}); cannot compare."
        )
    peak = 0.0
    for i in range(0, len(reference_xyz), 3):
        dx = float(current_xyz[i]) - float(reference_xyz[i])
        dy = float(current_xyz[i + 1]) - float(reference_xyz[i + 1])
        dz = float(current_xyz[i + 2]) - float(reference_xyz[i + 2])
        peak = max(peak, math.sqrt(dx * dx + dy * dy + dz * dz))
    return peak
