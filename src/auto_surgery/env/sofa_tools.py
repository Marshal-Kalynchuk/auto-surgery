"""SOFA tool action appliers and observers for the forceps shaft Rigid3d DOF."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from auto_surgery.env.sofa_scenes.forceps_assets import (
    DEFAULT_TOOL_TIP_OFFSET_LOCAL,
    _clasper_visual_transform,
    _shaft_origin_twist_from_tip_twist,
    _twist_camera_to_scene,
    load_dejavu_forceps_defaults,
)
from auto_surgery.schemas.commands import (
    ControlMode,
    Pose,
    Quaternion,
    RobotCommand,
    Twist,
    Vec3,
)
from auto_surgery.schemas.scene import ToolSpec
from auto_surgery.schemas.sensors import ToolState

_IDENTITY_POSE = Pose(
    position=Vec3(x=0.0, y=0.0, z=0.0),
    rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
)
_ZERO_TWIST = Twist(
    linear=Vec3(x=0.0, y=0.0, z=0.0),
    angular=Vec3(x=0.0, y=0.0, z=0.0),
)
_ZERO_VEC = Vec3(x=0.0, y=0.0, z=0.0)
_ZERO_TWIST_TUPLE: tuple[float, float, float, float, float, float] = (0.0,) * 6


def _to_list(value: Any) -> list[float]:
    """Flatten a SOFA Data value (scalar, sequence, or 2-D array) to ``list[float]``.

    ``SofaPython3`` exposes per-DOF MechanicalObject fields as nested
    ``[[v0, v1, ...]]`` numpy-like structures (one outer entry per DOF). The
    forceps shaft has exactly one DOF, so we always return the first row's
    floats.
    """

    if value is None:
        return []
    if isinstance(value, str):
        return [float(token) for token in value.split()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return [float(v) for v in value[0]]
        return [float(v) for v in value]
    try:
        items = list(iter(value))
    except TypeError as exc:
        raise TypeError(
            f"Cannot coerce SOFA data field of type {type(value).__name__} to list[float]."
        ) from exc
    if not items:
        return []
    head = items[0]
    if isinstance(head, (list, tuple)):
        return [float(v) for v in head]
    if hasattr(head, "__iter__") and not isinstance(head, str):
        try:
            return [float(v) for v in head]
        except TypeError:
            pass
    return [float(x) for x in items]


def _safe_get_data(obj: Any, name: str) -> Any | None:
    if obj is None:
        return None
    direct = getattr(obj, name, None)
    if direct is not None:
        return direct
    find_data = getattr(obj, "findData", None)
    if callable(find_data):
        return find_data(name)
    return None


def _safe_set_data_field(target: Any, values: Any) -> bool:
    """Assign ``values`` to a SOFA Data-like field.

    Real ``SofaPython3`` MechanicalObject Data fields are 2-D
    ``(num_dofs, dof_size)`` arrays and reject flat 1-D writes with a
    ``TypeError: 'float' object is not iterable``. Test mocks instead expose a
    plain Python ``list`` and accept either shape. We try the value as-is,
    then wrapped in a single outer list, and finally re-raise the original
    TypeError if both fail.
    """

    if target is None:
        return False
    has_value = hasattr(target, "value")
    if not has_value and not hasattr(target, "__setitem__"):
        return False
    last_error: TypeError | None = None
    for candidate in (values, [values]):
        try:
            if has_value:
                target.value = candidate
            else:
                for i in range(len(candidate)):
                    target[i] = candidate[i]
            return True
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return False


def _read_vector_field(handle: Any, name: str, *, length: int) -> list[float] | None:
    data = _safe_get_data(handle, name)
    if data is None:
        return None
    raw = data.value if hasattr(data, "value") else data
    try:
        values = _to_list(raw)
    except (TypeError, ValueError):
        return None
    if len(values) < length:
        return None
    return [float(v) for v in values[:length]]


def _read_pose_from_handle(handle: Any) -> Pose:
    """Return the shaft Rigid3d pose ``(position, quaternion)``.

    The position field is shape ``(num_dofs, 7)`` for a Rigid3d MO; we always
    consume DOF 0. Falls back to identity when no position data is available.
    """

    values = _read_vector_field(handle, "position", length=7)
    if values is None:
        # Some test mocks store position as a plain 3-element vector.
        translation = _read_vector_field(handle, "position", length=3)
        if translation is None:
            return _IDENTITY_POSE
        return Pose(
            position=Vec3(x=translation[0], y=translation[1], z=translation[2]),
            rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    return Pose(
        position=Vec3(x=values[0], y=values[1], z=values[2]),
        rotation=Quaternion(w=values[6], x=values[3], y=values[4], z=values[5]),
    )


def _read_twist_from_handle(handle: Any) -> Twist:
    """Return the shaft Rigid3d twist ``(linear, angular)``.

    Real SOFA Rigid3d MOs expose the full twist on the ``velocity`` field
    (shape ``(1, 6)``). Tests instead split it across legacy ``v``/``w`` Vec3
    fields, which we still honor for backward compatibility.
    """

    values = _read_vector_field(handle, "velocity", length=6)
    if values is not None:
        return Twist(
            linear=Vec3(x=values[0], y=values[1], z=values[2]),
            angular=Vec3(x=values[3], y=values[4], z=values[5]),
        )
    linear = _read_vector_field(handle, "v", length=3) or [0.0, 0.0, 0.0]
    angular = _read_vector_field(handle, "w", length=3) or [0.0, 0.0, 0.0]
    return Twist(
        linear=Vec3(x=linear[0], y=linear[1], z=linear[2]),
        angular=Vec3(x=angular[0], y=angular[1], z=angular[2]),
    )


def _write_velocity_to_handle(handle: Any, twist: tuple[float, float, float, float, float, float]) -> bool:
    """Write a 6-DOF twist to the shaft Rigid3d velocity field.

    Real SOFA Rigid3d MOs expose the full twist on the ``velocity`` field;
    tests instead split it across legacy ``v``/``w`` Vec3 fields. We try the
    canonical Rigid3d field first and fall back to the split fields so the
    same applier works against both.
    """

    velocity = _safe_get_data(handle, "velocity")
    if velocity is not None:
        return _safe_set_data_field(velocity, list(twist))
    legacy_v = _safe_get_data(handle, "v")
    legacy_w = _safe_get_data(handle, "w")
    if legacy_v is None and legacy_w is None:
        return False
    wrote_any = False
    if legacy_v is not None and _safe_set_data_field(legacy_v, list(twist[:3])):
        wrote_any = True
    if legacy_w is not None and _safe_set_data_field(legacy_w, list(twist[3:])):
        wrote_any = True
    return wrote_any


def _write_clasper_transform(
    visual: Any,
    jaw: float,
    side: str,
    *,
    jaw_open_angle_rad: float,
    jaw_closed_angle_rad: float,
    hinge_origin_local: tuple[float, float, float],
    hinge_axis_local: tuple[float, float, float],
) -> None:
    if visual is None:
        return
    transform = _clasper_visual_transform(
        jaw,
        side,
        jaw_open_angle_rad=jaw_open_angle_rad,
        jaw_closed_angle_rad=jaw_closed_angle_rad,
        hinge_origin_local=hinge_origin_local,
        hinge_axis_local=hinge_axis_local,
    )
    _safe_set_data_field(_safe_get_data(visual, "translation"), list(transform.translation))
    _safe_set_data_field(_safe_get_data(visual, "rotation"), list(transform.euler_xyz))


def _read_wrench_field(handle: Any) -> Vec3:
    for name in ("force", "wrench"):
        values = _read_vector_field(handle, name, length=3)
        if values is not None:
            return Vec3(x=values[0], y=values[1], z=values[2])
    return _ZERO_VEC


def _read_in_contact(shaft_collision_mos: tuple[Any | None, ...] | None) -> bool:
    if not shaft_collision_mos:
        return False
    for handle in shaft_collision_mos:
        if handle is None:
            continue
        contact_data = _safe_get_data(handle, "contacts")
        if contact_data is not None and hasattr(contact_data, "value"):
            contact_data = contact_data.value
        if isinstance(contact_data, str):
            if contact_data.strip():
                return True
            continue
        if hasattr(contact_data, "__len__"):
            try:
                if len(contact_data) > 0:
                    return True
            except TypeError:
                continue
    return False


def _read_camera_pose(scene: Any) -> Pose:
    """Best-effort pose read for the offscreen capture camera.

    Used as a fallback when the orchestration layer can't supply the
    authoritative camera pose via ``camera_pose_provider``.
    """

    if scene is None:
        return _IDENTITY_POSE
    for name in ("auto_surgery_offscreen_camera", "camera", "Camera"):
        candidate = _resolve_handle(scene, (name,))
        position = _read_vector_field(candidate, "position", length=3)
        if position is None:
            continue
        rotation_values = _read_vector_field(candidate, "orientation", length=4)
        if rotation_values is None:
            rotation_values = _read_vector_field(candidate, "rotation", length=4)
        if rotation_values is None:
            rotation_values = [0.0, 0.0, 0.0, 1.0]
        qx, qy, qz, qw = rotation_values
        return Pose(
            position=Vec3(x=position[0], y=position[1], z=position[2]),
            rotation=Quaternion(w=qw, x=qx, y=qy, z=qz),
        )
    return _IDENTITY_POSE


def _resolve_handle(container: Any, names: tuple[str, ...]) -> Any | None:
    """Look up a child node / object by name with cheap fallbacks for SOFA bindings."""

    if container is None:
        return None
    for name in names:
        direct = _safe_get_data(container, name)
        if direct is not None:
            return direct
        for accessor_name in ("getObject", "getChild"):
            accessor = getattr(container, accessor_name, None)
            if not callable(accessor):
                continue
            try:
                return accessor(name)
            except Exception:
                continue
    return None


def _resolve_visual_handle(
    container: Any,
    *,
    names: tuple[str, ...],
    intermediate_node_names: tuple[str, ...] = (),
) -> Any | None:
    """Resolve a visual handle, drilling through intermediate Visual nodes if needed.

    The Python forceps factory wraps the OglModel inside a child node (e.g.
    ``ShaftVisual`` -> ``shaftVisual``). The XML scene template renders the
    visual directly as a child of ``Shaft``. Drill into intermediate wrapper
    nodes first so we prefer the inner OglModel over the wrapper.
    """

    if container is None:
        return None
    get_child = getattr(container, "getChild", None)
    for node_name in intermediate_node_names:
        intermediate: Any | None = None
        if callable(get_child):
            try:
                intermediate = get_child(node_name)
            except Exception:
                continue
        if intermediate is None or intermediate is container:
            continue
        nested = _resolve_handle(intermediate, names)
        if nested is not None and nested is not intermediate:
            return nested
    return _resolve_handle(container, names)


@dataclass(frozen=True)
class _DiscoveredForcepsHandles:
    forceps_node: Any | None = None
    shaft_node: Any | None = None
    dof: Any | None = None
    visual: Any | None = None
    shaft_visual: Any | None = None
    left_clasper: Any | None = None
    right_clasper: Any | None = None
    shaft_collision_mos: tuple[Any | None, ...] = ()


_FORCEPS_NODE_NAMES = ("Forceps", "forceps")
_SHAFT_NODE_NAMES = ("Shaft", "shaft", "forcepsShaft", "forceps_shaft")
_SHAFT_DOF_NAMES = (
    "shaftMO",
    "shaftMo",
    "shaft_mo",
    "shaftDof",
    "forcepsDof",
    "forcepsMO",
    "forcepsMo",
)
_SHAFT_VISUAL_NAMES = (
    "ShaftVisual",
    "shaftVisual",
    "shaft_visual",
    "forcepsVisual",
    "ForcepsVisual",
    "forcepsBody",
    "ForcepsBody",
)
_LEFT_CLASPER_NAMES = (
    "ClasperLeft",
    "clasperLeft",
    "ClasperLeftVisual",
    "clasperLeftVisual",
    "ForcepsClasperLeft",
    "forcepsClasperLeft",
)
_RIGHT_CLASPER_NAMES = (
    "ClasperRight",
    "clasperRight",
    "ClasperRightVisual",
    "clasperRightVisual",
    "ForcepsClasperRight",
    "forcepsClasperRight",
)
_SHAFT_COLLISION_NODE_NAMES = (
    "ShaftCollision",
    "shaftCollision",
    "ShaftCollisionNode",
    "shaftCollisionNode",
)
_SHAFT_COLLISION_OBJECT_NAMES = (
    "shaftCollisionTriangle",
    "shaftCollisionLine",
    "shaftCollisionPoint",
    "ShaftCollisionTriangle",
    "ShaftCollisionLine",
    "ShaftCollisionPoint",
)


def _discover_shaft_collision_mos(shaft_node: Any) -> tuple[Any, ...]:
    if shaft_node is None:
        return ()
    collision_node = _resolve_handle(shaft_node, _SHAFT_COLLISION_NODE_NAMES)
    container = collision_node or shaft_node
    found: list[Any] = []
    for name in _SHAFT_COLLISION_OBJECT_NAMES:
        handle = _resolve_handle(container, (name,))
        if handle is not None and handle not in found:
            found.append(handle)
    return tuple(found)


def _discover_forceps_handles(scene: Any) -> _DiscoveredForcepsHandles:
    if scene is None:
        return _DiscoveredForcepsHandles()
    forceps_node = _resolve_handle(scene, _FORCEPS_NODE_NAMES)
    if forceps_node is None:
        return _DiscoveredForcepsHandles()
    shaft_node = _resolve_handle(forceps_node, _SHAFT_NODE_NAMES)
    dof = _resolve_handle(shaft_node or forceps_node, _SHAFT_DOF_NAMES)
    shaft_visual = _resolve_visual_handle(
        shaft_node or forceps_node,
        names=_SHAFT_VISUAL_NAMES,
        intermediate_node_names=("ShaftVisual", "shaftVisual", "Visual", "visual"),
    )
    left_clasper = _resolve_visual_handle(
        shaft_node or forceps_node,
        names=_LEFT_CLASPER_NAMES,
        intermediate_node_names=_LEFT_CLASPER_NAMES,
    )
    right_clasper = _resolve_visual_handle(
        shaft_node or forceps_node,
        names=_RIGHT_CLASPER_NAMES,
        intermediate_node_names=_RIGHT_CLASPER_NAMES,
    )
    return _DiscoveredForcepsHandles(
        forceps_node=forceps_node,
        shaft_node=shaft_node,
        dof=dof,
        visual=shaft_visual,
        shaft_visual=shaft_visual,
        left_clasper=left_clasper,
        right_clasper=right_clasper,
        shaft_collision_mos=_discover_shaft_collision_mos(shaft_node or forceps_node),
    )


def build_forceps_velocity_applier(
    *,
    force_scale: float = 0.01,
    forceps_dof: Any | None = None,
    jaw_open_angle_rad: float = 0.30,
    jaw_closed_angle_rad: float = 0.0,
    hinge_origin_local: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hinge_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0),
    tool_tip_offset_local: tuple[float, float, float] = DEFAULT_TOOL_TIP_OFFSET_LOCAL,
    jaw_ref: dict[str, float] | None = None,
    camera_pose_provider: Callable[[], Pose] | None = None,
) -> Callable[[Any, RobotCommand], None]:
    """Create the action applier that drives the Rigid3d shaft DOF from twists.

    The shaft's OglModel is animated by SOFA via ``RigidMapping`` from the
    Rigid3d shaft DOF, so we only need to write velocities on the DOF here;
    the per-clasper visual transforms still need explicit kinematic writes.
    """

    if jaw_ref is None:
        jaw_ref = {"jaw": 0.0}
    state: dict[str, Any] = {
        "dof": forceps_dof,
        "left_clasper": None,
        "right_clasper": None,
        "scene_id": None,
    }

    def _refresh_handles(scene: Any) -> None:
        scene_id = id(scene) if scene is not None else None
        if scene_id is not None and state["scene_id"] != scene_id:
            state["dof"] = forceps_dof
            state["left_clasper"] = None
            state["right_clasper"] = None
            state["scene_id"] = scene_id
        if scene is not None and (
            state["dof"] is None
            or state["left_clasper"] is None
            or state["right_clasper"] is None
        ):
            handles = _discover_forceps_handles(scene)
            state["dof"] = state["dof"] or handles.dof
            state["left_clasper"] = state["left_clasper"] or handles.left_clasper
            state["right_clasper"] = state["right_clasper"] or handles.right_clasper

    def _apply_claspers() -> None:
        jaw = float(jaw_ref.get("jaw", 0.0))
        for handle, side in ((state["left_clasper"], "left"), (state["right_clasper"], "right")):
            _write_clasper_transform(
                handle,
                jaw,
                side=side,
                jaw_open_angle_rad=jaw_open_angle_rad,
                jaw_closed_angle_rad=jaw_closed_angle_rad,
                hinge_origin_local=hinge_origin_local,
                hinge_axis_local=hinge_axis_local,
            )

    def _resolve_camera_pose(scene: Any) -> Pose:
        if camera_pose_provider is not None:
            try:
                pose = camera_pose_provider()
            except Exception:
                pose = None
            if isinstance(pose, Pose):
                return pose
        return _read_camera_pose(scene)

    def _command_disengaged(action: RobotCommand) -> bool:
        return (
            action.control_mode != ControlMode.CARTESIAN_TWIST
            or not action.enable
            or action.cartesian_twist is None
        )

    def _apply_action(scene: Any, action: RobotCommand) -> None:
        _refresh_handles(scene)
        if action.tool_jaw_target is not None:
            jaw_ref["jaw"] = float(action.tool_jaw_target)

        if _command_disengaged(action):
            _write_velocity_to_handle(state["dof"], _ZERO_TWIST_TUPLE)
            _apply_claspers()
            return

        scene_pose = _resolve_camera_pose(scene)
        scaled_camera_twist = Twist(
            linear=Vec3(
                x=float(action.cartesian_twist.linear.x) * force_scale,
                y=float(action.cartesian_twist.linear.y) * force_scale,
                z=float(action.cartesian_twist.linear.z) * force_scale,
            ),
            angular=action.cartesian_twist.angular,
        )
        scene_twist = _twist_camera_to_scene(scaled_camera_twist, scene_pose)
        shaft_twist = _shaft_origin_twist_from_tip_twist(
            scene_twist,
            _read_pose_from_handle(state["dof"]),
            tool_tip_offset_local,
        )
        _write_velocity_to_handle(state["dof"], shaft_twist)
        _apply_claspers()

    if state["dof"] is not None:
        _apply_claspers()
    return _apply_action


def build_forceps_observer(
    *,
    dof: Any | None,
    jaw_ref: dict[str, float] | None = None,
) -> Callable[[Any], ToolState]:
    """Create the observer that surfaces ``ToolState`` for the forceps shaft."""

    jaw_state = jaw_ref or {"jaw": 0.0}
    cached: dict[str, Any] = {
        "dof": dof,
        "shaft_collision_mos": (),
        "scene_id": None,
        "dof_explicit": dof is not None,
    }

    def _observe(scene: Any) -> ToolState:
        scene_id = id(scene) if scene is not None else None
        if scene_id is not None and cached["scene_id"] != scene_id:
            if not cached["dof_explicit"]:
                cached["dof"] = None
            cached["shaft_collision_mos"] = ()
            cached["scene_id"] = scene_id
        if scene is not None and (cached["dof"] is None or not cached["shaft_collision_mos"]):
            discovered = _discover_forceps_handles(scene)
            if cached["dof"] is None and discovered.dof is not None:
                cached["dof"] = discovered.dof
            cached["shaft_collision_mos"] = discovered.shaft_collision_mos

        current_dof = cached["dof"]
        return ToolState(
            pose=_read_pose_from_handle(current_dof),
            twist=_read_twist_from_handle(current_dof),
            jaw=float(jaw_state.get("jaw", 0.0)),
            wrench=_read_wrench_field(current_dof),
            in_contact=(
                _read_in_contact(cached["shaft_collision_mos"])
                if cached["shaft_collision_mos"]
                else bool(getattr(current_dof, "in_contact", False))
            ),
        )

    return _observe


def _unsupported_tool(tool: str) -> Callable[..., Callable[[Any, RobotCommand], None]]:
    def build(**_kwargs: Any) -> Callable[[Any, RobotCommand], None]:
        raise NotImplementedError(f"SOFA tool {tool!r} is planned and not implemented.")

    return build


IMPLEMENTED_TOOL_REGISTRY: dict[str, Callable[..., Callable[[Any, RobotCommand], None]]] = {
    "forceps": build_forceps_velocity_applier,
    "dejavu_forceps": build_forceps_velocity_applier,
}

PLANNED_TOOL_REGISTRY: dict[str, Callable[..., Callable[[Any, RobotCommand], None]]] = {
    "scissors": _unsupported_tool("scissors"),
    "scalpel": _unsupported_tool("scalpel"),
    "needle": _unsupported_tool("needle"),
}

TOOL_REGISTRY = {**IMPLEMENTED_TOOL_REGISTRY, **PLANNED_TOOL_REGISTRY}


def resolve_tool_action_applier(tool: str, **kwargs: Any) -> Callable[[Any, RobotCommand], None]:
    key = str(tool).strip().lower()
    builder = TOOL_REGISTRY.get(key)
    if builder is None:
        raise KeyError(f"Unknown SOFA tool id: {tool!r}")
    return builder(**kwargs)


def resolve_tool_action_applier_from_spec(
    tool_spec: ToolSpec,
    *,
    forceps_dof: Any | None = None,
    jaw_ref: dict[str, float] | None = None,
    camera_pose_provider: Callable[[], Pose] | None = None,
) -> Callable[[Any, RobotCommand], None]:
    normalized_tool_id = str(tool_spec.tool_id).strip().lower()
    if normalized_tool_id not in {"forceps", "dejavu_forceps"}:
        return resolve_tool_action_applier(tool_spec.tool_id)
    contract = load_dejavu_forceps_defaults()
    assembly = contract.assembly
    return build_forceps_velocity_applier(
        forceps_dof=forceps_dof,
        jaw_open_angle_rad=assembly.jaw_open_angle_rad,
        jaw_closed_angle_rad=assembly.jaw_closed_angle_rad,
        hinge_origin_local=assembly.hinge_origin_local,
        hinge_axis_local=assembly.hinge_axis_local,
        tool_tip_offset_local=assembly.tool_tip_offset_local,
        jaw_ref=jaw_ref,
        camera_pose_provider=camera_pose_provider,
    )
