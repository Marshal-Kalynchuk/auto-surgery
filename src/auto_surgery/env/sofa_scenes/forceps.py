"""Rigid forceps construction primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_dejavu_asset_path
from auto_surgery.env.sofa_scenes.forceps_assets import load_dejavu_forceps_defaults
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.scene import VisualOverrides


@dataclass(frozen=True)
class ForcepsMeshSet:
    """Resolved mesh and texture paths for the forceps assembly."""

    shaft_obj_path: str
    shaft_collision_obj_path: str
    clasp_left_obj_path: str
    clasp_right_obj_path: str
    shaft_uv_path: str | None = None
    clasper_left_uv_path: str | None = None
    clasper_right_uv_path: str | None = None

    @property
    def body_obj_path(self) -> str:
        return self.shaft_obj_path

    @property
    def collision_obj_path(self) -> str:
        return self.shaft_collision_obj_path

    @property
    def body_uv_path(self) -> str | None:
        return self.shaft_uv_path


@dataclass(frozen=True)
class ForcepsAssemblyParams:
    """Runtime constants for a forceps tool assembly."""

    mass: float = 0.05
    scale: float = 1.0
    body_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 1.0)
    clasper_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 1.0)
    stiffness: float = 1000.0
    solver_iterations: int = 25
    solver_tolerance: float = 1e-9
    hinge_origin_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hinge_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    jaw_open_angle_rad: float = 0.30
    jaw_closed_angle_rad: float = 0.0
    tool_tip_offset_local: tuple[float, float, float] = (0.0, 0.0, 9.4)
    alarm_distance: float = 0.001
    contact_distance: float = 0.0002


@dataclass(frozen=True)
class ForcepsHandles:
    """Typed runtime handles for forceps subtree consumers."""

    forceps_node: Any
    shaft_mo: Any
    shaft_collision_mos: tuple[Any | None, ...]
    clasper_left_visual: Any | None
    clasper_right_visual: Any | None
    assembly: ForcepsAssemblyParams
    dofs: Any | None = None
    shaft_visual: Any | None = None
    jaw_position_source: Any | None = None


_DEFAULT_FORCEPS_COLLISION_MESH_PATH = str(
    (Path(__file__).resolve().parents[4] / "assets" / "forceps" / "shaft_tip_collision.obj").resolve()
)


def _load_default_forceps_contract() -> tuple[ForcepsAssemblyParams, str]:
    contract = load_dejavu_forceps_defaults()
    return contract.assembly, contract.collision_mesh_path


def _resolve_mesh_set(*, collision_mesh_path: str | None = None) -> ForcepsMeshSet:
    tool_root = resolve_dejavu_asset_path("scenes/liver/data/dv_tool")
    collision_path = collision_mesh_path or _DEFAULT_FORCEPS_COLLISION_MESH_PATH
    return ForcepsMeshSet(
        shaft_obj_path=str(tool_root / "body_uv.obj"),
        shaft_collision_obj_path=str(collision_path),
        clasp_left_obj_path=str(tool_root / "clasper1_uv.obj"),
        clasp_right_obj_path=str(tool_root / "clasper2_uv.obj"),
        shaft_uv_path=str(tool_root / "instru.png"),
        clasper_left_uv_path=str(tool_root / "instru_clasper.png"),
        clasper_right_uv_path=str(tool_root / "instru_clasper.png"),
    )


def _pose_data(pose: Pose) -> str:
    return (
        f"{pose.position.x:.9g} {pose.position.y:.9g} {pose.position.z:.9g} "
        f"{pose.rotation.x:.9g} {pose.rotation.y:.9g} {pose.rotation.z:.9g} {pose.rotation.w:.9g}"
    )


def _color_to_string(value: tuple[float, float, float, float]) -> str:
    return f"{value[0]} {value[1]} {value[2]} {value[3]}"


def _safe_add_object(node: Any, object_type: str, **kwargs: Any) -> Any | None:
    try:
        return node.addObject(object_type, **kwargs)
    except Exception:
        return None


def _add_required_object(node: Any, object_type: str, **kwargs: Any) -> Any | None:
    return _safe_add_object(node, object_type, **kwargs)


def _build_ogl(
    node: Any,
    *,
    name: str,
    mesh_path: str,
    texture: str | None,
    color: tuple[float, float, float, float],
    scale: float,
) -> Any | None:
    loader_name = f"{name}Loader"
    loader = _safe_add_object(
        node,
        "MeshOBJLoader",
        name=loader_name,
        filename=mesh_path,
    )
    if loader is None:
        return None

    kwargs: dict[str, Any] = {
        "name": name,
        "src": f"@{loader_name}",
        "color": _color_to_string(color),
    }
    if texture:
        kwargs["texturename"] = texture
    if scale != 0 and scale != 1.0:
        kwargs["scale3d"] = f"{scale} {scale} {scale}"
    return _safe_add_object(node, "OglModel", **kwargs)


def _extract_pose_data_object(obj: Any) -> Any | None:
    if obj is None:
        return None
    return getattr(obj, "position", None) or getattr(obj, "findData", lambda _name: None)("position")


def _apply_visual_overrides(
    default_meshes: ForcepsMeshSet,
    overrides: VisualOverrides | None,
    defaults: ForcepsAssemblyParams,
) -> tuple[ForcepsMeshSet, ForcepsAssemblyParams]:
    if overrides is None:
        return default_meshes, defaults

    updated_meshes = default_meshes
    updated_params = defaults
    if (
        overrides.body_uv_path is not None
        or overrides.clasper_left_uv_path is not None
        or overrides.clasper_right_uv_path is not None
    ):
        updated_meshes = ForcepsMeshSet(
            shaft_obj_path=updated_meshes.shaft_obj_path,
            shaft_collision_obj_path=updated_meshes.shaft_collision_obj_path,
            clasp_left_obj_path=updated_meshes.clasp_left_obj_path,
            clasp_right_obj_path=updated_meshes.clasp_right_obj_path,
            shaft_uv_path=(
                str(overrides.body_uv_path)
                if overrides.body_uv_path is not None
                else updated_meshes.shaft_uv_path
            ),
            clasper_left_uv_path=(
                str(overrides.clasper_left_uv_path)
                if overrides.clasper_left_uv_path is not None
                else updated_meshes.clasper_left_uv_path
            ),
            clasper_right_uv_path=(
                str(overrides.clasper_right_uv_path)
                if overrides.clasper_right_uv_path is not None
                else updated_meshes.clasper_right_uv_path
            ),
        )

    if overrides.body_color is not None:
        updated_params = replace(updated_params, body_color=overrides.body_color)
    if getattr(overrides, "clasper_color", None) is not None:
        updated_params = replace(updated_params, clasper_color=overrides.clasper_color)

    return updated_meshes, updated_params


def create_forceps_node(
    root_node: Any,
    *,
    node_name: str = "Forceps",
    mesh_set: ForcepsMeshSet | None = None,
    pose: Pose | None = None,
    visual_overrides: VisualOverrides | None = None,
    params: ForcepsAssemblyParams | None = None,
) -> ForcepsHandles:
    """Create a rigid-body forceps subtree with mapping/collision wiring."""

    if pose is None:
        pose = Pose(
            position=Vec3(x=0.0, y=0.0, z=0.0),
            rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    default_params: ForcepsAssemblyParams | None = None
    collision_mesh_path = _DEFAULT_FORCEPS_COLLISION_MESH_PATH
    if params is None or mesh_set is None:
        default_params, collision_mesh_path = _load_default_forceps_contract()
    if params is None:
        params = default_params or ForcepsAssemblyParams()
    meshes = _resolve_mesh_set(collision_mesh_path=collision_mesh_path) if mesh_set is None else mesh_set
    meshes, params = _apply_visual_overrides(meshes, visual_overrides, params)

    forceps_node = root_node.addChild(node_name)
    _add_required_object(
        forceps_node,
        "EulerImplicitSolver",
        name="forcepsSolver",
        printLog=False,
    )
    _add_required_object(
        forceps_node,
        "CGLinearSolver",
        name="forcepsLinearSolver",
        iterations=params.solver_iterations,
        tolerance=params.solver_tolerance,
        threshold=params.solver_tolerance,
    )

    shaft_node = forceps_node.addChild("Shaft")
    shaft_mo = _add_required_object(
        shaft_node,
        "MechanicalObject",
        template="Rigid3d",
        name="shaftMO",
        position=_pose_data(pose),
        velocity="0 0 0 0 0 0",
    )
    _add_required_object(
        shaft_node,
        "UniformMass",
        name="shaftMass",
        totalMass=params.mass,
    )
    _add_required_object(
        shaft_node,
        "UncoupledConstraintCorrection",
        name="shaftConstraintCorrection",
    )

    shaft_visual_node = shaft_node.addChild("ShaftVisual")
    shaft_visual = _build_ogl(
        shaft_visual_node,
        name="shaftVisual",
        mesh_path=meshes.shaft_obj_path,
        texture=meshes.shaft_uv_path,
        color=params.body_color,
        scale=params.scale,
    )
    _add_required_object(
        shaft_visual_node,
        "RigidMapping",
        name="shaftVisualMapping",
        input="@../shaftMO",
        output="@.",
    )

    collision_node = shaft_node.addChild("ShaftCollision")
    _safe_add_object(
        collision_node,
        "MeshOBJLoader",
        name="shaftCollisionLoader",
        filename=meshes.shaft_collision_obj_path,
    )
    _safe_add_object(
        collision_node,
        "MechanicalObject",
        template="Vec3d",
        name="shaftCollisionState",
    )
    shaft_collision_triangle = _safe_add_object(
        collision_node,
        "TriangleCollisionModel",
        name="shaftCollisionTriangle",
        group=0,
    )
    shaft_collision_line = _safe_add_object(
        collision_node,
        "LineCollisionModel",
        name="shaftCollisionLine",
        group=0,
    )
    shaft_collision_point = _safe_add_object(
        collision_node,
        "PointCollisionModel",
        name="shaftCollisionPoint",
        group=0,
    )
    _safe_add_object(
        collision_node,
        "RigidMapping",
        name="shaftCollisionMapping",
        input="@../shaftMO",
        output="@.",
    )
    _safe_add_object(
        forceps_node,
        "MinProximityIntersection",
        alarmDistance=params.alarm_distance,
        contactDistance=params.contact_distance,
    )
    _safe_add_object(
        forceps_node,
        "PenaltyContactForceField",
        response=params.stiffness,
        forceFactor=1.0,
    )

    clasper_left_node = shaft_node.addChild("ClasperLeft")
    clasper_left_visual = _build_ogl(
        clasper_left_node,
        name="clasperLeftVisual",
        mesh_path=meshes.clasp_left_obj_path,
        texture=meshes.clasper_left_uv_path,
        color=params.clasper_color,
        scale=params.scale,
    )
    _safe_add_object(
        clasper_left_node,
        "RigidMapping",
        name="clasperLeftMapping",
        input="@../shaftMO",
        output="@.",
    )

    clasper_right_node = shaft_node.addChild("ClasperRight")
    clasper_right_visual = _build_ogl(
        clasper_right_node,
        name="clasperRightVisual",
        mesh_path=meshes.clasp_right_obj_path,
        texture=meshes.clasper_right_uv_path,
        color=params.clasper_color,
        scale=params.scale,
    )
    _safe_add_object(
        clasper_right_node,
        "RigidMapping",
        name="clasperRightMapping",
        input="@../shaftMO",
        output="@.",
    )

    jaw_position_source = _extract_pose_data_object(shaft_mo)

    return ForcepsHandles(
        forceps_node=forceps_node,
        shaft_mo=shaft_mo,
        shaft_collision_mos=(
            shaft_collision_triangle,
            shaft_collision_line,
            shaft_collision_point,
        ),
        clasper_left_visual=clasper_left_visual,
        clasper_right_visual=clasper_right_visual,
        assembly=params,
        dofs=shaft_mo,
        shaft_visual=shaft_visual,
        jaw_position_source=jaw_position_source,
    )
