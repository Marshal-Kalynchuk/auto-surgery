"""Scene models and scene graph utilities for SOFA runtime configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from auto_surgery.env.sofa_scenes.poc_scene import (
    camera_pose_scene_from_look_mm,
    parse_poc_scene,
)

from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.sensors import CameraIntrinsics


class SlotRecord(BaseModel):
    """One persistent slot instance."""

    model_config = {"extra": "forbid"}

    slot_id: str
    pose: dict[str, float] | None = None
    geometry_ref: str | None = Field(default=None, description="Blob URI for mesh/pointcloud.")
    embedding_ref: str | None = Field(default=None, description="Pointer to embedding tensor blob.")
    derived_labels: dict[str, Any] = Field(default_factory=dict)
    surgeon_tags: list[dict[str, Any]] = Field(default_factory=list)


class SceneGraph(BaseModel):
    """Snapshot of slots and lightweight scene metadata."""

    model_config = {"extra": "forbid"}

    schema_version: str = "scene_v1"
    frame_index: int = 0
    slots: list[SlotRecord] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)


def _identity_pose() -> Pose:
    return Pose(
        position=Vec3(x=0.0, y=0.0, z=0.0),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )


def _default_camera_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        width=640,
        height=480,
    )


class VisualOverrides(BaseModel):
    """Optional per-part visual overrides for the rigid forceps subtree."""

    model_config = {"extra": "forbid"}

    body_uv_path: Path | None = None
    clasper_left_uv_path: Path | None = None
    clasper_right_uv_path: Path | None = None
    body_color: tuple[float, float, float, float] | None = None
    clasper_color: tuple[float, float, float, float] | None = None
    tissue_texture_tint_rgb: tuple[float, float, float] | None = None


class TissueMaterial(BaseModel):
    """FEM constants for the tissue body in the rendered `.scn`."""

    model_config = {"extra": "forbid"}

    young_modulus_pa: float = Field(default=3000.0, gt=0.0)
    poisson_ratio: float = Field(default=0.45, ge=0.0, lt=0.5)
    total_mass_kg: float = Field(default=0.5, gt=0.0)
    rayleigh_stiffness: float = Field(default=0.1, ge=0.0)


class TissueTopology(BaseModel):
    """SparseGridTopology resolution for the tissue body."""

    model_config = {"extra": "forbid"}

    sparse_grid_n: tuple[int, int, int] = (16, 16, 16)

    @model_validator(mode="after")
    def _positive(self) -> "TissueTopology":
        if any(n <= 0 for n in self.sparse_grid_n):
            raise ValueError("sparse_grid_n components must be positive.")
        return self


class BulgeSpec(BaseModel):
    """Radial Gaussian displacement centred on a point in scene frame.

    ``radius_scene`` and ``amplitude_scene`` use the same length unit as scene geometry
    (millimetres for DejaVu brain meshes).
    """

    model_config = {"extra": "forbid"}

    center_scene: Vec3
    radius_scene: float = Field(gt=0.0)
    amplitude_scene: float


class MeshPerturbation(BaseModel):
    """Per-episode warp applied to tissue volume + surface meshes."""

    model_config = {"extra": "forbid"}

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    translation_scene: Vec3 = Field(
        default_factory=lambda: Vec3(x=0.0, y=0.0, z=0.0)
    )
    bulge: BulgeSpec | None = None

    def is_identity(self) -> bool:
        return (
            self.scale == (1.0, 1.0, 1.0)
            and self.translation_scene == Vec3(x=0.0, y=0.0, z=0.0)
            and self.bulge is None
        )


class DirectionalLight(BaseModel):
    """One directional light source for the rendered `.scn`."""

    model_config = {"extra": "forbid"}

    direction_scene: Vec3
    intensity: float = Field(default=1.0, ge=0.0)
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)


class SpotLight(BaseModel):
    """One spot light source for the rendered `.scn`."""

    model_config = {"extra": "forbid"}

    position_scene: Vec3
    direction_scene: Vec3
    cone_half_angle_deg: float = Field(default=30.0, gt=0.0, le=89.0)
    intensity: float = Field(default=1.0, ge=0.0)
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)


class LightingSpec(BaseModel):
    """Lights and background for the rendered `.scn`."""

    model_config = {"extra": "forbid"}

    directional: DirectionalLight | None = None
    spot: SpotLight | None = None
    background_rgb: tuple[float, float, float] = (0.05, 0.05, 0.08)


class ToolSpec(BaseModel):
    """Typed tool configuration for strict Piece-2 scene wiring."""

    model_config = {"extra": "forbid"}

    tool_id: Literal["dejavu_forceps"] = "dejavu_forceps"
    initial_pose_scene: Pose = Field(default_factory=_identity_pose)
    initial_jaw: float = Field(default=0.0, ge=0.0, le=1.0)
    visual_overrides: VisualOverrides | None = None


class TargetVolume(BaseModel):
    """Region of surgical interest in scene coordinates."""

    model_config = {"extra": "forbid"}

    label: Literal["tumor", "vessel", "general"]
    center_scene: Vec3
    half_extents_scene: Vec3
    shape: Literal["sphere", "bbox"] = "sphere"


def _default_target_volumes() -> list[TargetVolume]:
    return [
        TargetVolume(
            label="general",
            center_scene=Vec3(x=0.0, y=0.0, z=0.0),
            half_extents_scene=Vec3(x=0.001, y=0.001, z=0.001),
        )
    ]


_REPO_ROOT = Path(__file__).resolve().parents[2]


class TissueAssetSpec(BaseModel):
    """Relative tissue asset overrides resolved against DejaVu root."""

    model_config = {"extra": "forbid"}

    volume_mesh_relative_path: str = "scenes/brain/data/volume_simplified.obj"
    surface_mesh_relative_path: str = "scenes/brain/data/surface_full.obj"
    texture_relative_path: str = "scenes/brain/data/texture_outpaint.png"


class SceneConfig(BaseModel):
    """Piece-2 scene configuration with explicit tool contract."""

    model_config = {"extra": "forbid"}

    scene_id: Literal["dejavu_brain"] = "dejavu_brain"
    tissue_scene_path: Path
    poc_scene_path: Path | None = Field(
        default=None,
        description="Optional upstream POC `.scn`; viewpoint and viewport baseline when fields are omitted.",
    )
    tool: ToolSpec = Field(default_factory=ToolSpec)
    camera_extrinsics_scene: Pose = Field(default_factory=_identity_pose)
    camera_intrinsics: CameraIntrinsics = Field(default_factory=_default_camera_intrinsics)
    tissue_assets: TissueAssetSpec | None = None
    target_volumes: list[TargetVolume] = Field(
        default_factory=_default_target_volumes,
        min_length=1,
    )
    tissue_material: TissueMaterial = Field(default_factory=TissueMaterial)
    tissue_topology: TissueTopology = Field(default_factory=TissueTopology)
    tissue_mesh_perturbation: MeshPerturbation = Field(
        default_factory=MeshPerturbation
    )
    lighting: LightingSpec = Field(default_factory=LightingSpec)

    @model_validator(mode="before")
    @classmethod
    def _merge_baseline_viewpoint_from_poc(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        poc_raw = data.get("poc_scene_path")
        if poc_raw is None:
            return data

        poc_requested = Path(str(poc_raw))
        resolved_path: Path | None = None
        for candidate in (poc_requested, _REPO_ROOT / poc_requested):
            if candidate.is_file():
                resolved_path = candidate.resolve()
                break

        if resolved_path is None:
            raise ValueError(
                "poc_scene_path must resolve to an existing file; looked for "
                f"{poc_requested} and {_REPO_ROOT / poc_requested}.",
            )

        poc = parse_poc_scene(resolved_path)
        canonical_pose_dm = camera_pose_scene_from_look_mm(
            poc.position_scene_mm,
            poc.lookat_scene_mm,
        ).model_dump(mode="python")

        ce_payload = data.get("camera_extrinsics_scene")
        if ce_payload is None:
            data["camera_extrinsics_scene"] = canonical_pose_dm
        elif isinstance(ce_payload, dict):
            merged_ce = dict(ce_payload)
            if "position" not in merged_ce:
                merged_ce["position"] = canonical_pose_dm["position"]
            if "rotation" not in merged_ce:
                merged_ce["rotation"] = canonical_pose_dm["rotation"]
            data["camera_extrinsics_scene"] = merged_ce
        else:
            return data

        raw_ci = data.get("camera_intrinsics")
        user_ci_keys = set(raw_ci.keys()) if isinstance(raw_ci, dict) else set()
        base_ci = _default_camera_intrinsics().model_dump(mode="python")
        if isinstance(raw_ci, dict):
            merged_ci = {**base_ci, **dict(raw_ci)}
        elif raw_ci is None:
            merged_ci = {**base_ci}
        else:
            return data

        if poc.viewport is not None:
            pw, ph = poc.viewport
            if "width" not in user_ci_keys:
                merged_ci["width"] = pw
            if "height" not in user_ci_keys:
                merged_ci["height"] = ph

        if poc.field_of_view_deg is not None:
            hid = merged_ci.get("height")
            if hid is None and poc.viewport:
                hid = poc.viewport[1]
            if "fy" not in user_ci_keys and isinstance(hid, (int, float)):
                height_f = float(hid)
                fovy_rad = math.radians(poc.field_of_view_deg)
                merged_ci["fy"] = height_f / (2.0 * math.tan(fovy_rad / 2.0))
                if "fx" not in user_ci_keys:
                    merged_ci["fx"] = merged_ci["fy"]

        width_px = merged_ci.get("width")
        height_px = merged_ci.get("height")
        if "cx" not in user_ci_keys and isinstance(width_px, (int, float)):
            merged_ci["cx"] = float(width_px) / 2.0
        if "cy" not in user_ci_keys and isinstance(height_px, (int, float)):
            merged_ci["cy"] = float(height_px) / 2.0

        data["camera_intrinsics"] = merged_ci
        return data
