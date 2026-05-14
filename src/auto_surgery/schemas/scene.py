"""Scene models and scene graph utilities for SOFA runtime configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator

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


class VisualToneAugmentation(BaseModel):
    """Per-frame deterministic visual augmentation applied at capture time."""

    model_config = {"extra": "forbid"}

    brightness_scale: float = Field(default=1.0, gt=0.0)
    contrast_scale: float = Field(default=1.0, gt=0.0)
    gamma: float = Field(default=1.0, gt=0.0)
    saturation_scale: float = Field(default=1.0, gt=0.0)

    def is_identity(self) -> bool:
        return (
            self.brightness_scale == 1.0
            and self.contrast_scale == 1.0
            and self.gamma == 1.0
            and self.saturation_scale == 1.0
        )


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
    workspace_envelope: "WorkspaceEnvelope | None" = None
    orientation_bias: "OrientationBias" = Field(default_factory=lambda: OrientationBias())
    visual_overrides: VisualOverrides | None = None


class WorkspaceEnvelope(BaseModel):
    """Per-tool workspace constraint surface and safety envelope."""

    model_config = {"extra": "forbid"}

    def signed_distance_to_envelope(self, p_scene: Vec3) -> float:
        raise NotImplementedError("Workspace envelope must implement signed distance.")


class SceneGeometryEnvelope(WorkspaceEnvelope):
    """Workspace envelope as an axis-aligned box around tissue surface mesh bounds.

    ``signed_distance_to_envelope`` returns the minimum margin to the **inflated**
    axis-aligned bounding box of the surface mesh (``outer_margin_mm`` on each
    side). Positive values are inside that box; negative values are outside.

    This does **not** exclude the interior of the brain mesh; collision-driven
    contact and FEM coupling remain separate concerns.
    """

    model_config = {"extra": "forbid"}

    outer_margin_mm: float = Field(gt=0.0)
    inner_margin_mm: float = Field(ge=0.0)
    no_go_regions: list[dict[str, Any]] = Field(default_factory=list)
    surface_mesh_path: Path | None = None
    _mesh_geometry: object | None = PrivateAttr(default=None)

    def signed_distance_to_envelope(self, p_scene: Vec3) -> float:
        if self.surface_mesh_path is None:
            return 0.0
        if self._mesh_geometry is None:
            from auto_surgery.env.scene_geometry import MeshSceneGeometry

            self._mesh_geometry = MeshSceneGeometry(str(self.surface_mesh_path))
        bmin, bmax = self._mesh_geometry.bounds()  # type: ignore[union-attr]
        margin = float(self.outer_margin_mm)
        bmin_arr = np.array(
            (float(bmin.x) - margin, float(bmin.y) - margin, float(bmin.z) - margin),
            dtype=float,
        )
        bmax_arr = np.array(
            (float(bmax.x) + margin, float(bmax.y) + margin, float(bmax.z) + margin),
            dtype=float,
        )
        p = np.array((float(p_scene.x), float(p_scene.y), float(p_scene.z)), dtype=float)
        return float(np.min(np.minimum(p - bmin_arr, bmax_arr - p)))


class SphereEnvelope(WorkspaceEnvelope):
    """Fallback spherical envelope around a scene center."""

    model_config = {"extra": "forbid"}

    center_scene: Vec3
    radius_mm: float = Field(gt=0.0)
    outer_margin_mm: float = Field(gt=0.0)
    inner_margin_mm: float = Field(ge=0.0)

    def signed_distance_to_envelope(self, p_scene: Vec3) -> float:
        dx = p_scene.x - self.center_scene.x
        dy = p_scene.y - self.center_scene.y
        dz = p_scene.z - self.center_scene.z
        center_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return center_distance - self.radius_mm


class CameraFrustumEnvelope(WorkspaceEnvelope):
    """Workspace envelope = camera view frustum bounded by [near, far] depth.

    ``signed_distance_to_envelope(p_scene)`` returns the minimum margin (in
    scene mm) to the six frustum half-spaces:

    * near plane (``z_cam = near_depth_mm``)
    * far plane (``z_cam = far_depth_mm``)
    * four lateral planes from the pinhole intrinsics, optionally shrunk by
      ``lateral_margin_fraction`` of the half image extents (e.g. 0.05 leaves
      a 5% margin around the image edges to keep the tip clearly visible).

    Lateral pixel distances are converted back to scene mm at the projected
    depth so the result is a single signed distance in scene units, ready to
    intersect with other envelopes via ``CompositeEnvelope``.
    """

    model_config = {"extra": "forbid"}

    camera_extrinsics_scene: Pose
    camera_intrinsics: CameraIntrinsics
    near_depth_mm: float = Field(default=5.0, gt=0.0)
    far_depth_mm: float = Field(default=300.0, gt=0.0)
    lateral_margin_fraction: float = Field(default=0.05, ge=0.0, lt=0.5)
    outer_margin_mm: float = Field(default=1.0, gt=0.0)
    inner_margin_mm: float = Field(default=0.0, ge=0.0)
    _camera_basis_cache: object | None = PrivateAttr(default=None)

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(R_scene_from_cam, cam_origin_scene)`` cached after first call.

        ``R_scene_from_cam`` columns are ``(right, down, forward)`` in scene
        frame, matching the OpenCV-style basis SOFA's ``InteractiveCamera``
        builds from ``position`` and ``lookAt`` (see ``poc_scene._ortho_basis``).
        """

        cached = self._camera_basis_cache
        if isinstance(cached, tuple):
            return cached
        q = self.camera_extrinsics_scene.rotation
        qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)
        norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
        if norm_sq <= 1.0e-30:
            rotation = np.eye(3, dtype=float)
        else:
            inv = 1.0 / math.sqrt(norm_sq)
            qx *= inv
            qy *= inv
            qz *= inv
            qw *= inv
            xx = qx * qx
            yy = qy * qy
            zz = qz * qz
            xy = qx * qy
            xz = qx * qz
            yz = qy * qz
            wx = qw * qx
            wy = qw * qy
            wz = qw * qz
            rotation = np.array(
                [
                    [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                    [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                    [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
                ],
                dtype=float,
            )
        origin = np.array(
            (
                float(self.camera_extrinsics_scene.position.x),
                float(self.camera_extrinsics_scene.position.y),
                float(self.camera_extrinsics_scene.position.z),
            ),
            dtype=float,
        )
        self._camera_basis_cache = (rotation, origin)
        return rotation, origin

    def signed_distance_to_envelope(self, p_scene: Vec3) -> float:
        rotation, origin = self._camera_basis()
        scene_offset = np.array(
            (float(p_scene.x) - origin[0], float(p_scene.y) - origin[1], float(p_scene.z) - origin[2]),
            dtype=float,
        )
        cam_point = rotation.T @ scene_offset
        x_cam = float(cam_point[0])
        y_cam = float(cam_point[1])
        z_cam = float(cam_point[2])
        if z_cam <= 0.0:
            return float(z_cam - float(self.near_depth_mm))

        depth_to_near = z_cam - float(self.near_depth_mm)
        depth_to_far = float(self.far_depth_mm) - z_cam

        intr = self.camera_intrinsics
        fx = float(intr.fx)
        fy = float(intr.fy)
        cx = float(intr.cx)
        cy = float(intr.cy)
        half_w_px = max(cx, float(intr.width) - cx)
        half_h_px = max(cy, float(intr.height) - cy)
        margin = float(self.lateral_margin_fraction)
        half_w_px *= (1.0 - margin)
        half_h_px *= (1.0 - margin)

        half_w_mm = z_cam * half_w_px / fx if fx > 0.0 else float("inf")
        half_h_mm = z_cam * half_h_px / fy if fy > 0.0 else float("inf")
        right_dist = half_w_mm - abs(x_cam)
        down_dist = half_h_mm - abs(y_cam)

        return float(min(depth_to_near, depth_to_far, right_dist, down_dist))


class CompositeEnvelope(WorkspaceEnvelope):
    """Workspace envelope = intersection of a list of leaf envelopes.

    The signed distance is the minimum of the children's signed distances,
    so a point is inside the composite only when it is inside every child.

    ``outer_margin_mm`` and ``inner_margin_mm`` are exposed at the composite
    level for callers (sequencer, orchestrator) that use them as gating
    thresholds; they default to the max of the leaves so the composite stays
    at least as strict as any child.
    """

    model_config = {"extra": "forbid"}

    envelopes: list["SceneGeometryEnvelope | SphereEnvelope | CameraFrustumEnvelope"] = Field(
        min_length=1,
    )
    outer_margin_mm: float = Field(default=0.0, ge=0.0)
    inner_margin_mm: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _aggregate_margins(self) -> "CompositeEnvelope":
        if self.outer_margin_mm == 0.0:
            outer = 0.0
            for env in self.envelopes:
                outer = max(outer, float(getattr(env, "outer_margin_mm", 0.0)))
            object.__setattr__(self, "outer_margin_mm", outer)
        if self.inner_margin_mm == 0.0:
            inner = 0.0
            for env in self.envelopes:
                inner = max(inner, float(getattr(env, "inner_margin_mm", 0.0)))
            object.__setattr__(self, "inner_margin_mm", inner)
        return self

    def signed_distance_to_envelope(self, p_scene: Vec3) -> float:
        distances = [
            float(env.signed_distance_to_envelope(p_scene)) for env in self.envelopes
        ]
        return float(min(distances)) if distances else 0.0


WorkspaceEnvelope = (
    SceneGeometryEnvelope | SphereEnvelope | CameraFrustumEnvelope | CompositeEnvelope
)


class OrientationBias(BaseModel):
    """Per-tool orientation bias configuration."""

    model_config = {"extra": "forbid"}

    forward_axis_local: Vec3 = Field(default_factory=lambda: Vec3(x=0.0, y=0.0, z=1.0))
    surface_normal_blend: float = Field(default=0.7, ge=0.0, le=1.0)
    gain: float = Field(default=0.0, ge=0.0)
    deadband_rad: float = Field(default=0.0, ge=0.0)
    approach_axis_scene: Vec3 | None = Field(
        default=None,
        description=(
            "Optional fixed insertion-to-tip direction in scene frame. When set, the "
            "sequencer aligns the tool's ``forward_axis_local`` with this vector for "
            "every primitive, so the shaft body stays oriented from a consistent "
            "off-screen direction across the whole episode."
        ),
    )
    approach_axis_jitter_rad: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Per-primitive jitter applied around ``approach_axis_scene`` when locked. "
            "Kept small so the shaft direction stays visually consistent."
        ),
    )


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
    tone_augmentation: VisualToneAugmentation = Field(default_factory=VisualToneAugmentation)

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

    @model_validator(mode="after")
    def _auto_build_workspace_envelope(self) -> "SceneConfig":
        """Auto-populate ``tool.workspace_envelope`` and ``orientation_bias.approach_axis_scene``.

        Without this, the sequencer is free to sample tip targets anywhere inside
        a default 45 mm-radius sphere around origin, and the orchestrator has
        nothing to block on. The workspace envelope built here intersects:

        * the inflated **tissue surface mesh AABB** (so the tip stays near the
          brain), and
        * the **camera frustum** (so the tip stays inside the captured image
          regardless of how the AABB extends past the frustum at near depths).

        The orientation bias is given a fixed ``approach_axis_scene`` derived
        from the camera view direction so the shaft body comes from a stable
        off-screen insertion direction rather than re-orienting per primitive.
        """

        existing_envelope = self.tool.workspace_envelope
        explicit_approach = self.tool.orientation_bias.approach_axis_scene
        tool_update: dict[str, Any] = {}

        if existing_envelope is None:
            new_envelope = self._build_auto_workspace_envelope()
            if new_envelope is not None:
                tool_update["workspace_envelope"] = new_envelope

        if explicit_approach is None:
            new_axis = self._derive_default_approach_axis()
            if new_axis is not None:
                tool_update["orientation_bias"] = self.tool.orientation_bias.model_copy(
                    update={"approach_axis_scene": new_axis},
                )

        if tool_update:
            self.tool = self.tool.model_copy(update=tool_update)
        return self

    def _build_auto_workspace_envelope(self) -> "WorkspaceEnvelope | None":
        envelopes: list[WorkspaceEnvelope] = []

        mesh_envelope = self._build_tissue_aabb_envelope()
        if mesh_envelope is not None:
            envelopes.append(mesh_envelope)

        frustum_envelope = self._build_camera_frustum_envelope()
        if frustum_envelope is not None:
            envelopes.append(frustum_envelope)

        if not envelopes:
            return None
        if len(envelopes) == 1:
            return envelopes[0]
        return CompositeEnvelope(envelopes=envelopes)

    def _build_tissue_aabb_envelope(self) -> "SceneGeometryEnvelope | None":
        if self.tissue_assets is None:
            return None
        try:
            from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_dejavu_root

            root = resolve_dejavu_root()
        except RuntimeError:
            return None
        surface = (root / self.tissue_assets.surface_mesh_relative_path).resolve()
        if not surface.is_file():
            return None
        return SceneGeometryEnvelope(
            surface_mesh_path=surface,
            outer_margin_mm=5.0,
            inner_margin_mm=2.0,
        )

    def _build_camera_frustum_envelope(self) -> "CameraFrustumEnvelope | None":
        intrinsics = self.camera_intrinsics
        if intrinsics.width <= 0 or intrinsics.height <= 0:
            return None
        if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
            return None
        if self._camera_extrinsics_are_identity():
            return None
        return CameraFrustumEnvelope(
            camera_extrinsics_scene=self.camera_extrinsics_scene,
            camera_intrinsics=intrinsics,
            near_depth_mm=5.0,
            far_depth_mm=300.0,
            lateral_margin_fraction=0.08,
            outer_margin_mm=1.0,
            inner_margin_mm=2.0,
        )

    def _camera_extrinsics_are_identity(self) -> bool:
        pose = self.camera_extrinsics_scene
        position = pose.position
        rotation = pose.rotation
        return (
            float(position.x) == 0.0
            and float(position.y) == 0.0
            and float(position.z) == 0.0
            and float(rotation.w) == 1.0
            and float(rotation.x) == 0.0
            and float(rotation.y) == 0.0
            and float(rotation.z) == 0.0
        )

    def _derive_default_approach_axis(self) -> Vec3 | None:
        """Default approach axis = camera view dir tilted ~30° toward camera-down.

        This puts the insertion port "above" the image (in screen-up direction)
        for typical surgical viewpoints where world-up maps to screen-up. The
        shaft body therefore extends from the tip backward toward the top of
        the image, off-screen.
        """

        if self._camera_extrinsics_are_identity():
            return None
        try:
            rotation, _ = CameraFrustumEnvelope(
                camera_extrinsics_scene=self.camera_extrinsics_scene,
                camera_intrinsics=self.camera_intrinsics,
            )._camera_basis()
        except Exception:
            return None
        right = rotation[:, 0]
        down = rotation[:, 1]
        forward = rotation[:, 2]
        del right
        tilt = math.radians(30.0)
        axis = math.cos(tilt) * forward + math.sin(tilt) * (-down)
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            return None
        axis = axis / norm
        return Vec3(x=float(axis[0]), y=float(axis[1]), z=float(axis[2]))
