"""Render per-episode SOFA `.scn` files from double-braced placeholders."""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from auto_surgery.env.sofa_scenes.dejavu_paths import (
    DEJAVU_ROOT_PLACEHOLDER,
    resolve_dejavu_asset_path,
)
from auto_surgery.randomization.mesh_warp import warp_tissue_meshes
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.scene import MeshPerturbation, SceneConfig, TissueAssetSpec

_DEFAULT_SCENE_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "env"
    / "sofa_scenes"
    / "brain_dejavu_episodic.scn.template"
)
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_DEFAULT_DT = 0.01


def _format_float(value: float) -> str:
    return f"{float(value):.8g}"


def _format_vec(values: tuple[float, float, float] | list[float] | Vec3) -> str:
    if hasattr(values, "x") and hasattr(values, "y") and hasattr(values, "z"):
        x, y, z = values.x, values.y, values.z  # type: ignore[assignment]
    else:
        x, y, z = values  # type: ignore[assignment]
    return " ".join(_format_float(v) for v in (x, y, z))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _tissue_diffuse(scene: SceneConfig) -> tuple[float, float, float]:
    base_color = (1.0, 0.7, 0.7)
    overrides = scene.tool.visual_overrides
    if overrides is None or overrides.tissue_texture_tint_rgb is None:
        tint = (1.0, 1.0, 1.0)
    else:
        tint = overrides.tissue_texture_tint_rgb
    return tuple(_clamp01(base_color[i] * tint[i]) for i in range(3))


def _material_string(scene: SceneConfig) -> str:
    """OglModel expects MAT tokens with diffuse/ambient/specular as 8-bit triples."""

    def _rgb255(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
        ri = int(round(_clamp01(rgb[0]) * 255.0))
        gi = int(round(_clamp01(rgb[1]) * 255.0))
        bi = int(round(_clamp01(rgb[2]) * 255.0))
        return (max(0, min(255, ri)), max(0, min(255, gi)), max(0, min(255, bi)))

    r, g, b = _rgb255(_tissue_diffuse(scene))
    return (
        f"MAT Diffuse 1 {r} {g} {b} 1 "
        f"Ambient 1 {r} {g} {b} 1 "
        f"Specular 1 {r} {g} {b} 1 "
        f"Emissive 1 {r} {g} {b} 1 "
        "Shininess 1 45"
    )


def _normalize(v: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm > 0.0:
        return v / norm
    if fallback is None:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return np.array(fallback, dtype=float)


def _quat_to_matrix(rotation: Quaternion) -> np.ndarray:
    w, x, y, z = rotation.w, rotation.x, rotation.y, rotation.z
    return np.array(
        [
            [
                1.0 - 2.0 * y * y - 2.0 * z * z,
                2.0 * x * y - 2.0 * z * w,
                2.0 * x * z + 2.0 * y * w,
            ],
            [
                2.0 * x * y + 2.0 * z * w,
                1.0 - 2.0 * x * x - 2.0 * z * z,
                2.0 * y * z - 2.0 * x * w,
            ],
            [
                2.0 * x * z - 2.0 * y * w,
                2.0 * y * z + 2.0 * x * w,
                1.0 - 2.0 * x * x - 2.0 * y * y,
            ],
        ],
        dtype=float,
    )


def _look_at_rotation(
    pose: Pose,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    position = np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    rotation = _quat_to_matrix(pose.rotation)
    forward = rotation @ np.array([0.0, 0.0, -1.0], dtype=float)
    up = rotation @ np.array([0.0, 1.0, 0.0], dtype=float)
    up = _normalize(up, fallback=np.array([0.0, 1.0, 0.0], dtype=float))
    look_at = position + forward
    return (
        (float(position[0]), float(position[1]), float(position[2])),
        (float(look_at[0]), float(look_at[1]), float(look_at[2])),
        (float(up[0]), float(up[1]), float(up[2])),
    )


def _camera_block(scene: SceneConfig) -> str:
    position, look_at, up = _look_at_rotation(scene.camera_extrinsics_scene)
    intrinsics = scene.camera_intrinsics
    if intrinsics.fy <= 0.0 or intrinsics.fy != intrinsics.fy:
        raise ValueError("CameraIntrinsics.fy must be finite and positive.")
    fov_deg = 2.0 * math.degrees(math.atan(intrinsics.height / (2.0 * intrinsics.fy)))
    return (
        f'<OffscreenCamera position="{_format_vec(position)}" lookAt="{_format_vec(look_at)}" '
        f'up="{_format_vec(up)}" '
        f'widthViewport="{intrinsics.width}" heightViewport="{intrinsics.height}" '
        f'fieldOfView="{_format_float(fov_deg)}" />'
    )


def _lighting_block(scene: SceneConfig) -> str:
    directional = scene.lighting.directional
    spot = scene.lighting.spot
    if directional is None and spot is None:
        return ""

    light_lines: list[str] = ["    <LightManager />"]
    if directional is not None:
        direction = _normalize(
            np.array(
                [directional.direction_scene.x, directional.direction_scene.y, directional.direction_scene.z],
                dtype=float,
            )
        )
        color = np.array(directional.color_rgb, dtype=float) * directional.intensity
        light_lines.append(
            f'    <DirectionalLight '
            f'direction="{_format_vec(direction)}" '
            f'color="{_format_vec(color)}" />'
        )

    if spot is not None:
        position = _format_vec(spot.position_scene)
        direction = _normalize(
            np.array(
                [spot.direction_scene.x, spot.direction_scene.y, spot.direction_scene.z],
                dtype=float,
            )
        )
        spot_cutoff = 2.0 * float(spot.cone_half_angle_deg)
        color = np.array(spot.color_rgb, dtype=float) * spot.intensity
        light_lines.append(
            f'    <SpotLight '
            f'position="{position}" '
            f'direction="{_format_vec(direction)}" '
            f'cutoff="{_format_float(spot_cutoff)}" '
            f'color="{_format_vec(color)}" />'
        )

    return "\n".join(light_lines)


def _rigid3d_position(pose: Pose) -> str:
    """Emit a Rigid3d MechanicalObject ``position`` token (x y z qx qy qz qw)."""

    return (
        f"{_format_float(pose.position.x)} "
        f"{_format_float(pose.position.y)} "
        f"{_format_float(pose.position.z)} "
        f"{_format_float(pose.rotation.x)} "
        f"{_format_float(pose.rotation.y)} "
        f"{_format_float(pose.rotation.z)} "
        f"{_format_float(pose.rotation.w)}"
    )


def _forceps_block(scene: SceneConfig) -> str:
    """Forceps subtree backed by a Rigid3d MO + RigidMapping.

    The MechanicalObject lets the action applier drive motion via SOFA's solver
    (writing the velocity field on the MO; EulerImplicit integrates it).
    """

    initial_position = _rigid3d_position(scene.tool.initial_pose_scene)
    return (
        '\n    <Node name="Forceps">\n'
        '        <EulerImplicitSolver name="forcepsSolver" rayleighStiffness="0.05" '
        'rayleighMass="0.05"/>\n'
        '        <CGLinearSolver name="forcepsLinearSolver" iterations="25" '
        'tolerance="1e-9" threshold="1e-9"/>\n'
        '        <Node name="Shaft">\n'
        f'            <MechanicalObject template="Rigid3d" name="shaftMO" '
        f'position="{initial_position}" velocity="0 0 0 0 0 0"/>\n'
        '            <UniformMass name="shaftMass" totalMass="0.05"/>\n'
        '            <Node name="Visual">\n'
        '                <MeshOBJLoader name="forcepsLoader" '
        'filename="${DEJAVU_ROOT}/scenes/liver/data/dv_tool/body_uv.obj"/>\n'
        '                <OglModel name="forcepsVisual" src="@forcepsLoader" '
        'texturename="${DEJAVU_ROOT}/scenes/liver/data/dv_tool/instru.png" '
        'color="1 0.2 0.2 1"/>\n'
        '                <RigidMapping input="@../shaftMO" output="@."/>\n'
        '            </Node>\n'
        '        </Node>\n'
        '    </Node>\n'
    )


def _build_context(
    scene: SceneConfig,
    *,
    dejavu_root: Path,
    use_canonical_tissue_mesh_files: bool,
) -> dict[str, str]:
    resolved_root = Path(dejavu_root)

    if not resolved_root.is_dir():
        raise FileNotFoundError(f"DEJAVU root is not a directory: {resolved_root}")

    tissue_assets = scene.tissue_assets or TissueAssetSpec()
    canonical_volume = resolve_dejavu_asset_path(
        tissue_assets.volume_mesh_relative_path,
        root=resolved_root,
    )
    canonical_surface = resolve_dejavu_asset_path(
        tissue_assets.surface_mesh_relative_path,
        root=resolved_root,
    )
    canonical_texture = resolve_dejavu_asset_path(
        tissue_assets.texture_relative_path,
        root=resolved_root,
    )

    tissue_perturb = (
        MeshPerturbation()
        if use_canonical_tissue_mesh_files
        else scene.tissue_mesh_perturbation
    )
    if tissue_perturb.is_identity():
        volume_path = canonical_volume
        surface_path = canonical_surface
    else:
        volume_path, surface_path = warp_tissue_meshes(
            canonical_volume,
            canonical_surface,
            tissue_perturb,
        )

    tissue_material = scene.tissue_material
    tissue_topology = scene.tissue_topology
    camera = scene.camera_intrinsics
    lighting = scene.lighting

    tissue_color_r, tissue_color_g, tissue_color_b = _tissue_diffuse(scene)
    material_string = _material_string(scene)

    return {
        "dt": _format_float(_DEFAULT_DT),
        "gravity": "0 0 0",
        "background_rgb": _format_vec(lighting.background_rgb),
        "light_manager_block": _lighting_block(scene),
        "tissue_volume_mesh_path": str(volume_path),
        "tissue_surface_mesh_path": str(surface_path),
        "tissue_texture_path": str(canonical_texture),
        "tissue_material_string": material_string,
        "tissue_diffuse_r": _format_float(tissue_color_r),
        "tissue_diffuse_g": _format_float(tissue_color_g),
        "tissue_diffuse_b": _format_float(tissue_color_b),
        "young_modulus": _format_float(tissue_material.young_modulus_pa),
        "poisson_ratio": _format_float(tissue_material.poisson_ratio),
        "rayleigh_stiffness": _format_float(tissue_material.rayleigh_stiffness),
        "total_mass": _format_float(tissue_material.total_mass_kg),
        "sparse_grid_n": " ".join(str(v) for v in tissue_topology.sparse_grid_n),
        "camera_block": _camera_block(scene),
        "forceps_block": _forceps_block(scene),
        "viewer_width": str(camera.width),
        "viewer_height": str(camera.height),
        "viewer_distance": "70",
        "material_rgb": _format_vec((tissue_color_r, tissue_color_g, tissue_color_b)),
    }


def _render_template(raw_xml: str, context: dict[str, Any]) -> str:
    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        value = context.get(key)
        if value is None:
            return match.group(0)
        return str(value)

    rendered = _PLACEHOLDER_RE.sub(_sub, raw_xml)
    leftovers = sorted(set(match.group(1) for match in _PLACEHOLDER_RE.finditer(rendered)))
    if leftovers:
        raise ValueError(f"Unrendered placeholders: {', '.join(leftovers)}")
    return rendered


def render_scene_template(
    scene: SceneConfig,
    *,
    dejavu_root: Path,
    template_path: Path | None = None,
    use_canonical_tissue_mesh_files: bool = False,
) -> Path:
    """Render the per-episode `.scn` from a Jinja-like template. Return the rendered path."""

    template_file = Path(template_path or _DEFAULT_SCENE_TEMPLATE).expanduser().resolve()
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    raw_xml = template_file.read_text(encoding="utf-8")
    raw_xml = raw_xml.replace("${DEJAVU_ROOT}", str(Path(dejavu_root).expanduser().resolve()))
    context = _build_context(
        scene,
        dejavu_root=Path(dejavu_root),
        use_canonical_tissue_mesh_files=use_canonical_tissue_mesh_files,
    )
    rendered = _render_template(raw_xml, context)
    resolved_root = str(Path(dejavu_root).expanduser().resolve())
    rendered = rendered.replace(DEJAVU_ROOT_PLACEHOLDER, resolved_root)

    with tempfile.NamedTemporaryFile(
        suffix=".scn",
        prefix="auto-surgery-brain-dejavu-episode-",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(rendered)
        return Path(handle.name)
