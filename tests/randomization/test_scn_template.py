from __future__ import annotations

from pathlib import Path
import math

import pytest

from auto_surgery.config import load_scene_config
from auto_surgery.randomization.scn_template import render_scene_template
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.scene import (
    DirectionalLight,
    LightingSpec,
    MeshPerturbation,
    SpotLight,
    TissueAssetSpec,
)
from auto_surgery.schemas.sensors import CameraIntrinsics


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "configs" / "scenes" / "dejavu_brain.yaml"


def _write_fake_dejavu_root(tmp_path: Path) -> Path:
    root = tmp_path / "dejavu"
    brain_data = root / "scenes" / "brain" / "data"
    brain_data.mkdir(parents=True, exist_ok=True)
    (brain_data / "volume_simplified.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (brain_data / "surface_full.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (brain_data / "texture_outpaint.png").write_text("", encoding="utf-8")
    (brain_data / "surface_skull.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (brain_data / "texture.png").write_text("", encoding="utf-8")

    liver_data = root / "scenes" / "liver" / "data" / "dv_tool"
    liver_data.mkdir(parents=True, exist_ok=True)
    (liver_data / "body_uv.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (liver_data / "instru.png").write_text("", encoding="utf-8")

    return root


def _assert_no_double_braces(rendered: Path) -> None:
    text = rendered.read_text(encoding="utf-8")
    assert "{{" not in text
    assert "}}" not in text


def test_render_template_substitutes_identity_mesh_paths(tmp_path: Path) -> None:
    root = _write_fake_dejavu_root(tmp_path)
    scene = load_scene_config(SCENE_PATH)
    rendered = render_scene_template(scene, dejavu_root=root)

    text = rendered.read_text(encoding="utf-8")
    assert f"{root}/scenes/brain/data/volume_simplified.obj" in text
    assert f"{root}/scenes/brain/data/surface_full.obj" in text
    assert "<LightManager" not in text
    assert "Sofa.GL.Component.Shader" in text
    assert "MAT Diffuse 1 " in text
    assert "MAT Diffuse Color" not in text
    _assert_no_double_braces(rendered)


def test_render_template_uses_warped_mesh_paths_for_non_identity(tmp_path: Path) -> None:
    try:
        __import__("trimesh")
    except ModuleNotFoundError:
        pytest.skip("trimesh is not installed; skipping warped template rendering test.")

    root = _write_fake_dejavu_root(tmp_path)
    scene = load_scene_config(SCENE_PATH).model_copy(
        update={"tissue_mesh_perturbation": MeshPerturbation(scale=(1.03, 1.0, 1.0))}
    )
    rendered = render_scene_template(scene, dejavu_root=root)

    text = rendered.read_text(encoding="utf-8")
    assert "auto-surgery-tissue-warp-" in text
    assert f"{root}/scenes/brain/data/volume_simplified.obj" not in text
    assert "SparseGridTopology" in text
    _assert_no_double_braces(rendered)


def test_render_template_rejects_unresolved_placeholders(tmp_path: Path) -> None:
    root = _write_fake_dejavu_root(tmp_path)
    broken = tmp_path / "broken.template"
    broken.write_text(
        "<Node>{{ dt }} {{ missing_placeholder }}</Node>",
        encoding="utf-8",
    )
    scene = load_scene_config(SCENE_PATH)

    with pytest.raises(ValueError, match="Unrendered placeholders: missing_placeholder"):
        render_scene_template(scene, dejavu_root=root, template_path=broken)


def test_render_template_replaces_dejavu_root_and_braces(tmp_path: Path) -> None:
    root = _write_fake_dejavu_root(tmp_path)
    template = tmp_path / "custom.template"
    template.write_text(
        "<Node>{{ dt }} ${DEJAVU_ROOT}/scenes/brain/data/volume_simplified.obj</Node>",
        encoding="utf-8",
    )
    scene = load_scene_config(SCENE_PATH)

    rendered = render_scene_template(scene, dejavu_root=root, template_path=template)
    text = rendered.read_text(encoding="utf-8")
    assert "${DEJAVU_ROOT}" not in text
    assert str(root) in text


def test_render_template_respects_tissue_asset_overrides(tmp_path: Path) -> None:
    root = _write_fake_dejavu_root(tmp_path)
    assets = TissueAssetSpec(
        volume_mesh_relative_path="scenes/brain/data/custom_volume.obj",
        surface_mesh_relative_path="scenes/brain/data/custom_surface.obj",
        texture_relative_path="scenes/brain/data/custom_texture.png",
    )
    custom_root = root / "scenes" / "brain" / "data"
    (custom_root / "custom_volume.obj").write_text("v 1 0 0\n", encoding="utf-8")
    (custom_root / "custom_surface.obj").write_text("v 1 0 0\n", encoding="utf-8")
    (custom_root / "custom_texture.png").write_text("", encoding="utf-8")
    scene = load_scene_config(SCENE_PATH).model_copy(update={"tissue_assets": assets})

    rendered = render_scene_template(scene, dejavu_root=root)
    text = rendered.read_text(encoding="utf-8")
    assert f"{root}/scenes/brain/data/custom_volume.obj" in text
    assert f"{root}/scenes/brain/data/custom_surface.obj" in text
    assert f"{root}/scenes/brain/data/custom_texture.png" in text


def test_render_template_includes_scaled_lighting_block(tmp_path: Path) -> None:
    root = _write_fake_dejavu_root(tmp_path)
    scene = load_scene_config(SCENE_PATH).model_copy(
        update={
            "lighting": LightingSpec(
                directional=DirectionalLight(
                    direction_scene=Vec3(x=0.0, y=-1.0, z=0.0),
                    intensity=2.0,
                    color_rgb=(0.2, 0.4, 0.6),
                ),
                spot=SpotLight(
                    position_scene=Vec3(x=0.5, y=1.0, z=1.5),
                    direction_scene=Vec3(x=0.0, y=-1.0, z=0.0),
                    cone_half_angle_deg=30.0,
                    intensity=0.5,
                    color_rgb=(0.4, 0.8, 1.2),
                ),
                background_rgb=(0.05, 0.05, 0.08),
            )
        }
    )

    rendered = render_scene_template(scene, dejavu_root=root)
    text = rendered.read_text(encoding="utf-8")
    assert "<LightManager />" in text
    assert "DirectionalLight" in text
    assert "SpotLight" in text
    assert "color=\"0.4 0.8 1.2\"" in text
    assert "color=\"0.2 0.4 0.6\"" in text


def test_render_template_camera_block_uses_orientation_and_fov_from_intrinsics(tmp_path: Path) -> None:
    root = _write_fake_dejavu_root(tmp_path)
    scene = load_scene_config(SCENE_PATH).model_copy(
        update={
            "camera_extrinsics_scene": Pose(
                position=Vec3(x=1.0, y=2.0, z=3.0),
                rotation=Quaternion(w=0.7071067811865476, x=0.7071067811865476, y=0.0, z=0.0),
            ),
            "camera_intrinsics": CameraIntrinsics(
                fx=800.0,
                fy=400.0,
                cx=13.0,
                cy=17.0,
                width=200,
                height=100,
            ),
        }
    )
    rendered = render_scene_template(scene, dejavu_root=root)
    text = rendered.read_text(encoding="utf-8")
    look_at_line = next(line for line in text.splitlines() if "OffscreenCamera" in line)
    assert 'position="1 2 3"' in look_at_line
    assert 'orientation="' in look_at_line
    assert 'lookAt' not in look_at_line
    assert 'up=' not in look_at_line
    orientation_text = look_at_line.split('orientation="', 1)[1].split('"', 1)[0]
    orientation_vals = [float(value) for value in orientation_text.split(" ")]
    assert len(orientation_vals) == 4
    assert orientation_vals[0] == pytest.approx(0.7071067811865476, abs=1e-6)
    assert orientation_vals[1] == pytest.approx(0.0, abs=1e-12)
    assert orientation_vals[2] == pytest.approx(0.0, abs=1e-12)
    assert orientation_vals[3] == pytest.approx(0.7071067811865476, abs=1e-6)
    assert 'widthViewport="200"' in look_at_line
    assert 'heightViewport="100"' in look_at_line
    expected_fov = 2.0 * math.degrees(math.atan(100.0 / (2.0 * 400.0)))
    observed_fov = float(look_at_line.split('fieldOfView="', 1)[1].split('"', 1)[0])
    assert math.isclose(observed_fov, expected_fov, rel_tol=0.0, abs_tol=1e-5)
