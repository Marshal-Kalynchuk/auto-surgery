from __future__ import annotations

from pathlib import Path

import pytest

from auto_surgery.config import load_scene_config
from auto_surgery.randomization.scn_template import render_scene_template
from auto_surgery.schemas.scene import MeshPerturbation, TissueAssetSpec


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
