"""SceneConfig POC baseline merge behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_surgery.schemas.scene import SceneConfig

POC_REL = Path("src/auto_surgery/env/sofa_scenes/brain_dejavu_forceps_poc.scn")


def test_scene_config_derives_camera_from_poc_only() -> None:
    scene = SceneConfig.model_validate(
        {
            "tissue_scene_path": "src/auto_surgery/env/sofa_scenes/brain_dejavu_episodic.scn.template",
            "poc_scene_path": str(POC_REL),
        },
    )
    assert scene.camera_extrinsics_scene.position.x == pytest.approx(-56.838)
    assert scene.camera_extrinsics_scene.position.y == pytest.approx(-22.045)
    assert scene.camera_extrinsics_scene.position.z == pytest.approx(94.298)
    assert scene.camera_intrinsics.width == 950
    assert scene.camera_intrinsics.height == 700
    assert scene.camera_intrinsics.cx == pytest.approx(475.0)
    assert scene.camera_intrinsics.cy == pytest.approx(350.0)


def test_explicit_yaml_overrides_poc_viewport() -> None:
    scene = SceneConfig.model_validate(
        {
            "tissue_scene_path": "src/auto_surgery/env/sofa_scenes/brain_dejavu_episodic.scn.template",
            "poc_scene_path": str(POC_REL),
            "camera_intrinsics": {"width": 111, "height": 222, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        },
    )
    assert scene.camera_intrinsics.width == 111
    assert scene.camera_intrinsics.height == 222


def test_explicit_position_overrides_poc_pose() -> None:
    scene = SceneConfig.model_validate(
        {
            "tissue_scene_path": "src/auto_surgery/env/sofa_scenes/brain_dejavu_episodic.scn.template",
            "poc_scene_path": str(POC_REL),
            "camera_extrinsics_scene": {
                "position": {"x": 99.0, "y": 0.0, "z": 0.0},
            },
        },
    )
    assert scene.camera_extrinsics_scene.position.x == 99.0
