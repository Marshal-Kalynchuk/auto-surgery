"""POC `.scn` viewpoint parsing."""

from __future__ import annotations

from pathlib import Path

from auto_surgery.env.sofa_scenes.poc_scene import parse_poc_scene

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parse_brain_dejavu_forceps_poc() -> None:
    path = (
        _REPO_ROOT
        / "src"
        / "auto_surgery"
        / "env"
        / "sofa_scenes"
        / "brain_dejavu_forceps_poc.scn"
    )
    vp = parse_poc_scene(path)
    assert vp.position_scene_mm == (-56.838, -22.045, 94.298)
    assert vp.lookat_scene_mm == (0.0, 0.0, 0.0)
    assert vp.viewport == (950, 700)
    assert vp.field_of_view_deg is None
