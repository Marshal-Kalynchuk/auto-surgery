from __future__ import annotations

import hashlib
import json
from pathlib import Path
import yaml
import numpy as np

import pytest
from pydantic import ValidationError

from auto_surgery.env.sofa_scenes.forceps import ForcepsAssemblyParams
from auto_surgery.env.sofa_scenes.forceps_assets import load_dejavu_forceps_defaults
from auto_surgery.logging.writer import frames_to_table, table_to_frames
from auto_surgery.schemas.logging import LoggedFrame
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.scene import SceneConfig, ToolSpec, VisualOverrides


def _mesh_signature(path: Path) -> str:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif fields[0] == "f" and len(fields) >= 4:
            faces.append(
                (
                    int(fields[1].split("/")[0]) - 1,
                    int(fields[2].split("/")[0]) - 1,
                    int(fields[3].split("/")[0]) - 1,
                )
            )
    vertices_arr = np.asarray(vertices, dtype=np.float64).copy(order="C")
    faces_arr = np.asarray(faces, dtype=np.int64).copy(order="C")
    digest = hashlib.sha256()
    digest.update(vertices_arr.shape[0].to_bytes(8, byteorder="little", signed=False))
    digest.update(faces_arr.shape[0].to_bytes(8, byteorder="little", signed=False))
    digest.update(vertices_arr.tobytes())
    digest.update(faces_arr.tobytes())
    return digest.hexdigest()


def test_logged_frame_roundtrip_via_parquet() -> None:
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "golden_logged_frame.json"
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    frame = LoggedFrame.model_validate(raw)
    table = frames_to_table([frame])
    restored = table_to_frames(table)[0]
    assert restored.model_dump() == frame.model_dump()


def test_golden_fixture_loads() -> None:
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "golden_logged_frame.json"
    lf = LoggedFrame.model_validate_json(fixture.read_text(encoding="utf-8"))
    assert lf.frame_index == 0
    assert lf.sensor_payload.tool.in_contact in (True, False)


def test_tool_id_literal_is_enforced() -> None:
    assert ToolSpec(tool_id="dejavu_forceps").tool_id == "dejavu_forceps"
    with pytest.raises(ValueError):
        ToolSpec(tool_id="wrong")


def test_scene_initial_jaw_is_bounded() -> None:
    with pytest.raises(ValueError):
        ToolSpec(initial_jaw=-0.1)
    with pytest.raises(ValueError):
        ToolSpec(initial_jaw=1.1)
    scene = SceneConfig(tissue_scene_path=Path("legacy.scene.scn"))
    assert 0.0 <= scene.tool.initial_jaw <= 1.0


def test_scene_camera_pose_requires_valid_pose() -> None:
    with pytest.raises(ValueError):
        SceneConfig(
            tissue_scene_path=Path("legacy.scene.scn"),
            camera_extrinsics_scene=Pose(
                position=Vec3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(w=2.0, x=0.0, y=0.0, z=0.0),
            ),
        )


def test_visual_overrides_are_optional() -> None:
    scene = SceneConfig(
        tissue_scene_path=Path("legacy.scene.scn"),
        tool=ToolSpec(visual_overrides=VisualOverrides()),
    )
    assert scene.tool.visual_overrides is not None


def test_dejavu_default_forceps_contract_is_reproducible() -> None:
    contract = load_dejavu_forceps_defaults()
    mesh_path = Path(contract.collision_mesh_path)
    assert mesh_path.exists()
    assert contract.collision_mesh_sha256 == hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    defaults_path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "forceps"
        / "dejavu_default.yaml"
    )
    defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    collision_target_faces = (
        defaults.get("assembly", {}).get("extras", {}).get("collision_target_faces")
    )
    assert collision_target_faces == 500
    geometry_signature_path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "forceps"
        / "shaft_tip_collision.obj.sha256"
    )
    geometry_signature = geometry_signature_path.read_text(encoding="utf-8").strip()
    assert len(geometry_signature) == 64
    assert all(ch in "0123456789abcdef" for ch in geometry_signature)
    assert geometry_signature == _mesh_signature(mesh_path)
    assert contract.assembly.stiffness == 1000.0
    assert contract.assembly.body_color == (1.0, 0.2, 0.2, 1.0)
    assert contract.assembly.clasper_color == (1.0, 0.2, 0.2, 1.0)
    assert contract.assembly.jaw_open_angle_rad == 0.30
    assert contract.assembly.jaw_closed_angle_rad == 0.0
    assert contract.assembly.hinge_origin_local == (0.0, 0.0, 0.0)
    assert contract.assembly.hinge_axis_local == (1.0, 0.0, 0.0)
    assert contract.assembly.tool_tip_offset_local == (0.0, 0.0, 9.4)


def test_load_dejavu_forceps_contract_maps_nested_stiffness_fields(tmp_path: Path) -> None:
    contract_path = tmp_path / "nested.yaml"
    contract_path.write_text(
        """
assembly:
  constants:
    mass: 2.75
    scale: 1.35
    stiffness:
      collision_penalty: 500.0
      visual_body_rgba:
        - 0.1
        - 0.2
        - 0.3
        - 0.4
  limits:
    jaw:
      open_angle_rad: 0.6
      closed_angle_rad: 0.1
  offsets:
    hinge_origin_local:
      - 0.1
      - 0.2
      - 0.3
    hinge_axis_local:
      - 0.0
      - 1.0
      - 0.0
    tool_tip_offset_local:
      - 1.0
      - 2.0
      - 3.0
    """,
        encoding="utf-8",
    )
    contract = load_dejavu_forceps_defaults(contract_path=contract_path)
    assembly = contract.assembly

    assert assembly.mass == 2.75
    assert assembly.scale == 1.35
    assert assembly.stiffness == 500.0
    assert assembly.body_color == (0.1, 0.2, 0.3, 0.4)
    assert assembly.clasper_color == (0.1, 0.2, 0.3, 0.4)
    assert assembly.jaw_open_angle_rad == 0.6
    assert assembly.jaw_closed_angle_rad == 0.1
    assert assembly.hinge_origin_local == (0.1, 0.2, 0.3)
    assert assembly.hinge_axis_local == (0.0, 1.0, 0.0)
    assert assembly.tool_tip_offset_local == (1.0, 2.0, 3.0)


def test_load_dejavu_forceps_defaults_invalid_payload_fallbacks_to_defaults(tmp_path: Path) -> None:
    broken_contract = tmp_path / "bad.yaml"
    broken_contract.write_text(
        """
assembly:
  constants:
    mass: invalid
    scale: invalid
    visual_body_rgba: invalid
    collision_penalty: invalid
limits:
  jaw:
    open_angle_rad: "invalid"
    closed_angle_rad: invalid
offsets:
  hinge_origin_local: ["bad", "values", "here"]
""",
        encoding="utf-8",
    )
    defaults = load_dejavu_forceps_defaults(contract_path=broken_contract)
    expected = ForcepsAssemblyParams()
    assert defaults.assembly.mass == expected.mass
    assert defaults.assembly.scale == expected.scale
    assert defaults.assembly.body_color == expected.body_color
    assert defaults.assembly.clasper_color == expected.clasper_color
    assert defaults.assembly.stiffness == expected.stiffness
    assert defaults.assembly.jaw_open_angle_rad == expected.jaw_open_angle_rad
    assert defaults.assembly.hinge_axis_local == expected.hinge_axis_local
    expected_mesh_path = Path(
        Path(__file__).resolve().parents[2]
        / "assets"
        / "forceps"
        / "shaft_tip_collision.obj"
    )
    assert defaults.collision_mesh_path == str(expected_mesh_path)
    assert defaults.collision_mesh_sha256 == hashlib.sha256(
        expected_mesh_path.read_bytes()
    ).hexdigest()


def test_tissue_scene_path_is_required() -> None:
    with pytest.raises(ValidationError):
        SceneConfig()

