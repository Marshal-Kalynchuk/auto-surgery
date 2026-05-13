"""Tests for piece-2 forceps action and observer helpers."""

from __future__ import annotations

from types import SimpleNamespace
import math
import pytest

from typing import Any

from auto_surgery.env.sofa_scenes import forceps as forceps_scene
from auto_surgery.env.sofa_scenes.forceps import ForcepsAssemblyParams, ForcepsMeshSet
from auto_surgery.env.sofa_tools import (
    _discover_forceps_handles,
    _read_in_contact,
    build_forceps_observer,
    build_forceps_velocity_applier,
    resolve_tool_action_applier_from_spec,
)
from auto_surgery.env import sofa_tools as sofa_tools
from auto_surgery.env.sofa_scenes.forceps_assets import (
    _clasper_visual_transform,
    _shaft_origin_twist_from_tip_twist,
    _twist_camera_to_scene,
)
from auto_surgery.schemas.scene import ToolSpec, VisualOverrides
from auto_surgery.schemas.commands import ControlMode, Pose, Quaternion, RobotCommand, Twist, Vec3


class _Data:
    """Minimal SOFA-like container exposing a mutable value field."""

    def __init__(self, value: list[float] | tuple[float, ...]) -> None:
        self.value = list(value)

    def setValue(self, value: list[float] | tuple[float, ...]) -> None:
        self.value = list(value)


class _Dof:
    def __init__(self) -> None:
        self.position = _Data([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.v = _Data([0.0, 0.0, 0.0])
        self.w = _Data([0.0, 0.0, 0.0])
        self.force = _Data([0.0, 0.0, 0.0])

    def getObject(self, name: str) -> Any | None:
        if name == "position":
            return self.position
        if name == "v":
            return self.v
        if name == "w":
            return self.w
        if name == "force":
            return self.force
        return None


class _ForcepsDof:
    def __init__(self, initial: list[float] | tuple[float, ...]) -> None:
        self.position = _Data(initial)
        self.v = _Data([0.0, 0.0, 0.0])
        self.w = _Data([0.0, 0.0, 0.0])
        self.force = _Data([0.0, 0.0, 0.0])

    def getObject(self, name: str) -> Any | None:
        if name == "v":
            return self.v
        if name == "w":
            return self.w
        if name == "force":
            return self.force
        if name == "position":
            return self.position
        return None


class _ForcepsVisual:
    def __init__(self, initial: list[float] | tuple[float, ...] = (0.0, 0.0, 0.0)) -> None:
        self.translation = _Data(initial)
        self.rotation = _Data((0.0, 0.0, 0.0, 1.0))


class _ForcepsClasper:
    def __init__(self) -> None:
        self.translation = _Data((0.0, 0.0, 0.0))
        self.rotation = _Data((0.0, 0.0, 0.0, 1.0))


class _ForcepsNode:
    def __init__(self, *, forceps_dof: Any, forceps_visual: Any, left: Any | None = None, right: Any | None = None) -> None:
        self.forcepsDof = forceps_dof
        self.forcepsBody = forceps_visual
        self.ForcepsClasperLeft = left
        self.ForcepsClasperRight = right


class _ForcepsScene:
    def __init__(self, node: Any) -> None:
        self.Forceps = node


class _FakeSofaObject:
    def __init__(self, **fields: Any) -> None:
        for key, value in fields.items():
            if key == "name":
                setattr(self, key, value)
            else:
                setattr(self, key, _coerce_sofa_value(value))

    def findData(self, name: str) -> Any | None:
        return getattr(self, name, None)


def _coerce_sofa_value(value: Any) -> Any:
    if isinstance(value, _Data):
        return value
    if isinstance(value, (list, tuple)):
        return _Data(value)
    if isinstance(value, str):
        parts = value.split()
        if parts:
            try:
                return _Data([float(part) for part in parts])
            except ValueError:
                return value
        return value
    return value


class _FakeSofaNode:
    def __init__(self, name: str = "node") -> None:
        self._children: dict[str, _FakeSofaNode] = {}
        self._objects: dict[str, _FakeSofaObject] = {}
        self._name = name

    def addChild(self, name: str) -> "_FakeSofaNode":
        child = _FakeSofaNode(name=name)
        self._children[name] = child
        setattr(self, name, child)
        return child

    def addObject(self, _object_type: str, **kwargs: Any) -> _FakeSofaObject:
        del _object_type
        name = kwargs.get("name")
        obj = _FakeSofaObject(**kwargs)
        if name is not None:
            self._objects[name] = obj
            setattr(self, name, obj)
        return obj

    def getObject(self, name: str) -> Any | None:
        return self._objects.get(name)

    def getChild(self, name: str) -> Any | None:
        return self._children.get(name)


class _FakeSofaNodeWithContacts(_FakeSofaNode):
    def __init__(self, *, scene_contacts: Any | None = None) -> None:
        super().__init__(name="scene")
        self._scene_contacts = scene_contacts

    def getObject(self, name: str) -> Any | None:
        if name == "contacts":
            return self._scene_contacts
        return super().getObject(name)


def _first_collision_model(handles: tuple[Any | None, ...]) -> Any:
    for handle in handles:
        if handle is not None:
            return handle
    raise AssertionError("Expected a shaft collision MO handle.")


def _fake_forceps_mesh_set() -> ForcepsMeshSet:
    return ForcepsMeshSet(
        shaft_obj_path="body.obj",
        shaft_collision_obj_path="shaft_collision.obj",
        clasp_left_obj_path="clasper_left.obj",
        clasp_right_obj_path="clasper_right.obj",
    )


def _forceps_mesh_set_with_uvs() -> ForcepsMeshSet:
    return ForcepsMeshSet(
        shaft_obj_path="body.obj",
        shaft_collision_obj_path="shaft_collision.obj",
        clasp_left_obj_path="clasper_left.obj",
        clasp_right_obj_path="clasper_right.obj",
        shaft_uv_path="shaft_uv.png",
        clasper_left_uv_path="clasper_left_uv.png",
        clasper_right_uv_path="clasper_right_uv.png",
    )


def test_forceps_velocity_applier_updates_dof_pose_from_twist() -> None:
    dof = _Dof()
    applier = build_forceps_velocity_applier(forceps_dof=dof)

    cmd = RobotCommand(
        timestamp_ns=1000,
        cycle_id=0,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=0.1, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
        source="test",
    )

    applier(None, cmd)
    velocity = dof.v.value
    assert velocity[:3] == pytest.approx([0.001, 0.0, 0.0])


def test_forceps_velocity_applier_discovers_handles_from_scene() -> None:
    dof = _ForcepsDof((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    visual = _ForcepsVisual((0.0, 0.0, 0.0))
    left = _ForcepsClasper()
    right = _ForcepsClasper()
    scene = _ForcepsScene(_ForcepsNode(forceps_dof=dof, forceps_visual=visual, left=left, right=right))
    applier = build_forceps_velocity_applier(forceps_dof=None)

    cmd = RobotCommand(
        timestamp_ns=1001,
        cycle_id=1,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=1.0, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
        source="test",
    )

    applier(scene, cmd)
    assert dof.v.value[:3] == [0.01, 0.0, 0.0]
    assert left.rotation.value == pytest.approx((0.3, 0.0, 0.0))
    assert right.rotation.value == pytest.approx((-0.3, 0.0, 0.0))


def test_forceps_velocity_applier_prefers_injected_camera_pose_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dof = _Dof()
    provider_pose = Pose(
        position=Vec3(x=0.0, y=0.0, z=0.0),
        rotation=Quaternion(
            w=0.7071067811865475,
            x=0.0,
            y=0.0,
            z=0.7071067811865475,
        ),
    )
    def _forbid_scene_pose_read(_scene: Any) -> Pose:
        raise AssertionError("provider should skip scene reads")

    monkeypatch.setattr(sofa_tools, "_read_camera_pose", _forbid_scene_pose_read)
    applier = build_forceps_velocity_applier(
        forceps_dof=dof,
        camera_pose_provider=lambda: provider_pose,
    )
    cmd = RobotCommand(
        timestamp_ns=1004,
        cycle_id=4,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=1.0, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
        source="test",
    )

    expected = _twist_camera_to_scene(
        Twist(
            linear=Vec3(x=0.01, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        provider_pose,
    )
    applier(None, cmd)
    assert dof.v.value[:3] == pytest.approx(expected[:3], abs=1e-12)


def test_forceps_velocity_applier_uses_scene_camera_pose_when_provider_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, int] = {"pose_reads": 0}
    scene_pose = Pose(
        position=Vec3(x=0.0, y=0.0, z=0.0),
        rotation=Quaternion(
            w=0.7071067811865475,
            x=0.0,
            y=0.0,
            z=0.7071067811865475,
        ),
    )

    def _read_pose(_scene: Any) -> Pose:
        called["pose_reads"] += 1
        return scene_pose

    monkeypatch.setattr(sofa_tools, "_read_camera_pose", _read_pose)

    dof = _Dof()
    applier = build_forceps_velocity_applier(forceps_dof=dof)
    cmd = RobotCommand(
        timestamp_ns=1005,
        cycle_id=5,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=1.0, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
        source="test",
    )
    expected = _twist_camera_to_scene(
        Twist(
            linear=Vec3(x=0.01, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        scene_pose,
    )

    applier(None, cmd)
    assert dof.v.value[:3] == pytest.approx(expected[:3], abs=1e-12)
    assert called["pose_reads"] == 1


def test_forceps_velocity_applier_refreshes_cache_across_scene_reset_and_tracks_jaw_seed() -> None:
    jaw_ref = {"jaw": 0.2}
    applier = build_forceps_velocity_applier(forceps_dof=None, jaw_ref=jaw_ref)

    first_dof = _ForcepsDof((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    first_left = _ForcepsClasper()
    first_right = _ForcepsClasper()
    first_scene = _ForcepsScene(
        _ForcepsNode(
            forceps_dof=first_dof,
            forceps_visual=_ForcepsVisual((0.0, 0.0, 0.0)),
            left=first_left,
            right=first_right,
        )
    )
    reset_cmd = RobotCommand(
        timestamp_ns=2_001,
        cycle_id=101,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=1.0, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
        source="test",
    )

    applier(first_scene, reset_cmd)
    first_left_transform = _clasper_visual_transform(jaw_ref["jaw"], side="left")
    first_right_transform = _clasper_visual_transform(jaw_ref["jaw"], side="right")
    assert first_dof.v.value[:3] == pytest.approx([0.01, 0.0, 0.0])
    assert first_left.rotation.value == pytest.approx(list(first_left_transform.euler_xyz))
    assert first_right.rotation.value == pytest.approx(list(first_right_transform.euler_xyz))

    jaw_ref["jaw"] = 0.6
    second_dof = _ForcepsDof((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    second_left = _ForcepsClasper()
    second_right = _ForcepsClasper()
    second_scene = _ForcepsScene(
        _ForcepsNode(
            forceps_dof=second_dof,
            forceps_visual=_ForcepsVisual((0.0, 0.0, 0.0)),
            left=second_left,
            right=second_right,
        )
    )
    applier(second_scene, reset_cmd)

    second_left_transform = _clasper_visual_transform(jaw_ref["jaw"], side="left")
    second_right_transform = _clasper_visual_transform(jaw_ref["jaw"], side="right")
    assert second_dof.v.value[:3] == pytest.approx([0.01, 0.0, 0.0])
    assert second_left.rotation.value == pytest.approx(list(second_left_transform.euler_xyz))
    assert second_right.rotation.value == pytest.approx(list(second_right_transform.euler_xyz))
    assert first_left.rotation.value == pytest.approx(list(first_left_transform.euler_xyz))

    jaw_update_cmd = reset_cmd.model_copy(update={"tool_jaw_target": 0.77})
    applier(second_scene, jaw_update_cmd)
    updated_left_transform = _clasper_visual_transform(0.77, side="left")
    updated_right_transform = _clasper_visual_transform(0.77, side="right")
    assert second_left.rotation.value == pytest.approx(list(updated_left_transform.euler_xyz))
    assert second_right.rotation.value == pytest.approx(list(updated_right_transform.euler_xyz))
    assert jaw_ref["jaw"] == pytest.approx(0.77)


def test_forceps_velocity_applier_and_observer_use_factory_handles() -> None:
    root = _FakeSofaNode()
    handles = forceps_scene.create_forceps_node(root, mesh_set=_fake_forceps_mesh_set())
    dof = handles.shaft_mo
    left = handles.clasper_left_visual
    right = handles.clasper_right_visual
    assert left is not None
    assert right is not None
    assert handles.shaft_collision_mos

    dof.position.value = [0.7, 0.8, 0.9, 0.0, 0.0, 0.0, 1.0]
    dof.velocity = _Data([0.11, -0.22, 0.33, 0.01, 0.02, -0.03])
    dof.force = _Data([2.0, 3.0, -4.0])
    shaft_collision = handles.shaft_collision_mos[0]
    assert shaft_collision is not None
    shaft_collision.contacts = _Data([("shaft", "contact")])

    for clasp in (left, right):
        clasp.translation = _Data((0.0, 0.0, 0.0))
        clasp.rotation = _Data((0.0, 0.0, 0.0, 1.0))

    applier = build_forceps_velocity_applier(forceps_dof=dof)
    cmd = RobotCommand(
        timestamp_ns=1003,
        cycle_id=3,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=0.0, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=False,
        tool_jaw_target=0.4,
        source="test",
    )
    applier(root, cmd)

    assert dof.velocity.value[:3] == pytest.approx([0.0, 0.0, 0.0])

    left_transform = _clasper_visual_transform(0.4, side="left")
    right_transform = _clasper_visual_transform(0.4, side="right")
    assert left.rotation.value == pytest.approx(list(left_transform.euler_xyz))
    assert right.rotation.value == pytest.approx(list(right_transform.euler_xyz))

    dof.velocity = _Data([0.11, -0.22, 0.33, 0.01, 0.02, -0.03])
    observer = build_forceps_observer(dof=dof)
    state = observer(root)
    assert state.pose.position == Vec3(x=0.7, y=0.8, z=0.9)
    assert state.twist.linear == Vec3(x=0.11, y=-0.22, z=0.33)
    assert state.twist.angular == Vec3(x=0.01, y=0.02, z=-0.03)
    assert state.wrench == Vec3(x=2.0, y=3.0, z=-4.0)
    assert state.in_contact is True


def test_discover_forceps_handles_supports_create_forceps_node_shape_and_names() -> None:
    root = _FakeSofaNode()
    handles = forceps_scene.create_forceps_node(root, mesh_set=_fake_forceps_mesh_set())
    discovered = _discover_forceps_handles(root)

    assert discovered.forceps_node is handles.forceps_node
    assert discovered.shaft_node is handles.forceps_node.Shaft
    assert discovered.dof is handles.shaft_mo
    assert discovered.visual is handles.shaft_visual
    assert discovered.left_clasper is handles.clasper_left_visual
    assert discovered.right_clasper is handles.clasper_right_visual
    assert discovered.shaft_collision_mos == handles.shaft_collision_mos


def test_read_in_contact_only_tracks_shaft_collision_contacts() -> None:
    empty_collision = _FakeSofaObject()
    shaft_collision = _FakeSofaObject(contacts=[("shaft", "contact")])
    empty_collision.contacts = _Data([])

    assert _read_in_contact((shaft_collision,)) is True
    assert _read_in_contact((empty_collision,)) is False
    assert _read_in_contact(()) is False
    assert _read_in_contact((None,)) is False


def test_forceps_observer_in_contact_filters_non_shaft_collision_signals() -> None:
    scene = _FakeSofaNodeWithContacts(scene_contacts=_Data([("scene", "contact")]))
    handles = forceps_scene.create_forceps_node(scene, mesh_set=_fake_forceps_mesh_set())
    shaft_collision = None
    for candidate in handles.shaft_collision_mos:
        if candidate is not None:
            shaft_collision = candidate
            break
    assert shaft_collision is not None

    observer = build_forceps_observer(dof=handles.shaft_mo)
    state = observer(scene)
    assert state.in_contact is False

    shaft_collision.contacts = _Data([("shaft", "contact")])
    state = observer(scene)
    assert state.in_contact is True


def test_forceps_observer_refreshes_scene_cache_on_root_switch() -> None:
    jaw_ref = {"jaw": 0.0}
    observer = build_forceps_observer(dof=None, jaw_ref=jaw_ref)

    first_dof = _ForcepsDof((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))
    first_scene = _ForcepsScene(
        _ForcepsNode(
            forceps_dof=first_dof,
            forceps_visual=_ForcepsVisual((0.0, 0.0, 0.0)),
            left=_ForcepsClasper(),
            right=_ForcepsClasper(),
        )
    )

    first_state = observer(first_scene)
    assert first_state.pose.position == Vec3(x=1.0, y=2.0, z=3.0)

    jaw_ref["jaw"] = 0.55
    second_dof = _ForcepsDof((4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0))
    second_scene = _ForcepsScene(
        _ForcepsNode(
            forceps_dof=second_dof,
            forceps_visual=_ForcepsVisual((0.0, 0.0, 0.0)),
            left=_ForcepsClasper(),
            right=_ForcepsClasper(),
        )
    )

    second_state = observer(second_scene)
    assert second_state.pose.position == Vec3(x=4.0, y=5.0, z=6.0)
    assert second_state.jaw == pytest.approx(0.55)


@pytest.mark.parametrize(
    "shaft_contacts_present,non_shaft_contacts_present,wrench,expected",
    [
        (False, False, (0.0, 0.0, 0.0), False),
        (False, False, (1.0, 0.0, 0.0), False),
        (False, True, (0.0, 0.0, 0.0), False),
        (False, True, (1.0, 0.0, 0.0), False),
        (True, False, (0.0, 0.0, 0.0), True),
        (True, False, (1.0, 0.0, 0.0), True),
        (True, True, (0.0, 0.0, 0.0), True),
        (True, True, (1.0, 0.0, 0.0), True),
    ],
)
def test_forceps_observer_contact_truth_table(
    shaft_contacts_present: bool,
    non_shaft_contacts_present: bool,
    wrench: tuple[float, float, float],
    expected: bool,
) -> None:
    scene_contacts = _Data([("scene", "contact")]) if non_shaft_contacts_present else None
    scene = _FakeSofaNodeWithContacts(scene_contacts=scene_contacts)
    handles = forceps_scene.create_forceps_node(scene, mesh_set=_fake_forceps_mesh_set())
    shaft_collision = _first_collision_model(handles.shaft_collision_mos)

    shaft_collision.contacts = (
        _Data([("shaft", "contact")]) if shaft_contacts_present else _Data([])
    )
    if handles.shaft_mo is not None and hasattr(handles.shaft_mo, "force"):
        handles.shaft_mo.force.setValue(list(wrench))

    observer = build_forceps_observer(dof=handles.shaft_mo)
    state = observer(scene)
    assert state.in_contact is expected


def test_forceps_velocity_applier_rejects_joint_position_commands() -> None:
    dof = _Dof()
    dof.v.setValue([0.123, 0.0, -0.4])
    applier = build_forceps_velocity_applier(forceps_dof=dof)
    cmd = RobotCommand(
        timestamp_ns=1002,
        cycle_id=2,
        control_mode=ControlMode.JOINT_POSITION,
        cartesian_twist=None,
        joint_positions={"jaw": 0.1},
        enable=True,
        source="test",
    )

    applier(None, cmd)
    assert dof.v.value[:3] == [0.0, 0.0, 0.0]


def test_forceps_observer_reports_jaw_from_ref() -> None:
    dof = _Dof()
    observed = build_forceps_observer(dof=dof, jaw_ref={"jaw": 0.75})

    state = observed(None)
    assert state.jaw == 0.75
    assert state.pose.position.x == dof.position.value[0]


def test_forceps_observer_reports_wrench_and_keeps_contact_conservative() -> None:
    dof = _Dof()
    dof.v.setValue([0.05, -0.1, 0.2])
    dof.w.setValue([0.0, 0.0, 0.0])
    dof.force.setValue([1.0, 0.0, -2.0])
    observed = build_forceps_observer(dof=dof)

    state = observed(None)
    assert state.wrench.x == pytest.approx(1.0)
    assert state.wrench.y == pytest.approx(0.0)
    assert state.wrench.z == pytest.approx(-2.0)
    assert state.twist.linear.x == pytest.approx(0.05)
    assert state.twist.linear.y == pytest.approx(-0.1)
    assert state.twist.linear.z == pytest.approx(0.2)
    assert state.in_contact is False


def test_camera_to_scene_linear_twist_transform_applies_rotation() -> None:
    pose = Pose(
        position=Vec3(x=0.0, y=0.0, z=0.0),
        rotation=Quaternion(
            w=0.7071067811865475,
            x=0.0,
            y=0.0,
            z=0.7071067811865475,
        ),
    )
    transformed = _twist_camera_to_scene(
        Twist(linear=Vec3(x=1.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
        pose,
    )
    assert transformed[0] == pytest.approx(0.0)
    assert transformed[1] == pytest.approx(1.0)
    assert transformed[2] == pytest.approx(0.0)


def test_shaft_origin_twist_from_tip_shift_uses_omega_cross_r() -> None:
    shaft_pose = Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    twist_tip_scene = (0.0, 0.0, 0.02, 0.0, 0.0, 1.0)
    tip_offset_local = (0.0, 0.1, 0.0)
    shifted = _shaft_origin_twist_from_tip_twist(twist_tip_scene, shaft_pose, tip_offset_local)

    expected_linear = (
        twist_tip_scene[0]
        - (twist_tip_scene[4] * tip_offset_local[2] - twist_tip_scene[5] * tip_offset_local[1]),
        twist_tip_scene[1]
        - (twist_tip_scene[5] * tip_offset_local[0] - twist_tip_scene[3] * tip_offset_local[2]),
        twist_tip_scene[2]
        - (twist_tip_scene[3] * tip_offset_local[1] - twist_tip_scene[4] * tip_offset_local[0]),
    )

    assert shifted[0] == pytest.approx(expected_linear[0], abs=1e-12)
    assert shifted[1] == pytest.approx(expected_linear[1], abs=1e-12)
    assert shifted[2] == pytest.approx(expected_linear[2], abs=1e-12)
    assert shifted[3] == twist_tip_scene[3]
    assert shifted[4] == twist_tip_scene[4]
    assert shifted[5] == twist_tip_scene[5]


def test_create_forceps_node_prefers_loaded_contract_defaults_over_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeObject:
        def __init__(self) -> None:
            self.position = _Data([0.0, 0.0, 0.0, 1.0])

    class _FakeNode:
        def addChild(self, _name: str) -> "_FakeNode":
            return _FakeNode()

        def addObject(self, *args: Any, **kwargs: Any) -> Any:
            return _FakeObject()

    forced_defaults = ForcepsAssemblyParams(
        mass=12.5,
        scale=2.25,
        body_color=(0.4, 0.2, 0.1, 1.0),
    )
    monkeypatch.setattr(
        forceps_scene,
        "load_dejavu_forceps_defaults",
        lambda: SimpleNamespace(
            assembly=forced_defaults,
            collision_mesh_path="collision.obj",
        ),
    )
    mesh_set = ForcepsMeshSet(
        shaft_obj_path="body.obj",
        shaft_collision_obj_path="collision.obj",
        clasp_left_obj_path="clasper_left.obj",
        clasp_right_obj_path="clasper_right.obj",
    )
    handles = forceps_scene.create_forceps_node(_FakeNode(), mesh_set=mesh_set)
    assert handles.assembly.mass == forced_defaults.mass
    assert handles.assembly.scale == forced_defaults.scale
    handles = forceps_scene.create_forceps_node(_FakeNode(), mesh_set=mesh_set, params=ForcepsAssemblyParams(mass=4.2))
    assert handles.assembly.mass == 4.2


def test_clasper_visual_transform_uses_hinge_geometry() -> None:
    hinge_origin = (0.0, 0.0, 0.6)
    jaw = 0.5
    jaw_open = 0.8
    jaw_closed = 0.2
    angle = jaw_open + (jaw_closed - jaw_open) * jaw

    left = _clasper_visual_transform(
        jaw,
        side="left",
        jaw_open_angle_rad=jaw_open,
        jaw_closed_angle_rad=jaw_closed,
        hinge_origin_local=hinge_origin,
        hinge_axis_local=(1.0, 0.0, 0.0),
    )
    right = _clasper_visual_transform(
        jaw,
        side="right",
        jaw_open_angle_rad=jaw_open,
        jaw_closed_angle_rad=jaw_closed,
        hinge_origin_local=hinge_origin,
        hinge_axis_local=(1.0, 0.0, 0.0),
    )

    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    expected_left_translation = (
        0.0,
        sin_angle * hinge_origin[2],
        hinge_origin[2] * (1.0 - cos_angle),
    )
    expected_right_translation = (
        0.0,
        -sin_angle * hinge_origin[2],
        hinge_origin[2] * (1.0 - cos_angle),
    )

    assert left.translation == pytest.approx(expected_left_translation)
    assert right.translation == pytest.approx(expected_right_translation)
    assert left.euler_xyz[0] == pytest.approx(angle)
    assert right.euler_xyz[0] == pytest.approx(-angle)


def test_resolve_tool_action_applier_from_spec_prefills_dejavu_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_builder(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return lambda _scene, _action: None

    monkeypatch.setattr(
        sofa_tools,
        "load_dejavu_forceps_defaults",
        lambda: SimpleNamespace(
            assembly=ForcepsAssemblyParams(
                jaw_open_angle_rad=0.55,
                jaw_closed_angle_rad=0.15,
                hinge_origin_local=(0.1, 0.2, 0.3),
                hinge_axis_local=(0.0, 1.0, 0.0),
                tool_tip_offset_local=(4.0, 5.0, 6.0),
            )
        ),
    )
    monkeypatch.setattr(sofa_tools, "build_forceps_velocity_applier", _fake_builder)
    expected_camera_pose = Pose(
        position=Vec3(x=1.0, y=2.0, z=3.0),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    provider = lambda: expected_camera_pose

    applier = resolve_tool_action_applier_from_spec(
        ToolSpec(tool_id="dejavu_forceps"),
        camera_pose_provider=provider,
    )
    assert callable(applier)
    assert captured["jaw_open_angle_rad"] == 0.55
    assert captured["jaw_closed_angle_rad"] == 0.15
    assert captured["hinge_origin_local"] == (0.1, 0.2, 0.3)
    assert captured["hinge_axis_local"] == (0.0, 1.0, 0.0)
    assert captured["tool_tip_offset_local"] == (4.0, 5.0, 6.0)
    assert captured["camera_pose_provider"] is provider


@pytest.mark.parametrize(
    ("body_override", "clasper_override", "expected_body", "expected_clasper"),
    [
        (None, None, (1.0, 0.2, 0.2, 1.0), (1.0, 0.2, 0.2, 1.0)),
        ((0.1, 0.2, 0.3, 0.4), None, (0.1, 0.2, 0.3, 0.4), (1.0, 0.2, 0.2, 1.0)),
        (None, (0.9, 0.8, 0.7, 1.0), (1.0, 0.2, 0.2, 1.0), (0.9, 0.8, 0.7, 1.0)),
        ((0.1, 0.2, 0.3, 0.4), (0.9, 0.8, 0.7, 1.0), (0.1, 0.2, 0.3, 0.4), (0.9, 0.8, 0.7, 1.0)),
    ],
)
def test_apply_visual_overrides_color_precedence(
    body_override: tuple[float, float, float, float] | None,
    clasper_override: tuple[float, float, float, float] | None,
    expected_body: tuple[float, float, float, float],
    expected_clasper: tuple[float, float, float, float],
) -> None:
    base_meshes = _forceps_mesh_set_with_uvs()
    base_params = ForcepsAssemblyParams()
    overrides = VisualOverrides(body_color=body_override, clasper_color=clasper_override)

    meshes, params = forceps_scene._apply_visual_overrides(
        default_meshes=base_meshes,
        overrides=overrides,
        defaults=base_params,
    )

    assert meshes == base_meshes
    assert params.body_color == expected_body
    assert params.clasper_color == expected_clasper


@pytest.mark.parametrize(
    ("body_uv", "left_uv", "right_uv", "expected_body_uv", "expected_left_uv", "expected_right_uv"),
    [
        (None, None, None, "shaft_uv.png", "clasper_left_uv.png", "clasper_right_uv.png"),
        ("body_uv_override.png", None, None, "body_uv_override.png", "clasper_left_uv.png", "clasper_right_uv.png"),
        (None, "left_uv_override.png", None, "shaft_uv.png", "left_uv_override.png", "clasper_right_uv.png"),
        (None, None, "right_uv_override.png", "shaft_uv.png", "clasper_left_uv.png", "right_uv_override.png"),
        ("body_uv_override.png", "left_uv_override.png", None, "body_uv_override.png", "left_uv_override.png", "clasper_right_uv.png"),
        (None, "left_uv_override.png", "right_uv_override.png", "shaft_uv.png", "left_uv_override.png", "right_uv_override.png"),
    ],
)
def test_apply_visual_overrides_uv_precedence_and_fallback(
    body_uv: str | None,
    left_uv: str | None,
    right_uv: str | None,
    expected_body_uv: str,
    expected_left_uv: str,
    expected_right_uv: str,
) -> None:
    base_meshes = _forceps_mesh_set_with_uvs()
    base_params = ForcepsAssemblyParams()
    overrides = VisualOverrides(
        body_uv_path=body_uv,
        clasper_left_uv_path=left_uv,
        clasper_right_uv_path=right_uv,
    )

    meshes, _ = forceps_scene._apply_visual_overrides(
        default_meshes=base_meshes,
        overrides=overrides,
        defaults=base_params,
    )

    assert meshes.shaft_uv_path == expected_body_uv
    assert meshes.clasper_left_uv_path == expected_left_uv
    assert meshes.clasper_right_uv_path == expected_right_uv


def test_apply_visual_overrides_tissue_tint_is_not_applied_to_forceps() -> None:
    base_meshes = _forceps_mesh_set_with_uvs()
    base_params = ForcepsAssemblyParams()
    overrides = VisualOverrides(
        tissue_texture_tint_rgb=(0.15, 0.25, 0.35),
        body_color=(0.1, 0.2, 0.3, 0.4),
        clasper_color=(0.9, 0.8, 0.7, 1.0),
    )

    meshes, params = forceps_scene._apply_visual_overrides(
        default_meshes=base_meshes,
        overrides=overrides,
        defaults=base_params,
    )

    assert meshes == base_meshes
    assert params.body_color == (0.1, 0.2, 0.3, 0.4)
    assert params.clasper_color == (0.9, 0.8, 0.7, 1.0)
