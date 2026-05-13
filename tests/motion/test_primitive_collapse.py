from pathlib import Path
import sys
import importlib.util


def _load_primitives_module():
    module_path = Path(__file__).resolve().parents[2] / 'src' / 'auto_surgery' / 'motion' / 'primitives.py'
    module_name = "_test_motion_primitives"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


primitives = _load_primitives_module()
Reach = primitives.Reach
Hold = primitives.Hold
ContactReach = primitives.ContactReach
Pose = primitives.Pose
Quaternion = primitives.Quaternion
Vec3 = primitives.Vec3


def test_new_primitive_classes_exist_and_distinct():
    assert Reach is not Hold
    assert Reach is not ContactReach
    assert Hold is not ContactReach


def test_legacy_primitive_classes_are_not_present():
    assert not hasattr(primitives, 'Approach')
    assert not hasattr(primitives, 'Sweep')
    assert not hasattr(primitives, 'Rotate')
    assert not hasattr(primitives, 'Retract')
    assert not hasattr(primitives, 'Probe')
    assert not hasattr(primitives, 'Dwell')


def test_new_primitives_are_constructible():
    reach = Reach(
        target_pose_scene=Pose(
            position=Vec3(x=0.0, y=0.0, z=0.0),
            rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
        duration_s=1.0,
        end_on_contact=True,
        jaw_target_start=0.0,
        jaw_target_end=1.0,
    )
    hold = Hold(duration_s=0.5, jaw_target_start=0.0, jaw_target_end=0.0)
    contact_reach = ContactReach(
        direction_hint_scene=Vec3(x=0.0, y=0.0, z=1.0),
        max_search_m=0.1,
        peak_speed_m_per_s=0.05,
        duration_s=2.1,
        jaw_target_start=0.0,
        jaw_target_end=0.0,
    )

    assert isinstance(reach, Reach)
    assert isinstance(hold, Hold)
    assert isinstance(contact_reach, ContactReach)
