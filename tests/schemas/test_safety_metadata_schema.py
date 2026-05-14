from __future__ import annotations

from auto_surgery.schemas.commands import (
    RobotCommand,
    SafetyMetadata,
    Twist,
    Vec3,
)


def _minimal_twist() -> Twist:
    return Twist(
        linear=Vec3(x=0.0, y=0.0, z=0.0),
        angular=Vec3(x=0.0, y=0.0, z=0.0),
    )


def test_safety_metadata_model_accepts_all_fields() -> None:
    metadata = SafetyMetadata(
        clamped_linear=True,
        clamped_angular=False,
        biased_linear=True,
        biased_angular=False,
        scaled_by=0.75,
        signed_distance_to_envelope_mm=0.12,
        signed_distance_to_surface_mm=-0.04,
    )

    assert metadata.model_dump() == {
        "clamped_linear": True,
        "clamped_angular": False,
        "biased_linear": True,
        "biased_angular": False,
        "scaled_by": 0.75,
        "signed_distance_to_envelope_mm": 0.12,
        "signed_distance_to_surface_mm": -0.04,
        "pose_error_norm_mm": None,
        "pose_error_norm_rad": None,
    }


def test_robot_command_without_safety_defaults_to_none() -> None:
    command = RobotCommand(
        timestamp_ns=1_234_567_890,
        cycle_id=0,
        cartesian_twist=_minimal_twist(),
    )

    assert command.safety is None
    assert command.model_dump()["safety"] is None
    RobotCommand.model_validate_json(command.model_dump_json())


def test_robot_command_with_safety_serializes() -> None:
    metadata = SafetyMetadata(
        clamped_linear=True,
        clamped_angular=False,
        biased_linear=True,
        biased_angular=True,
        scaled_by=None,
        signed_distance_to_envelope_mm=0.25,
        signed_distance_to_surface_mm=0.18,
        pose_error_norm_mm=0.5,
        pose_error_norm_rad=0.01,
    )
    command = RobotCommand(
        timestamp_ns=987_654_321,
        cycle_id=1,
        cartesian_twist=_minimal_twist(),
        safety=metadata,
    )

    payload = command.model_dump()
    assert payload["safety"] == metadata.model_dump()

    restored = RobotCommand.model_validate_json(command.model_dump_json())
    assert isinstance(restored.safety, SafetyMetadata)
    assert restored.safety == metadata
