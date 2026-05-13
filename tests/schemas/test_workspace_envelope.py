from __future__ import annotations

from auto_surgery.schemas.commands import Vec3
from auto_surgery.schemas.scene import (
    OrientationBias,
    SceneGeometryEnvelope,
    SphereEnvelope,
    ToolSpec,
    WorkspaceEnvelope,
)


def test_workspace_envelope_signed_distance_to_envelope_is_callable() -> None:
    scene_envelope = SceneGeometryEnvelope(
        outer_margin_m=0.03,
        inner_margin_m=0.01,
        no_go_regions=[],
    )
    sphere_envelope = SphereEnvelope(
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        radius_m=0.5,
        outer_margin_m=0.03,
        inner_margin_m=0.01,
    )

    assert scene_envelope.signed_distance_to_envelope(Vec3(x=0.2, y=0.0, z=0.0)) == 0.0
    assert (
        sphere_envelope.signed_distance_to_envelope(Vec3(x=0.5, y=0.0, z=0.0))
        == 0.0
    )


def test_tool_spec_accepts_workspace_envelopes() -> None:
    scene_envelope: WorkspaceEnvelope = SceneGeometryEnvelope(
        outer_margin_m=0.03,
        inner_margin_m=0.01,
        no_go_regions=[],
    )
    sphere_envelope: WorkspaceEnvelope = SphereEnvelope(
        center_scene=Vec3(x=0.0, y=0.0, z=0.0),
        radius_m=0.2,
        outer_margin_m=0.03,
        inner_margin_m=0.01,
    )

    tool_with_scene = ToolSpec(workspace_envelope=scene_envelope)
    tool_with_sphere = ToolSpec(workspace_envelope=sphere_envelope)

    assert isinstance(tool_with_scene.workspace_envelope, SceneGeometryEnvelope)
    assert isinstance(tool_with_sphere.workspace_envelope, SphereEnvelope)
    assert tool_with_scene.orientation_bias.forward_axis_local == Vec3(x=0.0, y=0.0, z=1.0)
    assert tool_with_scene.orientation_bias.surface_normal_blend == 0.7
    assert tool_with_sphere.orientation_bias.deadband_rad >= 0.0
    assert OrientationBias().gain == 0.0
