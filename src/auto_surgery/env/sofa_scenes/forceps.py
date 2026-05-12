"""Minimal forceps factory for SOFA scenes.

This POC focuses on coherent instrument motion and RGB capture.
For now, the forceps is implemented as a *visual* rigid object (OglModel)
whose pose is controlled from Python by updating its ``translation`` field.

The action-applier in ``src/auto_surgery/training/sofa_forceps_smoke.py`` expects:
- a SOFA node named ``Forceps``
- an OglModel named ``forcepsVisual`` (with a writable ``translation`` data field)
"""

from __future__ import annotations

from typing import Any

from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_dejavu_asset_path

# Reuse DejaVu tool geometry so we don't need to source new assets.


def _default_forceps_mesh() -> tuple[str, str]:
    root = resolve_dejavu_asset_path("scenes/liver/data/dv_tool")
    return (
        str(root / "body_uv.obj"),
        str(root / "instru.png"),
    )


def create_forceps_node(
    root_node: Any,
    *,
    node_name: str = "Forceps",
    mesh_filename: str | None = None,
    texture_filename: str | None = None,
    initial_translation: tuple[float, float, float] = (0.0, -30.0, 40.0),
    color: str = "1 0.2 0.2 1",
) -> Any:
    """Create a minimal forceps node with a visual OglModel.

    Parameters
    - root_node: parent SOFA node to attach the forceps to
    - mesh_filename: OBJ mesh for the visual instrument
    - texture_filename: optional PNG texture for the instrument
    - initial_translation: initial pose (world translation) of the visual model
    """

    if mesh_filename is None or texture_filename is None:
        default_mesh, default_texture = _default_forceps_mesh()
        if mesh_filename is None:
            mesh_filename = default_mesh
        if texture_filename is None:
            texture_filename = default_texture

    forceps_node = root_node.addChild(node_name)

    # Collision/physics modeling is intentionally omitted in the POC.
    # We only need a visible object that we can move coherently and record.
    loader = forceps_node.addObject(
        "MeshOBJLoader",
        name="forcepsLoader",
        filename=mesh_filename,
    )

    # NOTE: `src="@forcepsLoader"` mirrors the existing XML wrapper convention.
    # SOFA link syntax uses `@objectName` to reference a sibling object's output.
    tx, ty, tz = initial_translation
    ogl_kwargs: dict[str, Any] = {
        "name": "forcepsVisual",
        "src": "@forcepsLoader",
        "translation": f"{tx} {ty} {tz}",
        "color": color,
    }
    if texture_filename:
        ogl_kwargs["texturename"] = texture_filename

    _ogl = forceps_node.addObject("OglModel", **ogl_kwargs)
    _ = loader  # loader is retained by SOFA internally; kept here for readability.

    return forceps_node

