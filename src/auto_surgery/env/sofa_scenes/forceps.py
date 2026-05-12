"""Minimal forceps factory for SOFA scenes.

This POC focuses on coherent instrument motion and RGB capture.
For now, the forceps is implemented as a *visual* rigid object (OglModel)
whose pose is controlled from Python by updating its ``translation`` field.

The action-applier in ``src/auto_surgery/training/sofa_forceps_smoke.py`` expects:
- a SOFA node named ``Forceps``
- an OglModel named ``forcepsVisual`` (with a writable ``translation`` data field)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Reuse DejaVu tool geometry so we don't need to source new assets.


def _resolve_dejavu_root() -> Path:
    env_root = os.environ.get("DEJAVU_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser()
        if candidate.exists():
            return candidate
    for candidate in (
        Path.home() / "repos" / "neuroarm" / "DejaVu-main",
        Path.home() / "DejaVu-main",
    ):
        if candidate.exists():
            return candidate
    return Path(env_root or (Path.home() / "DejaVu-main")).expanduser()


_DEJAVU_ROOT = _resolve_dejavu_root()
DEFAULT_FORCEPS_MESH = str(_DEJAVU_ROOT / "scenes" / "liver" / "data" / "dv_tool" / "body_uv.obj")
DEFAULT_FORCEPS_TEXTURE = str(
    _DEJAVU_ROOT / "scenes" / "liver" / "data" / "dv_tool" / "instru.png"
)


def create_forceps_node(
    root_node: Any,
    *,
    node_name: str = "Forceps",
    mesh_filename: str = DEFAULT_FORCEPS_MESH,
    texture_filename: str | None = DEFAULT_FORCEPS_TEXTURE,
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

