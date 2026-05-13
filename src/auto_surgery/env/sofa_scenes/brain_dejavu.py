"""DejaVu brain scene wrapper with a forceps overlay.

This file provides a Python scene factory intended to be used with
SofaPython3-based scene loading (per the repo's staging plan).

Current POC smoke uses an XML wrapper scene (`brain_dejavu_forceps_poc.scn`)
for simplicity/headless rendering; however, this Python factory is needed for
the planned registry/factory-based scene selection layer.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_dejavu_root
from auto_surgery.env.sofa_scenes.forceps import create_forceps_node
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.scene import VisualOverrides, _identity_pose


@dataclass(frozen=True)
class BrainDejavuConfig:
    """Configuration for the wrapped DejaVu brain scene."""

    dejavu_root: Path = field(default_factory=resolve_dejavu_root)

    # Make the result usable for offscreen screenshot capture.
    enable_rendering: bool = True


def _load_dejavu_brain_create_scene(brain_py_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_dejavu_brain_scene", str(brain_py_path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Failed to create module spec for: {brain_py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    create_scene = getattr(module, "createScene", None)
    if not callable(create_scene):
        raise RuntimeError(f"DejaVu brain scene file missing createScene(): {brain_py_path}")
    return create_scene


def _maybe_set_default_pipeline_draw(root_node: Any, *, draw: int) -> None:
    """Best-effort: set DefaultPipeline.draw so frames are renderable."""

    try:
        pipeline = root_node.getObject("DefaultPipeline")
    except Exception:
        pipeline = None

    if pipeline is None:
        return

    data = getattr(pipeline, "draw", None)
    if data is None:
        return

    # SOFA DataField patterns vary slightly between builds; handle common cases.
    try:
        if hasattr(data, "value"):
            data.value = str(draw)
        else:
            pipeline.draw = str(draw)
    except Exception:
        # Rendering is a POC concern; we don't want the scene factory to crash.
        return


def create_brain_scene(
    root_node: Any,
    config: BrainDejavuConfig | dict[str, Any] | EnvConfig | None = None,
) -> None:
    """Python scene factory: add forceps and rendering defaults on top of DejaVu brain."""

    # The SOFA adapter passes the repo's `EnvConfig` into scene factories during
    # `SofaEnvironment.reset()`. For Phase-1 we treat unknown config objects as
    # "use defaults" so factories remain usable without tight coupling.
    if config is None:
        cfg = BrainDejavuConfig()
        tool_pose = _identity_pose()
        visual_overrides: VisualOverrides | None = None
    elif isinstance(config, BrainDejavuConfig):
        cfg = config
        tool_pose = _identity_pose()
        visual_overrides = None
    elif isinstance(config, EnvConfig):
        cfg = BrainDejavuConfig()
        tool_pose = config.scene.tool.initial_pose_scene
        visual_overrides = config.scene.tool.visual_overrides
    elif isinstance(config, dict):
        cfg = BrainDejavuConfig(**config)
        tool_pose = _identity_pose()
        visual_overrides = None
    else:
        cfg = BrainDejavuConfig()
        tool_pose = _identity_pose()
        visual_overrides = None

    brain_py = (cfg.dejavu_root / "scenes/brain/brain.py").resolve()
    if not brain_py.exists():
        raise FileNotFoundError(f"Expected DejaVu brain scene at: {brain_py}")

    # DejaVu brain.py uses relative `data/...` paths; ensure cwd matches that directory.
    create_scene_fn = _load_dejavu_brain_create_scene(brain_py)
    brain_dir = brain_py.parent

    cwd = os.getcwd()
    try:
        os.chdir(str(brain_dir))
        create_scene_fn(root_node)
    finally:
        os.chdir(cwd)

    create_forceps_node(
        root_node,
        pose=tool_pose,
        visual_overrides=visual_overrides,
    )

    if cfg.enable_rendering:
        # DejaVu sets draw=0; switch it back on for rendering.
        _maybe_set_default_pipeline_draw(root_node, draw=1)
        with contextlib.suppress(Exception):
            root_node.animate = True

