from __future__ import annotations

from typing import Any

from auto_surgery.env.sofa import SofaSceneFactory
from auto_surgery.env.sofa_scenes.brain_dejavu import create_brain_scene


def _stub_scene(scene: str) -> SofaSceneFactory:
    def _factory(root_node: Any, config: Any) -> None:
        raise NotImplementedError(
            f"SOFA scene {scene!r} is registered for future expansion but is not implemented."
        )

    return _factory


SCENE_REGISTRY: dict[str, SofaSceneFactory] = {
    "dejavu_brain": create_brain_scene,
    "dejavu_liver": _stub_scene("dejavu_liver"),
    "dejavu_kidney": _stub_scene("dejavu_kidney"),
    "dejavu_eye": _stub_scene("dejavu_eye"),
    "dejavu_uterus": _stub_scene("dejavu_uterus"),
    "lapgym": _stub_scene("lapgym"),
}


def resolve_scene_factory(scene_id: str) -> SofaSceneFactory:
    """Return the Python scene factory for ``scene_id``."""

    key = scene_id.strip().lower()
    factory = SCENE_REGISTRY.get(key)
    if factory is None:
        raise KeyError(f"Unknown SOFA scene id: {scene_id!r}")
    return factory
