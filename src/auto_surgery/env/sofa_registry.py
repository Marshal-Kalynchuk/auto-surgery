from __future__ import annotations

from typing import Any

from auto_surgery.env.sofa_backend import SofaSceneFactory
from auto_surgery.env.sofa_scenes.brain_dejavu import create_brain_scene


def _stub_scene(scene: str) -> SofaSceneFactory:
    def _factory(root_node: Any, config: Any) -> None:
        raise NotImplementedError(
            f"SOFA scene {scene!r} is planned and currently not implemented."
        )

    return _factory


IMPLEMENTED_SCENE_REGISTRY: dict[str, SofaSceneFactory] = {
    "dejavu_brain": create_brain_scene,
}

PLANNED_SCENE_REGISTRY: dict[str, SofaSceneFactory] = {
    "dejavu_liver": _stub_scene("dejavu_liver"),
    "dejavu_kidney": _stub_scene("dejavu_kidney"),
    "dejavu_eye": _stub_scene("dejavu_eye"),
    "dejavu_uterus": _stub_scene("dejavu_uterus"),
    "lapgym": _stub_scene("lapgym"),
}

SCENE_REGISTRY = {**IMPLEMENTED_SCENE_REGISTRY, **PLANNED_SCENE_REGISTRY}


def resolve_scene_factory(scene_id: str) -> SofaSceneFactory:
    """Return the Python scene factory for ``scene_id``."""

    key = scene_id.strip().lower()
    factory = SCENE_REGISTRY.get(key)
    if factory is None:
        raise KeyError(f"Unknown SOFA scene id: {scene_id!r}")
    return factory
