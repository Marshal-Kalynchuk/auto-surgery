# SOFA scene & tool registry extension

This POC wires two small registries:

- `src/auto_surgery/env/sofa_registry.py` — scene factories (`dejavu_brain` enabled; other organs stubbed).
- `src/auto_surgery/env/sofa_tools.py` — instrument action appliers (`forceps` enabled; scissors/scalpel/needle stubbed).

## Adding a new DejaVu organ scene

1. Implement a `SofaSceneFactory` with signature `(root_node, EnvConfig) -> None` (see `create_brain_scene` in `sofa_scenes/brain_dejavu.py`).
2. Register it in `SCENE_REGISTRY` under a lowercase id (for example `dejavu_liver`).
3. Select it via `SceneConfig(scene_id="dejavu_liver", tool_id="forceps")` when constructing `SofaEnvironment`, or pass the same object through `run_sofa_rollout_dataset(..., scene_config=...)`.

## Adding a new tool

1. Implement a builder `(**kwargs) -> Callable[[scene, RobotCommand], None]` mirroring `build_forceps_action_applier`.
2. Register it in `TOOL_REGISTRY`.
3. Point `SceneConfig.tool_id` at the new id.

## XML vs Python factories

- If `SceneConfig.scene_xml_path` is set, that path wins and the registry factory is ignored.
- Otherwise `scene_id` must resolve to a Python factory for `sofa_scene_path=None` native runs.
