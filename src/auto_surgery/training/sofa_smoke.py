"""SOFA rollout and IDM smoke helpers for Stage-0 execution tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from auto_surgery.env.capture import CaptureModality
from auto_surgery.env.sofa import (
    SofaEnvironment,
    SofaRuntimeBackendFactory,
    SofaSceneFactory,
)
from auto_surgery.env.sofa_rgb_native import validate_native_capture_runtime
from auto_surgery.config import load_motion_config, load_scene_config
from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import SessionWriter
from auto_surgery.motion import SurgicalMotionGenerator
from auto_surgery.schemas.logging import LoggedFrame
from auto_surgery.schemas.manifests import (
    DataClassification,
    DatasetManifest,
    EnvConfig,
    RetentionTier,
    RunMetadata,
    SceneConfig,
)
from auto_surgery.training.datasets import frame_count_estimate
from auto_surgery.training.extract_pseudo_actions import extract_pseudo_actions
from auto_surgery.training.idm_train import train_idm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_CONFIG_PATH = (PROJECT_ROOT / "configs" / "scenes" / "dejavu_brain.yaml").resolve()
DEFAULT_MOTION_CONFIG_PATH = (PROJECT_ROOT / "configs" / "motion" / "default.yaml").resolve()


def _resolve_sofa_scene_path(scene_config: SceneConfig) -> str:
    return str(scene_config.tissue_scene_path.resolve())


def run_sofa_rollout_dataset(
    *,
    storage_root_uri: str,
    case_id: str,
    session_id: str,
    sofa_scene_path: str | None = None,
    steps: int = 64,
    segment_max_frames: int = 128,
    seed: int = 7,
    sofa_backend_factory: SofaRuntimeBackendFactory | None = None,
    sofa_scene_factory: SofaSceneFactory | None = None,
    capture_rig_id: str = "sofa_rig",
    capture_modalities: Sequence[CaptureModality] | None = None,
    scene_config: SceneConfig | None = None,
) -> DatasetManifest:
    """Run a bounded rollout and persist frames into a manifest-consumable dataset."""
    if scene_config is None:
        scene_config = load_scene_config(DEFAULT_SCENE_CONFIG_PATH)
    resolved_scene_path = None
    if sofa_scene_path is None and sofa_scene_factory is None:
        resolved_scene_path = _resolve_sofa_scene_path(scene_config)
    else:
        resolved_scene_path = sofa_scene_path
    if sofa_scene_factory is not None and resolved_scene_path is not None:
        sofa_scene_factory = None
    # Prefer scene factories when supplied; avoid passing both path and factory to runtime init.

    motion_config = load_motion_config(DEFAULT_MOTION_CONFIG_PATH).model_copy(update={"seed": seed})
    gen = SurgicalMotionGenerator(motion_config, scene_config)
    modalities = list(capture_modalities or [])
    active_modalities = modalities
    sensor_names = [m.modality_id() for m in active_modalities]
    pre_init_hooks = (
        [
            cap.pre_init_hook
            for cap in active_modalities
            if callable(getattr(cap, "pre_init_hook", None))
        ]
        if active_modalities
        else []
    )
    env_kwargs: dict[str, object] = {
        "scene_config": scene_config,
        "sofa_backend_factory": sofa_backend_factory,
        "pre_init_hooks": pre_init_hooks or None,
    }
    if resolved_scene_path is not None:
        env_kwargs["sofa_scene_path"] = resolved_scene_path
    if sofa_scene_factory is not None:
        env_kwargs["sofa_scene_factory"] = sofa_scene_factory
    env = SofaEnvironment(**env_kwargs)
    last_step = env.reset(
        EnvConfig(
            seed=seed,
            scene=scene_config,
        )
    )
    command = gen.reset(last_step)
    needs_native_rgb = any(
        cap.modality_id() == "rgb" for cap in active_modalities
    )
    if needs_native_rgb:
        validate_native_capture_runtime()

    root = storage_root_uri.rstrip("/") + "/"
    writer = SessionWriter(
        root,
        case_id,
        session_id,
        capture_rig_id=capture_rig_id,
        clock_source="monotonic",
        software_git_sha="stage0",
        data_classification=DataClassification.SIMULATION,
        retention_tier=RetentionTier.CURATED_TRAINING,
        sensor_list=sensor_names,
        segment_max_frames=segment_max_frames,
    )

    for i in range(steps):
        step = env.step(command)
        if active_modalities:
            scene_root = env.sofa_scene_root
            for cap in active_modalities:
                payload = cap.capture(root_node=scene_root, step_index=step.sim_step_index)
                if payload.get("implemented") is False:
                    continue
                raw = payload.get("bytes")
                if isinstance(raw, bytes) and raw:
                    writer.write_blob(f"{cap.modality_id()}/{i:06d}.png", raw)
        lf = LoggedFrame(
            frame_index=i,
            timestamp_ns=step.sensors.timestamp_ns,
            sensor_payload=step.sensors,
            scene_snapshot=env.get_scene(),
            commanded_action=command,
            executed_action=command,
            safety_decision=None,
            entity_state=None,
            surgeon_input=None,
            outcome_label=None,
        )
        writer.write_frame(lf)
        command = gen.next_command(step)
        last_step = step
    gen.finalize(last_step)

    run_meta = RunMetadata(
        software_git_sha="stage0",
        steps_requested=steps,
        fallback_to_stub=False,
        sofa_scene_path=resolved_scene_path,
        sofa_scene_id=scene_config.scene_id if scene_config else None,
        sofa_tool_id=(scene_config.tool.tool_id if scene_config else None),
        capture_modalities=[m.modality_id() for m in active_modalities],
    )
    writer.finalize(run_metadata=run_meta)
    session_manifest_uri = f"{root}{session_manifest_path(case_id, session_id)}"
    return DatasetManifest(
        dataset_id=f"sofa_stage0_{case_id}_{session_id}",
        session_manifest_paths=[session_manifest_uri],
        frame_filters={"start_step": 0, "num_steps": steps},
        data_classification=DataClassification.SIMULATION,
        retention_tier=RetentionTier.CURATED_TRAINING,
    )


def run_sofa_smoke_pipeline(
    *,
    out_root_uri: str,
    case_id: str,
    session_id: str,
    derived_case_id: str,
    derived_session_id: str,
    sofa_scene_path: str | None = None,
    sofa_backend_factory: SofaRuntimeBackendFactory | None = None,
    sofa_scene_factory: SofaSceneFactory | None = None,
    scene_config: SceneConfig | None = None,
    steps: int = 64,
    train_steps: int = 64,
    train_lr: float = 5e-3,
    hidden_dim: int = 32,
) -> tuple[DatasetManifest, dict[str, float], DatasetManifest]:
    """Execute end-to-end: rollout -> train_idm -> extract pseudo actions."""

    base_root = out_root_uri.rstrip("/") + "/"
    src_ds = run_sofa_rollout_dataset(
        storage_root_uri=base_root,
        case_id=case_id,
        session_id=session_id,
        sofa_scene_path=sofa_scene_path,
        sofa_backend_factory=sofa_backend_factory,
        sofa_scene_factory=sofa_scene_factory,
        scene_config=scene_config,
        steps=steps,
    )
    assert frame_count_estimate(src_ds) == steps

    ckpt_uri = f"{base_root}sofa_idm_stage0.pt"
    train_stats = train_idm(
        src_ds,
        out_ckpt_uri=ckpt_uri,
        checkpoint_id="sofa_stage0_ckpt",
        dataset_manifest_path=f"{base_root}{session_manifest_path(case_id, session_id)}",
        steps=train_steps,
        lr=train_lr,
        hidden_dim=hidden_dim,
        device=None,
        git_sha="stage0-smoke",
    )

    pseudo_ds = extract_pseudo_actions(
        src_ds,
        idm_ckpt_uri=ckpt_uri,
        out_root_uri=base_root,
        out_case_id=derived_case_id,
        out_session_id=derived_session_id,
        capture_rig_id="sofa_rig",
        clock_source="monotonic",
        software_git_sha="stage0-smoke",
        device=None,
    )

    return src_ds, train_stats, pseudo_ds
