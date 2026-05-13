"""Brain forceps capture and video rendering helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import shutil
import subprocess
import tempfile
from pathlib import Path
import warnings
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from pydantic import BaseModel, Field, ValidationError

from auto_surgery.env.capture import SofaNativeRgbCapture
from auto_surgery.env.sofa import SofaEnvironment
from auto_surgery.config import load_motion_config, load_scene_config
from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_dejavu_root
from auto_surgery.randomization.scn_template import render_scene_template
from auto_surgery.randomization.sampler import sample_episode as sample_episode_with_axes
from auto_surgery.randomization.presets import load_randomization_preset
from auto_surgery.motion import MotionGeneratorConfig, RealisedPrimitive, SurgicalMotionGenerator
from auto_surgery.logging import storage
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.logging import ForcepsContractTrace
from auto_surgery.schemas.manifests import (
    DataClassification,
    DatasetManifest,
    EnvConfig,
    RetentionTier,
    RunMetadata,
    SessionManifest,
)
from auto_surgery.schemas.scene import SceneConfig
from auto_surgery.schemas.randomization import EpisodeRandomizationConfig
from auto_surgery.schemas.sensors import Contact


def _repo_root() -> Path:
    """Return repository root assuming this file is under ``src/auto_surgery``."""

    return Path(__file__).resolve().parents[3]


_DEFAULT_SCENE_CONFIG_PATH = _repo_root() / "configs" / "scenes" / "dejavu_brain.yaml"
_DEFAULT_MOTION_CONFIG_PATH = (_repo_root() / "configs" / "motion" / "default.yaml").resolve()
_DEFAULT_RANDOMIZATION_PRESET_PATH = _repo_root() / "configs" / "randomization" / "default.yaml"
_MAX_UINT64 = 2**64 - 1
_BF_CASE_ID = "brain_forceps"
_BF_CAPTURE_RIG_ID = "brain_forceps_recorder"
_BF_RUN_MANIFEST = "run_manifest.json"
_BF_SOFTWARE_GIT_SHA = "brain-forceps"
_BF_TRACE_ARTIFACT_KEY = "forceps_contract_trace"


def _resolve_from_repo_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (_repo_root() / candidate).resolve()


def _coerce_manifest_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _coerce_manifest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_manifest_value(item) for item in value]
    return value


def _episode_session_id(episode_index: int) -> str:
    return f"episode_{episode_index:04d}"


def _coerce_trace_contacts(raw: Any) -> list[Contact]:
    if not isinstance(raw, list):
        return []
    out: list[Contact] = []
    for item in raw:
        if isinstance(item, Contact):
            out.append(item)
            continue
        if isinstance(item, dict):
            try:
                out.append(Contact.model_validate(item))
            except ValidationError:
                continue
    return out


def _extract_contacts(env: Any) -> list[Contact]:
    getter = getattr(env, "get_contacts", None)
    if not callable(getter):
        return []
    try:
        return _coerce_trace_contacts(getter())
    except (TypeError, OSError, RuntimeError):
        return []


def _build_forceps_trace_record(
    frame_index: int,
    step,
    command: RobotCommand,
    contacts: list[Contact],
) -> ForcepsContractTrace:
    return ForcepsContractTrace(
        frame_index=frame_index,
        timestamp_ns=step.sensors.timestamp_ns,
        step_sim_index=step.sim_step_index,
        command_cycle_id=command.cycle_id,
        command_enable=command.enable,
        command_twist=command.cartesian_twist,
        tool_pose=step.sensors.tool.pose,
        tool_twist=step.sensors.tool.twist,
        tool_wrench=step.sensors.tool.wrench,
        tool_jaw=step.sensors.tool.jaw,
        tool_in_contact=step.sensors.tool.in_contact,
        contact_count=len(contacts),
        contacts=contacts,
        safety_blocked=step.sensors.safety.command_blocked,
        safety_block_reason=step.sensors.safety.block_reason,
    )


class _EpisodeSpec(BaseModel):
    """Per-episode reproducibility bundle for piece-4 recorder outputs."""

    model_config = {"extra": "forbid"}

    seed: int
    scene: SceneConfig
    motion: MotionGeneratorConfig
    sample_record: dict[str, Any] = Field(default_factory=dict)


def _normalise_primitive(primitive: Any) -> dict[str, Any]:
    return _coerce_manifest_value(asdict(primitive))


def _resolve_master_seed(*, explicit_master_seed: int | None, legacy_seed: int | None) -> int:
    """Resolve a canonical master seed from canonical and deprecated aliases."""

    if explicit_master_seed is not None and legacy_seed is not None:
        if int(explicit_master_seed) != int(legacy_seed):
            raise ValueError("Conflicting values for --master-seed and deprecated --seed.")
        warnings.warn(
            "The --seed flag is deprecated; use --master-seed instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(explicit_master_seed)

    if legacy_seed is not None:
        warnings.warn(
            "The --seed flag is deprecated; use --master-seed instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(legacy_seed)

    if explicit_master_seed is None:
        raise ValueError("Either --master-seed or --seed is required.")

    return int(explicit_master_seed)


def _build_run_output_root(output_dir: Path, master_seed: int) -> Path:
    run_id = f"run_seed_{int(master_seed)}"
    return output_dir / run_id


def _no_camera_perturb_preset_path() -> Path:
    return _resolve_from_repo_root("configs/randomization/no_camera_perturb.yaml")


def _use_canonical_tissue_mesh_files_for_preset(preset_path: str) -> bool:
    """Match POC brain geometry when using the no-camera-jitter preset.

    That preset still samples tissue mesh warps; POC loads canonical OBJs only.
    """

    return Path(preset_path).expanduser().resolve() == _no_camera_perturb_preset_path()


def _episode_stream(master_seed: int) -> np.random.Generator:
    return np.random.default_rng(int(master_seed))


def _sample_episode(
    base_scene_config: SceneConfig,
    base_motion_config: MotionGeneratorConfig,
    randomization: EpisodeRandomizationConfig,
    episode_seed: int,
) -> _EpisodeSpec:
    sampled_spec = sample_episode_with_axes(
        base_scene_config,
        base_motion_config,
        randomization=randomization,
        episode_seed=int(episode_seed),
    )
    sample_record = sampled_spec.sample_record.model_dump()
    sample_record["motion_seed"] = int(sampled_spec.motion.seed)
    sample_record["randomization_preset"] = _coerce_manifest_value(randomization.model_dump())
    return _EpisodeSpec(
        seed=int(sampled_spec.episode_seed),
        scene=sampled_spec.scene,
        motion=sampled_spec.motion,
        sample_record=sample_record,
    )


def build_capture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture repeated PNGs from the DejaVu brain forceps scene via SOFA "
            "OffscreenCamera."
        )
    )
    parser.add_argument(
        "--scene-config",
        type=str,
        default=str(_DEFAULT_SCENE_CONFIG_PATH),
        help=(
            "Path to a scene config YAML (SceneConfig). Defaults to "
            f"configs/scenes/dejavu_brain.yaml."
        ),
    )
    parser.add_argument(
        "--motion-config",
        type=str,
        default=str(_DEFAULT_MOTION_CONFIG_PATH),
        help=(
            "Path to a motion config YAML (MotionGeneratorConfig). Defaults to "
            f"configs/motion/default.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "artifacts",
        help="Directory where PNG frames will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="brain_forceps_sample",
        help="Filename prefix for captured frames.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=64,
        help="Number of control ticks to run.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=950,
        help="Capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=700,
        help="Capture height in pixels.",
    )
    parser.add_argument(
        "--randomization-preset",
        type=str,
        default=str(_DEFAULT_RANDOMIZATION_PRESET_PATH),
        help=(
            "Path to the piece-4 randomization preset YAML. Defaults to "
            f"{_DEFAULT_RANDOMIZATION_PRESET_PATH}."
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="How many episodes to record.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        required=False,
        help="Master seed for per-episode episode_seed derivation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        dest="legacy_seed",
        help=(
            "Deprecated compatibility alias for --master-seed. "
            "Use --master-seed for new workflows."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files with the same generated names.",
    )
    return parser


def build_video_parser() -> argparse.ArgumentParser:
    parser = build_capture_parser()
    parser.description = (
        "Capture brain-forceps frames in SOFA and render them to a clip in one command."
    )
    parser.set_defaults(ticks=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output video path. Defaults to <output-dir>/<prefix>_video.mp4 if omitted."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output playback frame rate.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help=(
            "Keep PNG frame files after the video is generated. Defaults to deleting "
            "them."
        ),
    )
    return parser


def _load_motion_configs(args: argparse.Namespace) -> tuple[MotionGeneratorConfig, SceneConfig]:
    scene_config = load_scene_config(args.scene_config)
    motion_config = load_motion_config(args.motion_config)
    return motion_config, scene_config


def _motion_manifest_payload(sequence: tuple[RealisedPrimitive, ...]) -> list[dict[str, Any]]:
    return [
        {
            "primitive": item.primitive.__class__.__name__,
            "started_at_tick": item.started_at_tick,
            "ended_at_tick": item.ended_at_tick,
            "early_terminated": item.early_terminated,
            "params": _normalise_primitive(item.primitive),
        }
        for item in sequence
    ]


def _write_motion_manifest(*, output_path: Path, sequence: tuple[RealisedPrimitive, ...]) -> None:
    manifest = output_path
    manifest.write_text(
        json.dumps(_motion_manifest_payload(sequence), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_episode_spec(*, output_path: Path, spec: _EpisodeSpec) -> None:
    output_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")


def _write_randomization_preset(*, output_path: Path, preset_text: str) -> None:
    output_path.write_text(preset_text, encoding="utf-8")


def _write_control_command_parquet(*, output_path: Path, commands: list[RobotCommand]) -> None:
    table = pa.table(
        {
            "cycle_id": [command.cycle_id for command in commands],
            "timestamp_ns": [command.timestamp_ns for command in commands],
            "command_json": [command.model_dump_json() for command in commands],
        }
    )
    pq.write_table(table, output_path)


def _write_forceps_trace_parquet(*, output_path: Path, traces: list[ForcepsContractTrace]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "frame_index": pa.array([trace.frame_index for trace in traces], type=pa.int64()),
            "payload_json": pa.array(
                [trace.model_dump_json() for trace in traces],
                type=pa.string(),
            ),
        }
    )
    pq.write_table(table, output_path)


def _write_session_and_run_manifests(
    *,
    run_output_dir: Path,
    run_id: str,
    episode_index: int,
    args: argparse.Namespace,
    episode_spec: _EpisodeSpec,
    forceps_trace_relative: str,
) -> None:
    scene = episode_spec.scene
    session_id = _episode_session_id(episode_index)
    session_manifest_relative = storage.session_manifest_path(_BF_CASE_ID, session_id)
    run_metadata_relative = storage.run_metadata_path(_BF_CASE_ID, session_id)
    manifest = SessionManifest(
        session_id=session_id,
        case_id=_BF_CASE_ID,
        capture_rig_id=_BF_CAPTURE_RIG_ID,
        clock_source="monotonic",
        software_git_sha=_BF_SOFTWARE_GIT_SHA,
        logged_frame_schema_version="logged_frame_v1",
        sensor_list=["tool_pose", "tool_twist", "tool_wrench", "contact", "jaw"],
        data_classification=DataClassification.SIMULATION,
        retention_tier=RetentionTier.CURATED_TRAINING,
        artifact_paths={
            _BF_TRACE_ARTIFACT_KEY: forceps_trace_relative,
            "run_metadata": storage.run_metadata_path(_BF_CASE_ID, session_id),
            "episode_spec": f"episode_{episode_index:04d}/episode_spec.json",
            "motion_plan": f"episode_{episode_index:04d}/motion_plan.json",
            "control_commands": f"episode_{episode_index:04d}/control_commands.parquet",
            "preset": f"episode_{episode_index:04d}/preset.yaml",
        },
    )
    session_manifest_path = run_output_dir / session_manifest_relative
    session_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    session_manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    run_metadata = RunMetadata(
        run_id=run_id,
        software_git_sha=_BF_SOFTWARE_GIT_SHA,
        steps_requested=args.ticks,
        fallback_to_stub=False,
        sofa_scene_path=str(scene.tissue_scene_path),
        sofa_scene_id=scene.scene_id,
        sofa_tool_id=str(scene.tool.tool_id),
        action_generator_config=episode_spec.motion.model_dump(),
        capture_modalities=["rgb"],
        session_manifest_path=session_manifest_relative,
        run_manifest_path=_BF_RUN_MANIFEST,
        trace_artifact_paths={
            _BF_TRACE_ARTIFACT_KEY: forceps_trace_relative,
        },
    )
    run_metadata_path = run_output_dir / run_metadata_relative
    run_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    run_metadata_path.write_text(
        run_metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _capture_rate_hz(args: argparse.Namespace) -> float:
    """Resolve the per-tick simulation/capture rate for brain-forceps recording.

    The recorder captures every control tick (frame_decimation == 1) so the
    written video plays the simulation at real-time. We anchor ``control_rate_hz``
    to ``args.fps`` when present so that ``ticks / fps`` seconds of simulation map
    1:1 to ``ticks / fps`` seconds of playback.
    """

    raw_fps = getattr(args, "fps", None)
    fps = float(raw_fps) if raw_fps is not None else 30.0
    if fps <= 0.0:
        raise ValueError("fps must be greater than 0")
    return fps


def _run_one_episode(
    args: argparse.Namespace,
    episode_index: int,
    base_output_dir: Path,
    episode_spec: _EpisodeSpec,
    preset_text: str,
    *,
    use_canonical_tissue_mesh_files: bool,
) -> list[Path]:
    if args.ticks <= 0:
        raise ValueError("ticks must be greater than 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width/height must be positive")

    episode_dir = base_output_dir / f"episode_{episode_index:04d}"
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    _write_randomization_preset(output_path=episode_dir / "preset.yaml", preset_text=preset_text)

    capture = SofaNativeRgbCapture(width=args.width, height=args.height)
    episode_has_randomization = any(
        bool(episode_spec.sample_record.get(axis))
        for axis in (
            "tissue_material",
            "tissue_topology",
            "tissue_mesh",
            "camera",
            "lighting",
            "visual_tint",
        )
    )
    scene = episode_spec.scene
    if episode_has_randomization:
        scene = scene.model_copy(
            update={
                "tissue_scene_path": render_scene_template(
                    scene,
                    dejavu_root=resolve_dejavu_root(),
                    use_canonical_tissue_mesh_files=use_canonical_tissue_mesh_files,
                )
            }
        )
        episode_spec = episode_spec.model_copy(update={"scene": scene})

    _write_episode_spec(
        output_path=episode_dir / "episode_spec.json",
        spec=episode_spec,
    )

    capture_rate_hz = _capture_rate_hz(args)
    env_kwargs = dict(
        scene_config=scene,
        step_dt=1.0 / capture_rate_hz,
        pre_init_hooks=[capture.pre_init_hook],
    )
    env_kwargs["sofa_scene_path"] = str(
        _resolve_from_repo_root(scene.tissue_scene_path)
    )

    env = SofaEnvironment(**env_kwargs)
    try:
        last_step = env.reset(
            EnvConfig(
                seed=episode_spec.seed,
                scene=scene,
                control_rate_hz=capture_rate_hz,
                frame_rate_hz=capture_rate_hz,
            )
        )
        generator = SurgicalMotionGenerator(episode_spec.motion, scene)
        command = generator.reset(last_step)
        root_node = env.sofa_scene_root

        frame_digits = max(4, len(str(max(args.ticks - 1, 1))))
        written: list[Path] = []
        forceps_trace: list[ForcepsContractTrace] = []
        commands: list[RobotCommand] = []

        _print_progress_bar(0, args.ticks, label=f"Episode {episode_index}")
        for tick in range(args.ticks):
            last_step = env.step(command)
            commands.append(command)
            forceps_trace.append(
                _build_forceps_trace_record(
                    frame_index=tick,
                    step=last_step,
                    command=command,
                    contacts=_extract_contacts(env),
                )
            )
            if last_step.is_capture_tick:
                payload = capture.capture(
                    root_node=root_node, step_index=last_step.sim_step_index
                )
                if payload.get("encoding") != "image/png" or "bytes" not in payload:
                    raise RuntimeError(
                        f"Unsupported capture payload at tick {tick}: {payload!r}"
                    )
                out_path = frames_dir / f"{args.prefix}_{len(written):0{frame_digits}d}.png"
                if out_path.exists() and not args.overwrite:
                    raise FileExistsError(f"Output exists and --overwrite is not set: {out_path}")
                out_path.write_bytes(payload["bytes"])
                written.append(out_path)
                _print_progress_bar(len(written), args.ticks, label="Capturing frames")
            command = generator.next_command(last_step)
            _print_progress_bar(tick + 1, args.ticks, label=f"Episode {episode_index}")

        generator.finalize(last_step)
        trace_relative = storage.forceps_trace_path(
            _BF_CASE_ID, _episode_session_id(episode_index)
        )
        _write_forceps_trace_parquet(
            output_path=base_output_dir / trace_relative,
            traces=forceps_trace,
        )
        _write_motion_manifest(
            output_path=episode_dir / "motion_plan.json",
            sequence=generator.realised_sequence,
        )
        _write_control_command_parquet(
            output_path=episode_dir / "control_commands.parquet",
            commands=commands,
        )
        _write_session_and_run_manifests(
            run_output_dir=base_output_dir,
            run_id=f"run_seed_{int(args.master_seed)}",
            episode_index=episode_index,
            args=args,
            episode_spec=episode_spec,
            forceps_trace_relative=trace_relative,
        )

    finally:
        env.close()

    # Maintain compatibility for single-episode runs by preserving legacy names under the run dir.
    if episode_index == 0:
        legacy_motion = base_output_dir / f"{args.prefix}_motion_plan.json"
        legacy_commands = base_output_dir / f"{args.prefix}_control_commands.parquet"
        legacy_motion.write_text(
            (episode_dir / "motion_plan.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        legacy_commands.write_bytes((episode_dir / "control_commands.parquet").read_bytes())
    return written


def run_capture_brain_forceps_pngs(args: argparse.Namespace) -> list[Path]:
    args.master_seed = _resolve_master_seed(
        explicit_master_seed=getattr(args, "master_seed", None),
        legacy_seed=getattr(args, "legacy_seed", None),
    )
    output_dir = args.output_dir.expanduser().resolve()
    motion_config, base_scene_config = _load_motion_configs(args)
    randomization_preset = load_randomization_preset(args.randomization_preset)
    preset_text = yaml.safe_dump(
        randomization_preset.model_dump(exclude_none=True),
        sort_keys=False,
    )
    canonical_tissue = _use_canonical_tissue_mesh_files_for_preset(str(args.randomization_preset))
    run_output = _build_run_output_root(output_dir, args.master_seed)
    run_output.mkdir(parents=True, exist_ok=True)

    num_episodes = int(args.num_episodes)
    if num_episodes <= 0:
        raise ValueError("num_episodes must be greater than 0")
    episode_rng = _episode_stream(args.master_seed)
    all_written: list[Path] = []
    run_manifest_uri = (
        _build_run_output_root(output_dir, args.master_seed).as_uri().rstrip("/") + "/"
    )
    run_session_manifests: list[str] = []

    for episode_index in range(num_episodes):
        episode_seed = int(episode_rng.integers(0, _MAX_UINT64, dtype=np.uint64))
        episode_spec = _sample_episode(
            base_scene_config,
            motion_config,
            randomization_preset,
            episode_seed,
        )
        all_written.extend(
            _run_one_episode(
                args,
                episode_index=episode_index,
                base_output_dir=run_output,
                episode_spec=episode_spec,
                preset_text=preset_text,
                use_canonical_tissue_mesh_files=canonical_tissue,
            )
        )
        session_id = _episode_session_id(episode_index)
        run_session_manifests.append(
            f"{run_manifest_uri}{storage.session_manifest_path(_BF_CASE_ID, session_id)}"
        )

    run_manifest = DatasetManifest(
        dataset_id=f"{_BF_CASE_ID}_{args.master_seed}",
        session_manifest_paths=run_session_manifests,
        frame_filters={
            "num_episodes": num_episodes,
            "ticks_per_episode": args.ticks,
            "seed": args.master_seed,
        },
        data_classification=DataClassification.SIMULATION,
        retention_tier=RetentionTier.CURATED_TRAINING,
    )
    (run_output / _BF_RUN_MANIFEST).write_text(
        run_manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return all_written


def _run_ffmpeg(frames: list[Path], fps: float, output: Path) -> None:
    if fps <= 0:
        raise ValueError("fps must be greater than 0.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render video output.")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sofa-forceps-video-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for index, frame_path in enumerate(frames):
            shutil.copy(frame_path, tmp_root / f"frame_{index:06d}.png")
            _print_progress_bar(index + 1, len(frames), label="Copying frames")

        _print_progress_bar(1, 1, label="Encoding video")

        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(tmp_root / "frame_%06d.png"),
            "-vsync",
            "cfr",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-r",
            f"{fps:.6f}",
            str(output),
        ]
        subprocess.run(cmd, check=True, text=True, capture_output=True)


def _infer_video_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return _resolve_from_repo_root(args.output)
    return (
        _build_run_output_root(
            _resolve_from_repo_root(args.output_dir),
            int(args.master_seed),
        )
        / f"{args.prefix}_video.mp4"
    )


def _print_progress_bar(current: int, total: int, *, label: str, width: int = 30) -> None:
    if total <= 0:
        return
    current = min(max(0, current), total)
    ratio = current / total
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(ratio * 100)
    print(
        f"\r{label}: [{bar}] {percent:3d}% ({current}/{total})",
        end="" if current < total else "\n",
        flush=True,
    )


def run_capture_brain_forceps_video(args: argparse.Namespace) -> Path:
    if int(args.num_episodes) != 1:
        raise ValueError("capture-brain-forceps-video supports a single episode.")
    output = _infer_video_output_path(args).expanduser()
    if output.suffix.lower() != ".mp4":
        raise ValueError("This command currently supports MP4 output only.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists and --overwrite is not set: {output}")

    frame_paths = run_capture_brain_forceps_pngs(args)
    try:
        _run_ffmpeg(frame_paths, fps=args.fps, output=output)
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        for frame_path in frame_paths:
            frame_path.unlink(missing_ok=True)
        raise

    if not args.keep_frames:
        for frame_path in frame_paths:
            frame_path.unlink(missing_ok=True)

    return output


def run_capture_pngs_cli(argv: list[str] | None = None) -> list[Path]:
    parser = build_capture_parser()
    args = parser.parse_args(argv)
    return run_capture_brain_forceps_pngs(args)


def run_capture_video_cli(argv: list[str] | None = None) -> Path:
    parser = build_video_parser()
    args = parser.parse_args(argv)
    return run_capture_brain_forceps_video(args)


def main() -> None:
    raise SystemExit(
        "Direct execution of this module is deprecated. "
        "Use: uv run auto-surgery capture-brain-forceps-video or "
        "uv run auto-surgery capture-brain-forceps-pngs."
    )


if __name__ == "__main__":
    main()
