"""Typer CLI for smoke tests and capture/record utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import typer

from auto_surgery.env.capture import default_captures
from auto_surgery.recording import brain_forceps
from auto_surgery.schemas.manifests import SceneConfig
from auto_surgery.training.bootstrap import run_m1_tiny_overfit, run_m2_contrastive_stub
from auto_surgery.training.extract_pseudo_actions import (
    extract_pseudo_actions as run_extract_pseudo_actions,
)
from auto_surgery.training.idm_train import _load_dataset_manifest_from_uri, train_idm as run_train_idm
from auto_surgery.training.render_rollout_video import build_preview
from auto_surgery.training.sofa_smoke import run_sofa_rollout_dataset
from auto_surgery.training.sofa_forceps_smoke import run_dejavu_forceps_smoke
from auto_surgery.training.smoke import run_blackwell_smoke

app = typer.Typer(no_args_is_help=True)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRAINFORCEPS_DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"


@app.command()
def smoke(
    skip_gpu: bool = typer.Option(False, help="Skip CUDA exercises (CI / laptops without GPU)."),
) -> None:
    """Run Blackwell/CUDA smoke gate from `training.smoke`."""

    typer.echo(json.dumps(run_blackwell_smoke(skip_gpu=skip_gpu), indent=2))


@app.command()
def bootstrap_m1() -> None:
    """Run tiny overfit loop (requires torch)."""

    loss = run_m1_tiny_overfit()
    typer.echo(json.dumps({"final_loss": loss}, indent=2))


@app.command()
def bootstrap_m2() -> None:
    """Run contrastive stub (requires torch)."""

    loss = run_m2_contrastive_stub()
    typer.echo(json.dumps({"final_loss": loss}, indent=2))


def _build_video_namespace(
    qgl_view: Path | None,
    scene: Path | None,
    output: Path | None,
    output_dir: Path,
    prefix: str,
    frames: int,
    width: int,
    height: int,
    seed: int,
    position_x: float,
    position_y: float,
    position_z: float,
    look_at_x: float,
    look_at_y: float,
    look_at_z: float,
    qgl_distance: float | None,
    fps: float,
    joint_start: float,
    joint_step: float,
    joint_sine_amplitude: float,
    joint_sine_frequency: float,
    base_timestamp: int,
    timestamp_step: int,
    overwrite: bool,
    keep_frames: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        scene=str(scene) if scene is not None else None,
        output_dir=output_dir,
        prefix=prefix,
        frames=frames,
        width=width,
        height=height,
        seed=seed,
        position=(position_x, position_y, position_z),
        look_at=(look_at_x, look_at_y, look_at_z),
        qgl_view=str(qgl_view) if qgl_view is not None else None,
        qgl_distance=qgl_distance,
        joint_start=joint_start,
        joint_step=joint_step,
        joint_sine_amplitude=joint_sine_amplitude,
        joint_sine_frequency=joint_sine_frequency,
        base_timestamp=base_timestamp,
        timestamp_step=timestamp_step,
        output=output,
        fps=fps,
        overwrite=overwrite,
        keep_frames=keep_frames,
    )


def _build_png_namespace(
    qgl_view: Path | None,
    scene: Path | None,
    output_dir: Path,
    prefix: str,
    frames: int,
    width: int,
    height: int,
    seed: int,
    position_x: float,
    position_y: float,
    position_z: float,
    look_at_x: float,
    look_at_y: float,
    look_at_z: float,
    qgl_distance: float | None,
    joint_start: float,
    joint_step: float,
    joint_sine_amplitude: float,
    joint_sine_frequency: float,
    base_timestamp: int,
    timestamp_step: int,
    overwrite: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        scene=str(scene) if scene is not None else None,
        output_dir=output_dir,
        prefix=prefix,
        frames=frames,
        width=width,
        height=height,
        seed=seed,
        position=(position_x, position_y, position_z),
        look_at=(look_at_x, look_at_y, look_at_z),
        qgl_view=str(qgl_view) if qgl_view is not None else None,
        qgl_distance=qgl_distance,
        joint_start=joint_start,
        joint_step=joint_step,
        joint_sine_amplitude=joint_sine_amplitude,
        joint_sine_frequency=joint_sine_frequency,
        base_timestamp=base_timestamp,
        timestamp_step=timestamp_step,
        overwrite=overwrite,
    )


@app.command()
def capture_brain_forceps_video(
    qgl_view: Path | None = typer.Option(
        None,
        "--qgl-view",
        help="Path to a qglviewer view file (position + quaternion).",
    ),
    scene: Path | None = typer.Option(
        None,
        "--scene",
        help="Path to a SOFA .scn scene file.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output mp4 path. Defaults to <output-dir>/<prefix>_video.mp4.",
    ),
    output_dir: Path = typer.Option(
        BRAINFORCEPS_DEFAULT_OUTPUT_DIR,
        "--output-dir",
        help="Directory where intermediate frames are written.",
    ),
    prefix: str = typer.Option(
        "brain_forceps_sample",
        "--prefix",
        help="Filename prefix for captured frames.",
    ),
    frames: int = typer.Option(180, "--frames", min=1, help="Number of frames to capture."),
    width: int = typer.Option(950, "--width", min=1, help="Capture width in pixels."),
    height: int = typer.Option(700, "--height", min=1, help="Capture height in pixels."),
    seed: int = typer.Option(7, "--seed", help="Random seed used for EnvConfig during reset."),
    position_x: float = typer.Option(
        0.0,
        "--position-x",
        help="Offscreen camera X coordinate when not using --qgl-view.",
    ),
    position_y: float = typer.Option(
        45.0,
        "--position-y",
        help="Offscreen camera Y coordinate when not using --qgl-view.",
    ),
    position_z: float = typer.Option(
        140.0,
        "--position-z",
        help="Offscreen camera Z coordinate when not using --qgl-view.",
    ),
    look_at_x: float = typer.Option(
        0.0,
        "--look-at-x",
        help="Camera look-at X coordinate.",
    ),
    look_at_y: float = typer.Option(
        0.0,
        "--look-at-y",
        help="Camera look-at Y coordinate.",
    ),
    look_at_z: float = typer.Option(
        0.0,
        "--look-at-z",
        help="Camera look-at Z coordinate.",
    ),
    qgl_distance: float | None = typer.Option(
        None,
        "--qgl-distance",
        help=(
            "Explicit look-at distance from qgl camera position. If omitted, uses "
            "distance from qgl position to --look-at."
        ),
    ),
    fps: float = typer.Option(30.0, "--fps", min=0.1, help="Output playback frame rate."),
    joint_start: float = typer.Option(
        0.0,
        "--joint-start",
        help="Initial j0 action value for forceps.",
    ),
    joint_step: float = typer.Option(
        0.0,
        "--joint-step",
        help="Increment added to j0 per frame.",
    ),
    joint_sine_amplitude: float = typer.Option(
        0.0,
        "--joint-sine-amplitude",
        help="Optional sine motion amplitude added to j0.",
    ),
    joint_sine_frequency: float = typer.Option(
        0.1,
        "--joint-sine-frequency",
        help="Frequency multiplier for optional sine motion.",
    ),
    base_timestamp: int = typer.Option(
        1_000_000,
        "--base-timestamp",
        help="Timestamp base used for the first frame (ns).",
    ),
    timestamp_step: int = typer.Option(
        1,
        "--timestamp-step",
        help="Timestamp increment (ns) between frames.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing frame/video outputs.",
    ),
    keep_frames: bool = typer.Option(
        False,
        "--keep-frames",
        help="Keep temporary PNG frames after video generation.",
    ),
) -> None:
    """Capture SOFA brain-forceps PNGs and render them into an MP4 in one command."""

    command_args = _build_video_namespace(
        qgl_view=qgl_view,
        scene=scene,
        output=output,
        output_dir=output_dir,
        prefix=prefix,
        frames=frames,
        width=width,
        height=height,
        seed=seed,
        position_x=position_x,
        position_y=position_y,
        position_z=position_z,
        look_at_x=look_at_x,
        look_at_y=look_at_y,
        look_at_z=look_at_z,
        qgl_distance=qgl_distance,
        fps=fps,
        joint_start=joint_start,
        joint_step=joint_step,
        joint_sine_amplitude=joint_sine_amplitude,
        joint_sine_frequency=joint_sine_frequency,
        base_timestamp=base_timestamp,
        timestamp_step=timestamp_step,
        overwrite=overwrite,
        keep_frames=keep_frames,
    )
    output_path = brain_forceps.run_capture_brain_forceps_video(command_args)
    typer.echo(str(output_path))


@app.command()
def capture_brain_forceps_pngs(
    qgl_view: Path | None = typer.Option(
        None,
        "--qgl-view",
        help="Path to a qglviewer view file (position + quaternion).",
    ),
    scene: Path | None = typer.Option(
        None,
        "--scene",
        help="Path to a SOFA .scn scene file.",
    ),
    output_dir: Path = typer.Option(
        BRAINFORCEPS_DEFAULT_OUTPUT_DIR,
        "--output-dir",
        help="Directory where PNG frames are written.",
    ),
    prefix: str = typer.Option(
        "brain_forceps_sample",
        "--prefix",
        help="Filename prefix for captured frames.",
    ),
    frames: int = typer.Option(180, "--frames", min=1, help="Number of frames to capture."),
    width: int = typer.Option(950, "--width", min=1, help="Capture width in pixels."),
    height: int = typer.Option(700, "--height", min=1, help="Capture height in pixels."),
    seed: int = typer.Option(7, "--seed", help="Random seed used for EnvConfig during reset."),
    position_x: float = typer.Option(
        0.0,
        "--position-x",
        help="Offscreen camera X coordinate when not using --qgl-view.",
    ),
    position_y: float = typer.Option(
        45.0,
        "--position-y",
        help="Offscreen camera Y coordinate when not using --qgl-view.",
    ),
    position_z: float = typer.Option(
        140.0,
        "--position-z",
        help="Offscreen camera Z coordinate when not using --qgl-view.",
    ),
    look_at_x: float = typer.Option(
        0.0,
        "--look-at-x",
        help="Camera look-at X coordinate.",
    ),
    look_at_y: float = typer.Option(
        0.0,
        "--look-at-y",
        help="Camera look-at Y coordinate.",
    ),
    look_at_z: float = typer.Option(
        0.0,
        "--look-at-z",
        help="Camera look-at Z coordinate.",
    ),
    qgl_distance: float | None = typer.Option(
        None,
        "--qgl-distance",
        help=(
            "Explicit look-at distance from qgl camera position. If omitted, uses "
            "distance from qgl position to --look-at."
        ),
    ),
    joint_start: float = typer.Option(
        0.0,
        "--joint-start",
        help="Initial j0 action value for forceps.",
    ),
    joint_step: float = typer.Option(
        0.0,
        "--joint-step",
        help="Increment added to j0 per frame.",
    ),
    joint_sine_amplitude: float = typer.Option(
        0.0,
        "--joint-sine-amplitude",
        help="Optional sine motion amplitude added to j0.",
    ),
    joint_sine_frequency: float = typer.Option(
        0.1,
        "--joint-sine-frequency",
        help="Frequency multiplier for optional sine motion.",
    ),
    base_timestamp: int = typer.Option(
        1_000_000,
        "--base-timestamp",
        help="Timestamp base used for the first frame (ns).",
    ),
    timestamp_step: int = typer.Option(
        1,
        "--timestamp-step",
        help="Timestamp increment (ns) between frames.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing frame outputs.",
    ),
) -> None:
    """Capture SOFA brain-forceps PNG frames without creating a movie."""

    command_args = _build_png_namespace(
        qgl_view=qgl_view,
        scene=scene,
        output_dir=output_dir,
        prefix=prefix,
        frames=frames,
        width=width,
        height=height,
        seed=seed,
        position_x=position_x,
        position_y=position_y,
        position_z=position_z,
        look_at_x=look_at_x,
        look_at_y=look_at_y,
        look_at_z=look_at_z,
        qgl_distance=qgl_distance,
        joint_start=joint_start,
        joint_step=joint_step,
        joint_sine_amplitude=joint_sine_amplitude,
        joint_sine_frequency=joint_sine_frequency,
        base_timestamp=base_timestamp,
        timestamp_step=timestamp_step,
        overwrite=overwrite,
    )
    frame_paths = brain_forceps.run_capture_brain_forceps_pngs(command_args)
    for path in frame_paths:
        typer.echo(str(path))


@app.command()
def run_one_episode(
    storage_root_uri: str = typer.Option(..., help="fsspec root URI ending with '/'."),
    case_id: str = typer.Option(...),
    session_id: str = typer.Option(...),
    sofa_scene_path: str | None = typer.Option(
        None,
        help="Optional `.scn` path; when omitted, `scene_id` selects the Python factory.",
    ),
    steps: int = typer.Option(64),
    seed: int = typer.Option(7),
    stub: bool = typer.Option(True, help="Use stub SOFA backend when true."),
    rgb: bool = typer.Option(False, help="Persist native SOFA RGB blobs."),
    scene_id: str = typer.Option("dejavu_brain"),
    tool_id: str = typer.Option("forceps"),
) -> None:
    """Materialize one dataset manifest + optional RGB blobs."""
    captures = default_captures(include_stereo_depth_stubs=False) if rgb else []
    scene_cfg = SceneConfig(
        scene_id=scene_id,
        tool_id=tool_id,
        scene_xml_path=sofa_scene_path,
    )
    ds = run_sofa_rollout_dataset(
        storage_root_uri=storage_root_uri,
        case_id=case_id,
        session_id=session_id,
        sofa_scene_path=None,
        scene_config=scene_cfg,
        fallback_to_stub=stub,
        steps=steps,
        seed=seed,
        capture_modalities=captures if rgb else None,
    )
    typer.echo(json.dumps(ds.model_dump(), indent=2))


@app.command()
def train_idm(
    dataset_manifest_uri: str = typer.Option(..., help="fsspec URI to DatasetManifest JSON."),
    out_ckpt_uri: str = typer.Option(..., help="fsspec URI to write .pt checkpoint."),
    steps: int = typer.Option(300),
    lr: float = typer.Option(2e-3),
    hidden_dim: int = typer.Option(256),
    device: str | None = typer.Option(None),
) -> None:
    """Train Stage-0 IDM on a dataset manifest."""
    manifest = _load_dataset_manifest_from_uri(dataset_manifest_uri)
    metrics = run_train_idm(
        manifest,
        out_ckpt_uri=out_ckpt_uri,
        steps=steps,
        lr=lr,
        hidden_dim=hidden_dim,
        device=device,
        dataset_manifest_path=dataset_manifest_uri,
    )
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def extract_pseudo_actions(
    dataset_manifest_uri: str = typer.Option(..., help="fsspec URI to DatasetManifest JSON."),
    idm_ckpt_uri: str = typer.Option(..., help="fsspec URI to saved IDM checkpoint."),
    out_root_uri: str = typer.Option(..., help="fsspec URI destination root."),
    out_case_id: str = typer.Option(...),
    out_session_id: str = typer.Option(...),
    capture_rig_id: str = typer.Option("rig"),
    clock_source: str = typer.Option("monotonic"),
    software_git_sha: str = typer.Option("stage0"),
    device: str | None = typer.Option(None),
) -> None:
    """Run IDM over logged frames and write derived pseudo-action dataset."""
    manifest = _load_dataset_manifest_from_uri(dataset_manifest_uri)
    ds = run_extract_pseudo_actions(
        manifest,
        idm_ckpt_uri=idm_ckpt_uri,
        out_root_uri=out_root_uri,
        out_case_id=out_case_id,
        out_session_id=out_session_id,
        capture_rig_id=capture_rig_id,
        clock_source=clock_source,
        software_git_sha=software_git_sha,
        device=device,
    )
    typer.echo(json.dumps(ds.model_dump(), indent=2))


@app.command()
def render_rollout_preview(
    storage_root_uri: str = typer.Option(...),
    case_id: str = typer.Option(...),
    session_id: str = typer.Option(...),
    duration_sec: float = typer.Option(10.0, help="Target output duration in seconds."),
    output: str = typer.Option(
        "/tmp/sofa_rollout_preview.gif", help="Output path (supports .mp4 or .gif)."
    ),
    prefer_mp4: bool = typer.Option(
        False, help="Allow MP4 when output extension is .mp4; otherwise fallback to GIF."
    ),
) -> None:
    """Render rollout preview clip from persisted RGB blobs."""
    output_path = Path(output)
    build_preview(
        storage_root_uri=storage_root_uri,
        case_id=case_id,
        session_id=session_id,
        duration_seconds=duration_sec,
        output_path=output_path,
        prefer_mp4=prefer_mp4,
    )
    typer.echo(f"preview written: {output_path}")


@app.command()
def sofa_forceps_smoke(
    scene_path: str | None = typer.Option(
        None,
        help="Optional .scn scene path. Defaults to resolved DejaVu placeholder scene.",
    ),
    steps: int = typer.Option(2, min=1),
) -> None:
    """Run quick native SOFA forceps smoke capture and print first frame path."""
    outputs = run_dejavu_forceps_smoke(scene_path=scene_path, steps=steps)
    if outputs:
        typer.echo(str(outputs[0]))
    else:
        raise RuntimeError("No outputs were produced by sofa forceps smoke.")



def main() -> None:
    app()


if __name__ == "__main__":
    main()
