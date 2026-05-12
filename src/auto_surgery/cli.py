"""Typer CLI for smoke tests and capture/record utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import typer

from auto_surgery.recording import brain_forceps
from auto_surgery.training.bootstrap import run_m1_tiny_overfit, run_m2_contrastive_stub
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
