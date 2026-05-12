"""Brain forceps capture and video rendering helpers."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from auto_surgery.env.capture import SofaNativeRgbCapture
from auto_surgery.env.sofa import SofaEnvironment
from auto_surgery.env.sofa_rgb_native import attach_capture_camera
from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_brain_forceps_scene_path
from auto_surgery.env.sofa_tools import build_forceps_action_applier
from auto_surgery.env.action_generators import build_sine_joint_position
from auto_surgery.schemas.commands import ControlMode, RobotCommand, Twist, Vec3
from auto_surgery.schemas.manifests import EnvConfig


def _repo_root() -> Path:
    """Return repository root assuming this file is under ``src/auto_surgery``."""

    return Path(__file__).resolve().parents[3]


def _resolve_scene_path(path: str | None) -> str:
    return resolve_brain_forceps_scene_path(path)


def _distance(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _parse_float_tuple(text: str, expected_count: int, *, label: str) -> tuple[float, ...]:
    values = text.split()
    if len(values) != expected_count:
        raise ValueError(
            f"Expected {expected_count} values for {label} in qgl view file, got {len(values)}."
        )

    try:
        return tuple(float(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"Could not parse float values for {label!r}: {text!r}") from exc


def _load_qgl_view(path: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    qgl_path = Path(path).expanduser()
    if not qgl_path.exists():
        alt_path = Path(str(path).replace("\\", "/")).expanduser()
        if alt_path.exists():
            qgl_path = alt_path

    qgl_path = qgl_path.resolve()
    try:
        lines = [
            line.strip()
            for line in qgl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise ValueError(f"Cannot read qgl view file: {qgl_path}") from exc

    if len(lines) < 2:
        raise ValueError(f"Invalid qgl view file; expected 2 lines: {qgl_path}")

    position = _parse_float_tuple(lines[0], 3, label="camera position")
    orientation = _parse_float_tuple(lines[1], 4, label="camera orientation")
    return position, orientation


def _quat_to_forward_axis(
    quaternion: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("Invalid orientation quaternion with zero norm.")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    forward_x = 2.0 * (x * z + w * y)
    forward_y = 2.0 * (y * z - w * x)
    forward_z = 1.0 - 2.0 * (x * x + y * y)
    forward_norm = math.sqrt(forward_x * forward_x + forward_y * forward_y + forward_z * forward_z)
    if forward_norm <= 0.0:
        raise ValueError("Invalid orientation produced a zero forward vector.")

    return (
        forward_x / forward_norm,
        forward_y / forward_norm,
        forward_z / forward_norm,
    )


def _resolve_camera_pose(
    args: argparse.Namespace
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if args.qgl_view is None:
        return tuple(args.position), tuple(args.look_at)

    qgl_position, qgl_orientation = _load_qgl_view(args.qgl_view)
    look_direction = _quat_to_forward_axis(qgl_orientation)
    look_direction_opposite = (-look_direction[0], -look_direction[1], -look_direction[2])

    look_reference = tuple(args.look_at)
    target_delta = (
        look_reference[0] - qgl_position[0],
        look_reference[1] - qgl_position[1],
        look_reference[2] - qgl_position[2],
    )
    target_distance = _distance((0.0, 0.0, 0.0), target_delta)
    if target_distance > 0.0:
        target_unit = (
            target_delta[0] / target_distance,
            target_delta[1] / target_distance,
            target_delta[2] / target_distance,
        )
        align = (
            look_direction[0] * target_unit[0]
            + look_direction[1] * target_unit[1]
            + look_direction[2] * target_unit[2]
        )
        align_opposite = (
            look_direction_opposite[0] * target_unit[0]
            + look_direction_opposite[1] * target_unit[1]
            + look_direction_opposite[2] * target_unit[2]
        )
        if align_opposite > align:
            look_direction = look_direction_opposite

    look_distance = args.qgl_distance
    if look_distance is None:
        look_distance = _distance(qgl_position, look_reference)
    if look_distance <= 0.0:
        raise ValueError(
            "Cannot infer look distance from --qgl-view and --look-at. "
            "Pass --look-at a different point or --qgl-distance."
        )

    return (
        qgl_position,
        (
            qgl_position[0] + look_direction[0] * look_distance,
            qgl_position[1] + look_direction[1] * look_distance,
            qgl_position[2] + look_direction[2] * look_distance,
        ),
    )


def build_capture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture repeated PNGs from the DejaVu brain forceps scene via SOFA "
            "OffscreenCamera."
        )
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help=(
            "Path to a SOFA .scn scene file. Defaults to "
            "src/auto_surgery/env/sofa_scenes/brain_dejavu_forceps_poc.scn "
            "(resolved with AUTO_SURGERY_DEJAVU_ROOT placeholders)."
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
        "--frames",
        type=int,
        default=1,
        help="Number of frames to capture.",
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
        "--seed",
        type=int,
        default=7,
        help="Simulation seed passed to EnvConfig during reset.",
    )
    parser.add_argument(
        "--position",
        nargs=3,
        type=float,
        default=(0.0, 45.0, 140.0),
        metavar=("X", "Y", "Z"),
        help="Offscreen camera position as three floats.",
    )
    parser.add_argument(
        "--look-at",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Offscreen camera look target as three floats.",
    )
    parser.add_argument(
        "--qgl-view",
        type=str,
        default=None,
        help=(
            "Path to a qglviewer view file (position + quaternion). When set, "
            "position and look direction are derived from this file."
        ),
    )
    parser.add_argument(
        "--qgl-distance",
        type=float,
        default=None,
        help=(
            "Explicit look-at distance from qgl camera position. If omitted, uses "
            "distance from qgl position to --look-at (default 0,0,0)."
        ),
    )
    parser.add_argument(
        "--joint-start",
        type=float,
        default=0.0,
        help="Initial j0 action value for forceps.",
    )
    parser.add_argument(
        "--joint-step",
        type=float,
        default=0.0,
        help="Increment added to j0 per frame.",
    )
    parser.add_argument(
        "--joint-sine-amplitude",
        type=float,
        default=0.0,
        help="Optional sine motion amplitude added to j0.",
    )
    parser.add_argument(
        "--joint-sine-frequency",
        type=float,
        default=0.1,
        help="Frequency multiplier for optional sine motion.",
    )
    parser.add_argument(
        "--base-timestamp",
        type=int,
        default=1_000_000,
        help="Timestamp base used for the first frame (ns).",
    )
    parser.add_argument(
        "--timestamp-step",
        type=int,
        default=1,
        help="Timestamp increment (ns) between frames.",
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
    parser.set_defaults(frames=180)
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


def _build_twist(step: int, args: argparse.Namespace) -> Twist:
    linear_x = build_sine_joint_position(
        step,
        joint_start=args.joint_start,
        joint_step=args.joint_step,
        amplitude=args.joint_sine_amplitude,
        phase_scale=args.joint_sine_frequency,
    )
    return Twist(
        linear=Vec3(x=linear_x, y=0.0, z=0.0),
        angular=Vec3(x=0.0, y=0.0, z=0.0),
    )


def _format_timestamp(step: int, args: argparse.Namespace) -> int:
    return args.base_timestamp + args.timestamp_step * step


def run_capture_brain_forceps_pngs(args: argparse.Namespace) -> list[Path]:
    if args.frames <= 0:
        raise ValueError("frames must be greater than 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width/height must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_path = _resolve_scene_path(args.scene)
    position, look_at = _resolve_camera_pose(args)
    capture = SofaNativeRgbCapture(width=args.width, height=args.height)
    action_applier = build_forceps_action_applier()

    camera_position = tuple(position)
    camera_look_at = tuple(look_at)

    def pre_init_hook(root_node, _config) -> None:
        attach_capture_camera(
            root_node,
            width=args.width,
            height=args.height,
            position=camera_position,
            look_at=camera_look_at,
        )

    env = SofaEnvironment(
        sofa_scene_path=scene_path,
        step_dt=0.01,
        action_applier=action_applier,
        pre_init_hooks=[pre_init_hook],
    )
    env.reset(EnvConfig(seed=args.seed, domain_randomization={}))
    root_node = env.sofa_scene_root

    digits = max(4, len(str(max(args.frames - 1, 1))))
    written: list[Path] = []
    for step in range(args.frames):
        action = RobotCommand(
            timestamp_ns=_format_timestamp(step, args),
            cycle_id=step,
            control_mode=ControlMode.CARTESIAN_TWIST,
            cartesian_twist=_build_twist(step, args),
            enable=True,
            source="scripted",
        )
        env.step(action)
        payload = capture.capture(root_node=root_node, step_index=step)
        if payload.get("encoding") != "image/png" or "bytes" not in payload:
            raise RuntimeError(f"Unsupported capture payload at frame {step}: {payload!r}")

        out_path = output_dir / f"{args.prefix}_{step:0{digits}d}.png"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists and --overwrite is not set: {out_path}"
            )

        out_path.write_bytes(payload["bytes"])
        written.append(out_path)
        _print_progress_bar(step + 1, args.frames, label="Capturing frames")

    return written


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
        return Path(args.output)
    return Path(args.output_dir) / f"{args.prefix}_video.mp4"


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
    output = _infer_video_output_path(args).expanduser()
    if output.suffix.lower() != ".mp4":
        raise ValueError("This command currently supports MP4 output only.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists and --overwrite is not set: {output}")

    frame_paths = run_capture_brain_forceps_pngs(args)
    try:
        _run_ffmpeg(frame_paths, fps=args.fps, output=output)
    except Exception:
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
