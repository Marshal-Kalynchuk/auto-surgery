"""Render a rollout into a short preview video from saved `rgb` blobs."""

from __future__ import annotations

import argparse
import io
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import fsspec
from PIL import Image


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


def _session_rgb_frame_uris(
    *, storage_root_uri: str, case_id: str, session_id: str
) -> list[str]:
    fs, root = fsspec.url_to_fs(storage_root_uri)
    frame_dir = f"{root}/cases/{case_id}/sessions/{session_id}/blobs/rgb"
    if not fs.exists(frame_dir):
        return []
    names = sorted(fs.ls(frame_dir, detail=False))
    return [
        item
        for item in names
        if str(item).lower().endswith(".png") and "blobs/rgb" in str(item)
    ]


def _read_frames(frame_uris: list[str], fs: Any) -> list[Image.Image]:
    images: list[Image.Image] = []
    for index, uri in enumerate(frame_uris):
        with fs.open(uri, "rb") as fp:
            images.append(Image.open(io.BytesIO(fp.read())).convert("RGB"))
        _print_progress_bar(index + 1, len(frame_uris), label="Reading frames")
    return images


def _write_mp4_with_ffmpeg(frames: list[Image.Image], fps: float, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sofa-rollout-preview-") as tmpdir:
        tmp_root = Path(tmpdir)
        for idx, frame in enumerate(frames):
            frame_path = tmp_root / f"frame_{idx:06d}.png"
            frame.save(frame_path)
            _print_progress_bar(idx + 1, len(frames), label="Preparing MP4 frames")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is not installed on this host.")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(tmp_root / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        subprocess.run(cmd, check=True, text=True, capture_output=True)


def _write_gif(frames: list[Image.Image], fps: float, output: Path) -> None:
    duration_ms = int(math.ceil(1000.0 / fps))
    first = frames[0]
    rest = frames[1:]
    first.save(
        output,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def build_preview(
    *,
    storage_root_uri: str,
    case_id: str,
    session_id: str,
    duration_seconds: float,
    output_path: Path,
    prefer_mp4: bool = True,
) -> None:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0.")
    fs, _ = fsspec.url_to_fs(storage_root_uri)
    frame_uris = _session_rgb_frame_uris(
        storage_root_uri=storage_root_uri,
        case_id=case_id,
        session_id=session_id,
    )
    if not frame_uris:
        raise FileNotFoundError(
            f"No rgb frames found for case={case_id} session={session_id} under {storage_root_uri}"
        )

    frames = _read_frames(frame_uris, fs)
    if not frames:
        raise RuntimeError("Found frame paths but could not decode any frames.")

    fps = max(1.0, len(frames) / duration_seconds)
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".mp4":
        if not prefer_mp4:
            output_path = output_path.with_suffix(".gif")
        else:
            _write_mp4_with_ffmpeg(frames, fps=fps, output=output_path)
            return
    if output_path.suffix.lower() == ".gif":
        _write_gif(frames, fps=fps, output=output_path)
        return
    if output_path.suffix.lower() in {".mp4", ".gif"}:
        raise ValueError(f"Unsupported output extension: {output_path.suffix}")

    # Backward-compatible behavior: produce GIF when no extension is provided.
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".gif")
    _write_gif(frames, fps=fps, output=output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a rollout preview clip from rgb blob frames."
    )
    parser.add_argument("--storage-root-uri", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=10.0,
        help="Target output duration in seconds.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/sofa_rollout_preview.gif",
        help="Output path (supports .mp4 or .gif).",
    )
    parser.add_argument(
        "--prefer-mp4",
        action="store_true",
        help="Allow MP4 when output extension is .mp4; otherwise fallback to GIF.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    build_preview(
        storage_root_uri=args.storage_root_uri,
        case_id=args.case_id,
        session_id=args.session_id,
        duration_seconds=args.duration_sec,
        output_path=output,
        prefer_mp4=args.prefer_mp4,
    )
    print(f"preview written: {output}")


if __name__ == "__main__":
    raise SystemExit(
        "Direct execution of this module is deprecated. "
        "Use: uv run auto-surgery render-rollout-preview."
    )
