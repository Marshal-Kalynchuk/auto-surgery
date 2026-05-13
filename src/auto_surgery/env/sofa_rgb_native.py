"""SOFA-native RGB capture via `OffscreenCamera`.

This module intentionally fails fast when the offscreen capture runtime is
unavailable. The SOFA-native path is now a single `OffscreenCamera` flow for
headless RGB frame output.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import math
import io
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np
from PIL import Image

from auto_surgery.schemas.commands import Pose
from auto_surgery.schemas.sensors import CameraIntrinsics

_OFFSCREEN_CAMERA_NAME = "auto_surgery_offscreen_camera"
_FRAME_ROOT_DIR = Path(tempfile.gettempdir()) / "auto_surgery_sofa_offscreen_frames"
_runtime_loaded = False
_SOFA_MODULE: Any | None = None
_OffscreenCameraType: Any | None = None


class SofaNativeRenderError(RuntimeError):
    """Raised when SOFA-native rendering cannot initialize or run."""


def _ensure_frame_root_dir() -> None:
    try:
        _FRAME_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SofaNativeRenderError(
            "SOFA offscreen capture cache directory is not writable. "
            "Set TMPDIR or make the system temp directory writable."
        ) from exc


def _ensure_runtime_loaded() -> None:
    global _runtime_loaded, _SOFA_MODULE, _OffscreenCameraType
    if _runtime_loaded:
        return

    try:
        import Sofa
        import Sofa.Core  # noqa: F401  (imports bind classes)
        import Sofa.Simulation  # noqa: F401
        import SofaRuntime
    except ImportError as exc:  # pragma: no cover - runtime dependent
        raise SofaNativeRenderError(
            "SOFA Python bindings are missing. Run infra/sofa/setup_sofa_conda.sh and "
            "`source .env.sofa` before initializing SOFA-native capture."
        ) from exc

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        raise SofaNativeRenderError(
            "Activate the `sofa-env` conda environment before using SOFA-native rendering."
        )

    plugin_lib_dir = os.path.join(conda_prefix, "plugins", "SofaOffscreenCamera", "lib")
    try:
        SofaRuntime.PluginRepository.addFirstPath(plugin_lib_dir)
        SofaRuntime.importPlugin("SofaOffscreenCamera")
    except Exception as exc:
        raise SofaNativeRenderError(
            "Failed to register the SofaOffscreenCamera plugin. Run infra/sofa/setup_sofa_conda.sh "
            "and `source .env.sofa` to bootstrap the plugin."
        ) from exc

    try:
        from SofaOffscreenCamera import OffscreenCamera as _ImportedOffscreenCamera  # noqa: F401

        _OffscreenCameraType = _ImportedOffscreenCamera
    except ImportError as exc:
        raise SofaNativeRenderError(
            "SofaOffscreenCamera bindings could not be imported. Ensure "
            "infra/sofa/setup_sofa_conda.sh ran successfully and `.env.sofa` is sourced."
        ) from exc

    _SOFA_MODULE = Sofa
    _runtime_loaded = True


def validate_native_capture_runtime() -> None:
    """Validate that the SOFA runtime and OffscreenCamera are available."""
    _ensure_runtime_loaded()

    if not callable(_OffscreenCameraType):
        raise SofaNativeRenderError(
            "OffscreenCamera bindings are unavailable. Re-run infra/sofa/setup_sofa_conda.sh "
            "and `source .env.sofa`."
        )

    if not (os.environ.get("QT_QPA_PLATFORM") or os.environ.get("DISPLAY")):
        raise SofaNativeRenderError(
            "Set QT_QPA_PLATFORM=offscreen for headless rendering, or source .env.sofa."
        )

    _ensure_frame_root_dir()
    if not callable(attach_capture_camera):
        raise SofaNativeRenderError(
            "attach_capture_camera is unavailable. The capture module may be corrupted."
        )


def _get_camera(root_node: Any) -> Any | None:
    get_object = getattr(root_node, "getObject", None)
    if not callable(get_object):
        return None
    return get_object(_OFFSCREEN_CAMERA_NAME)


def _pose_position(pose: Pose) -> tuple[float, float, float]:
    return (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )


def _pose_look_at(
    pose: Pose,
    *,
    look_distance: float = 1.0,
) -> tuple[float, float, float]:
    position = _pose_position(pose)
    if look_distance <= 0.0:
        return position

    x = float(pose.rotation.x)
    y = float(pose.rotation.y)
    z = float(pose.rotation.z)
    w = float(pose.rotation.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return (position[0], position[1], position[2] + look_distance)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # Approximate forward vector for local +Z in SOFA/OpenGL convention.
    forward_x = 2.0 * (x * z + w * y)
    forward_y = 2.0 * (y * z - w * x)
    forward_z = 1.0 - 2.0 * (x * x + y * y)
    forward_norm = math.sqrt(forward_x * forward_x + forward_y * forward_y + forward_z * forward_z)
    if forward_norm <= 0.0:
        return (position[0], position[1], position[2] + look_distance)

    inv_forward = 1.0 / forward_norm
    return (
        position[0] + look_distance * forward_x * inv_forward,
        position[1] + look_distance * forward_y * inv_forward,
        position[2] + look_distance * forward_z * inv_forward,
    )


def _coerce_intrinsics_for_capture(camera_intrinsics: CameraIntrinsics | None) -> CameraIntrinsics:
    if camera_intrinsics is None:
        return CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=950, height=700)
    width = int(camera_intrinsics.width)
    height = int(camera_intrinsics.height)
    return CameraIntrinsics(
        fx=float(camera_intrinsics.fx),
        fy=float(camera_intrinsics.fy),
        cx=float(camera_intrinsics.cx),
        cy=float(camera_intrinsics.cy),
        width=max(1, width),
        height=max(1, height),
    )


def _shift_principal_point(
    rgb: np.ndarray,
    *,
    width: int,
    height: int,
    camera_intrinsics: CameraIntrinsics,
    background_rgb: Sequence[float | int],
) -> np.ndarray:
    if width <= 0 or height <= 0:
        return rgb
    if camera_intrinsics.fx <= 1.0 or camera_intrinsics.fy <= 1.0:
        return rgb

    center_x = float(width) / 2.0
    center_y = float(height) / 2.0
    if (
        math.isclose(camera_intrinsics.cx, center_x)
        and math.isclose(camera_intrinsics.cy, center_y)
    ):
        return rgb
    if camera_intrinsics.cx == 0.0 and camera_intrinsics.cy == 0.0:
        # Older defaults in schema can represent "unset" principal point.
        return rgb

    shift_x = int(round(float(camera_intrinsics.cx) - center_x))
    shift_y = int(round(float(camera_intrinsics.cy) - center_y))
    if shift_x == 0 and shift_y == 0:
        return rgb

    output = np.zeros_like(rgb)
    bg_channels = []
    for index in range(3):
        value = float(background_rgb[index] if index < len(background_rgb) else 0.0)
        if value <= 1.0:
            value *= 255.0
        bg_channels.append(int(min(255.0, max(0.0, round(value)))))
    output[:, :] = np.array((bg_channels[0], bg_channels[1], bg_channels[2]), dtype=np.uint8)

    source_start_x = 0
    source_start_y = 0
    dest_start_x = shift_x
    dest_start_y = shift_y
    if shift_x < 0:
        source_start_x = -shift_x
        dest_start_x = 0
    if shift_y < 0:
        source_start_y = -shift_y
        dest_start_y = 0
    overlap_x = min(width - source_start_x, width - dest_start_x)
    overlap_y = min(height - source_start_y, height - dest_start_y)
    if overlap_x <= 0 or overlap_y <= 0:
        return output
    source_end_x = source_start_x + overlap_x
    source_end_y = source_start_y + overlap_y
    dest_end_x = dest_start_x + overlap_x
    dest_end_y = dest_start_y + overlap_y
    output[dest_start_y:dest_end_y, dest_start_x:dest_end_x] = rgb[
        source_start_y:source_end_y, source_start_x:source_end_x
    ]
    return output


def compensate_principal_point(
    frame_bytes: bytes,
    *,
    width: int,
    height: int,
    camera_intrinsics: CameraIntrinsics | None,
    background_rgb: Sequence[float | int] = (0, 0, 0),
) -> bytes:
    if camera_intrinsics is None:
        return frame_bytes
    try:
        with io.BytesIO(frame_bytes) as input_buffer:
            with Image.open(input_buffer) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    except Exception:
        return frame_bytes
    compensated = _shift_principal_point(
        rgb,
        width=width,
        height=height,
        camera_intrinsics=camera_intrinsics,
        background_rgb=background_rgb,
    )
    if compensated is rgb:
        return frame_bytes
    with io.BytesIO() as output_buffer:
        Image.fromarray(compensated).save(output_buffer, format="PNG")
        return output_buffer.getvalue()


def attach_capture_camera(
    root_node: Any,
    *,
    width: int | None = None,
    height: int | None = None,
    position: tuple[float, float, float] | None = None,
    look_at: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    camera_pose: Pose | None = None,
    camera_intrinsics: CameraIntrinsics | None = None,
) -> Any:
    """Attach the OffscreenCamera before Sofa.Simulation.init(root) runs.
    
    orientation is a quaternion in (x, y, z, w) order matching SOFA's Quat<SReal>.
    """
    _ensure_runtime_loaded()
    resolved_intrinsics = _coerce_intrinsics_for_capture(camera_intrinsics)
    resolved_width = int(width) if width is not None else int(resolved_intrinsics.width)
    resolved_height = int(height) if height is not None else int(resolved_intrinsics.height)
    if resolved_width <= 0:
        resolved_width = 950
    if resolved_height <= 0:
        resolved_height = 700
    resolved_position = position
    resolved_look_at = look_at
    resolved_orientation = orientation
    if camera_pose is not None:
        pose_position = _pose_position(camera_pose)
        if resolved_position is None:
            resolved_position = pose_position
        if resolved_look_at is None:
            resolved_look_at = _pose_look_at(camera_pose)
        if resolved_orientation is None:
            resolved_orientation = (
                float(camera_pose.rotation.x),
                float(camera_pose.rotation.y),
                float(camera_pose.rotation.z),
                float(camera_pose.rotation.w),
            )
    if resolved_position is None:
        resolved_position = (0.0, 30.0, 90.0)
    if resolved_look_at is None:
        resolved_look_at = (0.0, 0.0, 0.0)
    if resolved_orientation is None:
        resolved_orientation = (0.0, 0.0, 0.0, 1.0)
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("render width/height must be positive ints.")
    if resolved_intrinsics.fy <= 0.0:
        raise ValueError("CameraIntrinsics.fy must be finite and positive.")
    field_of_view = 2.0 * math.degrees(
        math.atan(float(resolved_intrinsics.height) / (2.0 * float(resolved_intrinsics.fy)))
    )

    existing_camera = _get_camera(root_node)
    if existing_camera is not None:
        return existing_camera

    add_object = getattr(root_node, "addObject", None)
    if not callable(add_object):
        raise SofaNativeRenderError(
            "SOFA root node is missing addObject(). Offscreen rendering cannot attach "
            "OffscreenCamera for headless capture."
        )

    try:
        camera = add_object(
            "OffscreenCamera",
            name=_OFFSCREEN_CAMERA_NAME,
            widthViewport=resolved_width,
            heightViewport=resolved_height,
            position=list(resolved_position),
            orientation=list(resolved_orientation),
            zNear=0.1,
            zFar=1000.0,
            fieldOfView=float(field_of_view),
            projectionType=0,
            computeZClip=False,
            save_frame_before_first_step=False,
            save_frame_after_each_n_steps=0,
        )
    except Exception as exc:
        raise SofaNativeRenderError(
            "Failed to attach OffscreenCamera. Ensure SofaOffscreenCamera is installed "
            "via scripts/setup_sofa_conda.sh and `.env.sofa` is sourced."
        ) from exc

    if camera is None:
        raise SofaNativeRenderError(
            "OffscreenCamera addObject returned None. Verify the plugin installation."
        )

    return camera


def _load_frame_image(frame_path: Path, expected_width: int, expected_height: int) -> np.ndarray:
    if not frame_path.exists() or frame_path.stat().st_size == 0:
        raise SofaNativeRenderError("Captured frame file is missing or empty.")

    try:
        with Image.open(frame_path) as img:
            rgb = img.convert("RGB")
            if rgb.size != (expected_width, expected_height):
                raise SofaNativeRenderError(
                    "Captured frame size "
                    f"{rgb.size} != expected {(expected_width, expected_height)}."
                )
            return np.array(rgb, dtype=np.uint8)
    except SofaNativeRenderError:
        raise
    except Exception as exc:
        raise SofaNativeRenderError("Failed to read captured frame image.") from exc
    finally:
        with contextlib.suppress(OSError):
            frame_path.unlink()


def render_frame_to_rgb(
    root_node: Any,
    *,
    step_index: int,
    width: int = 950,
    height: int = 700,
) -> np.ndarray:
    """Render one RGB frame from a live SOFA root node via OffscreenCamera."""
    _ensure_runtime_loaded()
    if step_index < 0:
        raise ValueError("step_index must be a non-negative int.")
    if width <= 0 or height <= 0:
        raise ValueError("render width/height must be positive ints.")

    camera = _get_camera(root_node)
    if camera is None:
        raise SofaNativeRenderError(
            "OffscreenCamera not attached. Call attach_capture_camera(root, ...) BEFORE "
            "Sofa.Simulation.init(root), or wire SofaNativeRgbCapture.pre_init_hook into "
            "SofaEnvironment(pre_init_hooks=[...])."
        )

    _ensure_frame_root_dir()

    sofa = _SOFA_MODULE
    if sofa is None:
        raise SofaNativeRenderError("SOFA runtime is not initialized.")

    sofa.Simulation.updateVisual(root_node)

    frame_path = _FRAME_ROOT_DIR / f"frame_{os.getpid()}_{id(root_node)}_{step_index:06d}.png"
    try:
        camera.save_frame(str(frame_path))
    except Exception as exc:
        raise SofaNativeRenderError(
            "Failed to execute OffscreenCamera.save_frame(). Set QT_QPA_PLATFORM=offscreen "
            "for headless rendering and ensure `.env.sofa` is sourced."
        ) from exc

    rgb = _load_frame_image(frame_path, width, height)
    # OffscreenCamera's native capture appears vertically inverted relative to
    # the reference DejaVu assets. Flip for visual/pose consistency.
    return np.flipud(rgb)
