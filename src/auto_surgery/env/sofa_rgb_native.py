"""SOFA-native RGB capture via `OffscreenCamera`.

This module intentionally fails fast when the offscreen capture runtime is
unavailable. The SOFA-native path is now a single `OffscreenCamera` flow for
headless RGB frame output.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


def attach_capture_camera(
    root_node: Any,
    *,
    width: int = 950,
    height: int = 700,
    # Keep camera parameters consistent with the scene's InteractiveCamera.
    # brain_dejavu_forceps_poc.scn uses: position="0 30 90" lookAt="0 0 0".
    position: tuple[float, float, float] = (0.0, 30.0, 90.0),
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    """Attach the OffscreenCamera before Sofa.Simulation.init(root) runs."""
    _ensure_runtime_loaded()
    if width <= 0 or height <= 0:
        raise ValueError("render width/height must be positive ints.")

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
            widthViewport=width,
            heightViewport=height,
            position=list(position),
            lookAt=list(look_at),
            zNear=0.1,
            zFar=1000.0,
            fieldOfView=45.0,
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
