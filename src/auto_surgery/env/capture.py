"""Extensible capture modalities for SOFA (RGB now; stereo / depth later)."""

from __future__ import annotations

import io
from typing import Any, Protocol
import math

from PIL import Image

from auto_surgery.env.sofa_rgb_native import (
    attach_capture_camera,
    render_frame_to_rgb,
    validate_native_capture_runtime,
)
from auto_surgery.schemas.manifests import EnvConfig


def _pose_up_from_camera(pose: Any) -> tuple[float, float, float]:
    x = float(pose.rotation.x)
    y = float(pose.rotation.y)
    z = float(pose.rotation.z)
    w = float(pose.rotation.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return (0.0, 1.0, 0.0)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    up_x = 2.0 * x * y - 2.0 * z * w
    up_y = 1.0 - 2.0 * x * x - 2.0 * z * z
    up_z = 2.0 * y * z + 2.0 * x * w
    up_norm = math.sqrt(up_x * up_x + up_y * up_y + up_z * up_z)
    if up_norm <= 0.0:
        return (0.0, 1.0, 0.0)
    inv_up = 1.0 / up_norm
    return (up_x * inv_up, up_y * inv_up, up_z * inv_up)


class CaptureModality(Protocol):
    """One sensor modality that can be captured from a live SOFA scene graph."""

    def modality_id(self) -> str: ...

    def capture(self, *, root_node: Any, step_index: int) -> dict[str, Any]: ...


class SofaNativeRgbCapture:
    """Native SOFA RGB capture using the `OffscreenCamera` backend."""

    def __init__(self, width: int = 950, height: int = 700) -> None:
        validate_native_capture_runtime()
        self._width = width
        self._height = height

    def modality_id(self) -> str:
        return "rgb"

    def capture(self, *, root_node: Any, step_index: int) -> dict[str, Any]:
        rgb = render_frame_to_rgb(
            root_node,
            step_index=step_index,
            width=self._width,
            height=self._height,
        )
        with io.BytesIO() as buffer:
            Image.fromarray(rgb).save(buffer, format="PNG")
            return {
                "encoding": "image/png",
                "bytes": buffer.getvalue(),
            }

    def pre_init_hook(self, root_node: Any, config: EnvConfig) -> None:
        """Attach the OffscreenCamera before Sofa.Simulation.init(root) runs."""
        scene = config.scene
        intrinsics = scene.camera_intrinsics
        pose = scene.camera_extrinsics_scene
        attach_capture_camera(
            root_node,
            width=self._width if self._width else intrinsics.width,
            height=self._height if self._height else intrinsics.height,
            up=_pose_up_from_camera(pose),
            camera_pose=pose,
            camera_intrinsics=intrinsics,
        )


class StereoRgbStubCapture:
    def modality_id(self) -> str:
        return "stereo_rgb_stub"

    def capture(self, *, root_node: Any, step_index: int) -> dict[str, Any]:
        del root_node, step_index
        return {"implemented": False}


class DepthStubCapture:
    def modality_id(self) -> str:
        return "depth_stub"

    def capture(self, *, root_node: Any, step_index: int) -> dict[str, Any]:
        del root_node, step_index
        return {"implemented": False}


class SegmentationStubCapture:
    def modality_id(self) -> str:
        return "segmentation_stub"

    def capture(self, *, root_node: Any, step_index: int) -> dict[str, Any]:
        del root_node, step_index
        return {"implemented": False}


def default_captures(
    *,
    include_stereo_depth_stubs: bool = False,
) -> list[CaptureModality]:
    """Return the native RGB capture plus optional placeholder modalities."""

    out: list[CaptureModality] = [SofaNativeRgbCapture()]
    if include_stereo_depth_stubs:
        out.extend(
            [
                StereoRgbStubCapture(),
                DepthStubCapture(),
                SegmentationStubCapture(),
            ]
        )
    return out
