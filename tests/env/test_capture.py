from __future__ import annotations

import math
import io
import pytest

import auto_surgery.env.capture as capture_module
from auto_surgery.config import load_scene_config
from auto_surgery.env.sofa_backend import _apply_tone_augmentation
from auto_surgery.env.sofa_rgb_native import attach_capture_camera
import auto_surgery.env.sofa_rgb_native as sofa_rgb_native
from auto_surgery.schemas.commands import Pose, Quaternion, Vec3
from auto_surgery.schemas.scene import VisualToneAugmentation
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.env.capture import SofaNativeRgbCapture
from PIL import Image
import numpy as np


class _RootForCamera:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def getObject(self, name: str):  # noqa: N802
        del name
        return None

    def addObject(self, *args: object, **kwargs: object) -> object:
        del args
        self.kwargs = kwargs
        return object()


def _pose_up_x_positive_90() -> tuple[float, float, float]:
    return (0.0, 0.0, 1.0)


def test_attach_capture_camera_uses_computed_up_and_fov(monkeypatch) -> None:
    root = _RootForCamera()
    pose = Pose(
        position=Vec3(x=1.0, y=2.0, z=3.0),
        rotation=Quaternion(w=0.7071067811865476, x=0.7071067811865476, y=0.0, z=0.0),
    )
    intrinsics = load_scene_config("configs/scenes/dejavu_brain.yaml").camera_intrinsics.model_copy(
        update={"fx": 900.0, "fy": 350.0, "width": 128, "height": 96}
    )

    monkeypatch.setattr(sofa_rgb_native, "_ensure_runtime_loaded", lambda: None)
    attach_capture_camera(
        root,
        width=64,
        height=48,
        camera_pose=pose,
        camera_intrinsics=intrinsics,
    )

    observed_up = tuple(float(value) for value in root.kwargs["up"])  # type: ignore[arg-type]
    expected_up = _pose_up_x_positive_90()
    assert observed_up[0] == pytest.approx(expected_up[0], abs=1e-12)
    assert observed_up[1] == pytest.approx(expected_up[1], abs=1e-12)
    assert observed_up[2] == pytest.approx(expected_up[2], abs=1e-12)
    expected_fov = 2.0 * math.degrees(math.atan(96.0 / (2.0 * 350.0)))
    assert math.isclose(float(root.kwargs["fieldOfView"]), expected_fov, rel_tol=0.0, abs_tol=1e-7)
    assert root.kwargs["widthViewport"] == 64
    assert root.kwargs["heightViewport"] == 48


def test_pre_init_hook_passes_camera_up_to_attach_capture_camera(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_attach_capture_camera(root_node: object, **kwargs: object) -> object:
        del root_node
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(capture_module, "attach_capture_camera", fake_attach_capture_camera)
    monkeypatch.setattr(capture_module, "validate_native_capture_runtime", lambda: None)

    scene = load_scene_config("configs/scenes/dejavu_brain.yaml").model_copy(
        update={
            "camera_extrinsics_scene": Pose(
                position=Vec3(x=0.0, y=1.0, z=2.0),
                rotation=Quaternion(w=0.7071067811865476, x=0.7071067811865476, y=0.0, z=0.0),
            ),
        }
    )
    capture = SofaNativeRgbCapture(width=64, height=64)
    capture.pre_init_hook(object(), EnvConfig(seed=3, scene=scene))

    captured_up = tuple(float(value) for value in captured["up"])  # type: ignore[arg-type]
    expected_up = _pose_up_x_positive_90()
    assert captured_up[0] == pytest.approx(expected_up[0], abs=1e-12)
    assert captured_up[1] == pytest.approx(expected_up[1], abs=1e-12)
    assert captured_up[2] == pytest.approx(expected_up[2], abs=1e-12)
    assert captured["camera_intrinsics"] == scene.camera_intrinsics


def _rgb_bytes(array: np.ndarray) -> bytes:
    with io.BytesIO() as output:
        Image.fromarray(array, mode="RGB").save(output, format="PNG")
        return output.getvalue()


def test_apply_tone_augmentation_changes_frame_bytes_for_non_identity() -> None:
    base_pixels = np.array(
        [
            [[10, 20, 30], [128, 64, 32]],
            [[16, 48, 80], [200, 180, 160]],
        ],
        dtype=np.uint8,
    )
    base_frame = _rgb_bytes(base_pixels)
    tone = VisualToneAugmentation(
        brightness_scale=1.25,
        contrast_scale=0.95,
        gamma=1.8,
        saturation_scale=1.1,
    )
    toned = _apply_tone_augmentation(base_frame, tone=tone)
    assert _apply_tone_augmentation(base_frame, tone=VisualToneAugmentation()) == base_frame
    assert toned != base_frame


def test_apply_tone_augmentation_sweep_produces_distinct_outputs() -> None:
    base_pixels = np.array(
        [
            [[24, 48, 72], [96, 64, 32]],
            [[192, 180, 168], [12, 240, 128]],
        ],
        dtype=np.uint8,
    )
    base_frame = _rgb_bytes(base_pixels)

    tones = [
        VisualToneAugmentation(),
        VisualToneAugmentation(brightness_scale=1.15, contrast_scale=1.0, gamma=1.0, saturation_scale=1.0),
        VisualToneAugmentation(brightness_scale=1.0, contrast_scale=0.85, gamma=1.35, saturation_scale=1.2),
        VisualToneAugmentation(brightness_scale=0.9, contrast_scale=1.05, gamma=0.9, saturation_scale=0.95),
    ]

    outputs = [_apply_tone_augmentation(base_frame, tone=tone) for tone in tones]
    assert outputs[0] == base_frame
    assert len(set(outputs[1:])) == len(outputs[1:])
