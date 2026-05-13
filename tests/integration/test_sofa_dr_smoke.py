from __future__ import annotations

from io import BytesIO
import math

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from auto_surgery.config import load_motion_config, load_scene_config
from auto_surgery.env.capture import SofaNativeRgbCapture
from auto_surgery.env.sofa import SofaEnvironment, SofaNotIntegratedError
from auto_surgery.env.sofa_discovery import resolve_sofa_runtime_import_candidates
from auto_surgery.motion.generator import SurgicalMotionGenerator
from auto_surgery.randomization import sample_episode
from auto_surgery.randomization.presets import load_randomization_preset
from auto_surgery.randomization.scn_template import render_scene_template
from auto_surgery.schemas.manifests import EnvConfig
from auto_surgery.schemas.scene import MeshPerturbation
from tests.integration.sofa_integration_observability import (
    find_brain_volume_mechanical_object,
    flatten_vec3_positions,
    peak_vertex_displacement_magnitude,
)


_SCENE_YAML = Path(__file__).resolve().parents[2] / "configs" / "scenes" / "dejavu_brain.yaml"
_MOTION_YAML = Path(__file__).resolve().parents[2] / "configs" / "motion" / "default.yaml"

_SOFA_DR_MASTER_SEED = 1312
_NUM_EPISODES = 3
_EPISODE_TICKS = 200
_TICK_CAPTURE = 100
_MIN_VISUAL_MSE = 1.0e-3
_MAX_ABS_WRENCH_COMPONENT = 1.0e6


def _require_integration_prereqs() -> None:
    _, mod = resolve_sofa_runtime_import_candidates()
    if mod is None:
        pytest.skip("SofaPython3 runtime not available for SOFA domain-randomization smoke.")

    try:
        from auto_surgery.env.sofa_rgb_native import validate_native_capture_runtime
        from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_dejavu_root
    except ImportError as exc:
        pytest.skip(f"Cannot import SOFA capture helpers: {exc}")

    try:
        validate_native_capture_runtime()
    except Exception as exc:
        pytest.skip(f"Sofa native capture runtime unavailable: {exc}")

    try:
        dejavu_root = resolve_dejavu_root()
    except RuntimeError as exc:
        pytest.skip(f"AUTO_SURGERY_DEJAVU_ROOT is unavailable: {exc}")

    if not (dejavu_root / "scenes/brain/brain.py").is_file():
        pytest.skip(f"Missing expected DejaVu brain scene entry point: {dejavu_root / 'scenes/brain/brain.py'}")

    try:
        import trimesh  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("trimesh is not installed; skipping SOFA DR smoke test.")


def _decode_png(payload: bytes, *, width: int, height: int) -> np.ndarray:
    with BytesIO(payload) as buffer:
        with Image.open(buffer) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8)
    if array.shape[:2] != (height, width):
        raise RuntimeError(
            f"Unexpected frame shape {array.shape}; expected {(height, width, 3)}."
        )
    return array


def _frame_mse(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Frame shapes differ; cannot compute MSE.")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return float(np.mean(delta * delta))


def _wrench_norm(tool_state: object) -> float:
    wrench = getattr(tool_state, "wrench", None)
    if wrench is None:
        return 0.0
    x = float(getattr(wrench, "x", 0.0))
    y = float(getattr(wrench, "y", 0.0))
    z = float(getattr(wrench, "z", 0.0))
    return math.sqrt(x * x + y * y + z * z)


def _run_smoke_episode(
    *,
    episode_seed: int,
    base_scene_path: Path,
    capture: SofaNativeRgbCapture,
    spec,
    width: int,
    height: int,
    ticks: int,
) -> tuple[np.ndarray, list[float], float]:
    scene = spec.scene
    if any(bool(spec.sample_record.get(axis)) for axis in spec.sample_record.model_dump()):
        try:
            from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_dejavu_root
        except RuntimeError as exc:
            raise RuntimeError(f"Cannot resolve DejaVu root for scene rendering: {exc}")
        scene = scene.model_copy(
            update={"tissue_scene_path": render_scene_template(scene, dejavu_root=resolve_dejavu_root())}
        )

    env = SofaEnvironment(scene_config=scene, pre_init_hooks=[capture.pre_init_hook])
    try:
        reset = env.reset(
            EnvConfig(
                seed=episode_seed,
                scene=scene,
                control_rate_hz=250.0,
                frame_rate_hz=250.0,
            )
        )
        generator = SurgicalMotionGenerator(spec.motion, scene)
        command = generator.reset(reset)

        brain_mo = find_brain_volume_mechanical_object(env.sofa_scene_root)
        if brain_mo is None:
            raise SofaNotIntegratedError("Could not resolve brain MechanicalObject for deformation checks.")
        baseline_positions = flatten_vec3_positions(brain_mo)

        final_wrench_norms: list[float] = []
        captured_frame: np.ndarray | None = None
        for tick in range(ticks):
            step = env.step(command)
            wrench_norm = _wrench_norm(step.sensors.tool)
            if not math.isfinite(wrench_norm):
                raise SofaNotIntegratedError(f"Non-finite wrench norm at tick {tick}: {wrench_norm}")
            if abs(wrench_norm) > _MAX_ABS_WRENCH_COMPONENT:
                raise SofaNotIntegratedError(
                    f"Unusually large wrench norm at tick {tick}: {wrench_norm} (possible FEM blowup)."
                )
            final_wrench_norms.append(wrench_norm)

            if tick == _TICK_CAPTURE:
                payload = capture.capture(root_node=env.sofa_scene_root, step_index=step.sim_step_index)
                if payload.get("encoding") != "image/png" or "bytes" not in payload:
                    raise SofaNotIntegratedError(f"Invalid capture payload at tick {tick}: {payload!r}")
                captured_frame = _decode_png(payload["bytes"], width=width, height=height)

            command = generator.next_command(step)

        generator.finalize(step)
        if captured_frame is None:
            raise SofaNotIntegratedError(f"Did not capture tick {_TICK_CAPTURE} during episode {episode_seed}.")

        final_positions = flatten_vec3_positions(brain_mo)
        peak_displacement = peak_vertex_displacement_magnitude(baseline_positions, final_positions)
        return captured_frame, final_wrench_norms, float(peak_displacement)
    finally:
        env.close()


def test_domain_randomization_smoke_minimal_has_visual_diversity_and_stable_rollouts() -> None:
    _require_integration_prereqs()

    scene = load_scene_config(_SCENE_YAML)
    motion = load_motion_config(_MOTION_YAML)
    preset = load_randomization_preset("minimal")

    rng = np.random.default_rng(_SOFA_DR_MASTER_SEED)
    episode_seeds = rng.integers(0, 2**64, size=_NUM_EPISODES, dtype=np.uint64)
    frames: list[np.ndarray] = []
    displacements: list[float] = []
    all_wrenches: list[list[float]] = []
    capture = SofaNativeRgbCapture(width=192, height=128)

    for index in range(_NUM_EPISODES):
        spec = sample_episode(scene, motion, preset, episode_seed=int(episode_seeds[index]))
        frame, wrench_norms, peak_disp = _run_smoke_episode(
            episode_seed=int(episode_seeds[index]),
            base_scene_path=_SCENE_YAML,
            capture=capture,
            spec=spec,
            width=192,
            height=128,
            ticks=_EPISODE_TICKS,
        )

        if len(wrench_norms) != _EPISODE_TICKS:
            pytest.fail(f"Episode {index} produced {len(wrench_norms)} wrench samples, expected {_EPISODE_TICKS}.")
        if peak_disp <= 0.0:
            pytest.fail(f"Episode {index} produced no brain vertex displacement.")

        frames.append(frame)
        displacements.append(peak_disp)
        all_wrenches.append(wrench_norms)

    for left_index in range(len(frames)):
        for right_index in range(left_index + 1, len(frames)):
            diff = _frame_mse(frames[left_index], frames[right_index])
            assert diff > _MIN_VISUAL_MSE, (
                f"Episode pair ({left_index}, {right_index}) had low visual diversity at tick {_TICK_CAPTURE}: "
                f"MSE={diff:.6g}"
            )

    # Sanity guard: episodes should not all collapse into near-identical contact/wrench profiles.
    for index, norms in enumerate(all_wrenches):
        if max(norms) <= 0.0:
            pytest.fail(f"Episode {index} produced zero wrench norm across all ticks (unexpected contact silence).")
