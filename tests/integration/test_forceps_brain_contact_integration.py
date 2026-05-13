"""§9.3 integration (spec 2026-05-12): forceps + brain penalty contact telemetry.

Skipped automatically when SofaPython3 or DejaVu assets are unavailable. When run,
loads ``dejavu_brain`` via the Python scene registry (``create_brain_scene``),
exercises ``SofaEnvironment`` + default forceps factories, then asserts ordered
witnesses requested in the approved design doc:
``in_contact`` False→True, growth of ``‖wrench‖``, and measurable brain FEM motion.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from auto_surgery.config import load_scene_config
from auto_surgery.env.sofa import SofaEnvironment, SofaNotIntegratedError
from auto_surgery.env.sofa_discovery import resolve_sofa_runtime_import_candidates
from auto_surgery.schemas.commands import ControlMode, RobotCommand, Twist, Vec3
from auto_surgery.schemas.manifests import EnvConfig

from tests.integration.sofa_integration_observability import (
    find_brain_volume_mechanical_object,
    flatten_vec3_positions,
    peak_vertex_displacement_magnitude,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCENE_YAML_DEFAULT = _PROJECT_ROOT / "configs" / "scenes" / "dejavu_brain.yaml"


def _require_integration_prereqs(scene_yaml: Path = _SCENE_YAML_DEFAULT) -> None:
    _, mod = resolve_sofa_runtime_import_candidates()
    if mod is None:
        pytest.skip("SofaPython3 runtime not available for integration tests.")

    try:
        from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_dejavu_root

        root = resolve_dejavu_root()
    except RuntimeError:
        pytest.skip("AUTO_SURGERY_DEJAVU_ROOT is unset; cannot assemble DejaVu brain scene.")

    brain_py = (root / "scenes/brain/brain.py").expanduser().resolve()
    if not brain_py.is_file():
        pytest.skip(f"Missing expected DejaVu brain scene controller: {brain_py}")

    if not scene_yaml.is_file():
        pytest.skip(f"Scene config YAML not readable: {scene_yaml}")


def _wrench_norm(tool_state: object) -> float:
    wrench = getattr(tool_state, "wrench", None)
    if wrench is None:
        return 0.0
    fx = float(getattr(wrench, "x"))
    fy = float(getattr(wrench, "y"))
    fz = float(getattr(wrench, "z"))
    return math.sqrt(fx * fx + fy * fy + fz * fz)


def _median(samples: list[float]) -> float:
    if not samples:
        raise ValueError("median of empty sequence")
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def _sliding_window_medians(samples: list[float], window_size: int) -> list[float]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(samples) < window_size:
        return []
    return [_median(list(samples[i : i + window_size])) for i in range(len(samples) - window_size + 1)]


def test_forceps_brain_contact_wrench_and_tissue_motion(
    *,
    contact_ticks: int = 340,
    min_brain_displacement: float = 0.015,
    min_wrench_growth_ratio: float = 1.35,
    early_window_ticks: int = 40,
    late_window_ticks: int = 40,
    linear_drive: float = -18.0,
) -> None:
    """Deterministic penetration along local -Z twist; observes contact+FEM deformation.

    Acceptance (mirrors §9.3 wording):
      * ``in_contact`` transitions from ``False`` to ``True`` during the rollout.
      * ``‖wrench‖`` rises while pressing — sliding medians right after transition vs end of
        rollout tolerate brief FEM chatter.
      * At least one brain grid vertex shifts by ``min_brain_displacement`` scene units
        relative to the FEM configuration captured immediately post-reset.
    """
    _require_integration_prereqs()

    scene_config = load_scene_config(_SCENE_YAML_DEFAULT)

    twist = Twist(
        linear=Vec3(x=0.0, y=0.0, z=float(linear_drive)),
        angular=Vec3(x=0.0, y=0.0, z=0.0),
    )

    env: SofaEnvironment
    try:
        env = SofaEnvironment(scene_config=scene_config)
        rollout_cfg = EnvConfig(seed=9042, control_rate_hz=250.0, frame_rate_hz=30.0, scene=scene_config)
        env.reset(rollout_cfg)

        brain_mo = find_brain_volume_mechanical_object(env.sofa_scene_root)
        if brain_mo is None:
            pytest.fail("Brain FEM MechanicalObject missing from assembled scene.")

        baseline_positions = flatten_vec3_positions(brain_mo)
        wrench_records: list[float] = []
        contact_states: list[bool] = []

        for tick in range(int(contact_ticks)):
            step = env.step(
                RobotCommand(
                    timestamp_ns=1_000_000 + tick,
                    cycle_id=int(tick),
                    control_mode=ControlMode.CARTESIAN_TWIST,
                    cartesian_twist=twist,
                    enable=True,
                    tool_jaw_target=None,
                    source="integration_forceps_penetration",
                )
            )
            tool_state = step.sensors.tool
            wrench_records.append(_wrench_norm(tool_state))
            contact_states.append(bool(tool_state.in_contact))

        assert contact_states, "No wrench/contact telemetry was recorded."
        assert not contact_states[0], "Expected initial forceps contact state to be False."

        transition_ticks = [
            idx for idx in range(1, len(contact_states)) if not contact_states[idx - 1] and contact_states[idx]
        ]
        if not transition_ticks:
            pytest.fail(
                "Expected ``tool.in_contact`` to transition False→True while driving into tissue "
                "(no contact transition observed)."
            )
        transition_index = transition_ticks[0]

        if transition_index + early_window_ticks > len(wrench_records):
            pytest.fail("Rollout ended before capturing an early-contact wrench baseline.")

        post_contact = wrench_records[transition_index:]
        early_windows = _sliding_window_medians(post_contact, early_window_ticks)
        late_windows = _sliding_window_medians(post_contact[early_window_ticks:], late_window_ticks)

        if not early_windows:
            pytest.fail("Insufficient early-contact wrench samples to compute baseline window.")
        if not late_windows:
            pytest.fail("Insufficient late-contact wrench samples to compute growth window.")

        early_med = early_windows[0]
        late_med = max(late_windows)
        growth_ratio = late_med / early_med if early_med > 1e-12 else pytest.fail(
            "Early-contact wrench baseline is unexpectedly ~0."
        )

        assert growth_ratio >= min_wrench_growth_ratio, (
            f"Expected median ‖wrench‖ to grow (early={early_med:.6g}, late={late_med:.6g}, "
            f"ratio={growth_ratio:.3f}); check penalty tuning or commanded penetration depth."
        )

        final_xyz = flatten_vec3_positions(brain_mo)
        peak_disp = peak_vertex_displacement_magnitude(baseline_positions, final_xyz)
        assert peak_disp >= min_brain_displacement, (
            f"Expected brain FEM deformation ≥ {min_brain_displacement} scene units "
            f"after deterministic pressing; peak vertex motion was {peak_disp:.6g}."
        )
    except SofaNotIntegratedError as exc:
        pytest.fail(f"Sofa integration failed during forceps rollout: {exc}")
    finally:
        if "env" in locals():
            env.close()
