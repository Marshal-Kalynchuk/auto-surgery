"""Small end-to-end non-stub SOFA scene smoke script for the POC."""

from __future__ import annotations

import tempfile
from pathlib import Path

from auto_surgery.env.capture import SofaNativeRgbCapture
from auto_surgery.env.sofa import SofaEnvironment
from auto_surgery.env.sofa_tools import build_forceps_action_applier
from auto_surgery.env.sofa_scenes.dejavu_paths import resolve_brain_forceps_scene_path
from auto_surgery.schemas.commands import ControlMode, RobotCommand
from auto_surgery.schemas.commands import Twist, Vec3
from auto_surgery.schemas.manifests import EnvConfig

def run_dejavu_forceps_smoke(
    *, scene_path: str | None = None, steps: int = 2
) -> list[Path]:
    if steps <= 0:
        raise ValueError("steps must be greater than 0 for screenshot capture.")
    scene_path = resolve_brain_forceps_scene_path(scene_path)

    action_applier = build_forceps_action_applier()
    capture = SofaNativeRgbCapture()
    outputs: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="sofa-forceps-poc-", dir="/tmp") as tmp_root:
        env = SofaEnvironment(
            sofa_scene_path=scene_path,
            step_dt=0.01,
            action_applier=action_applier,
            pre_init_hooks=[capture.pre_init_hook],
        )
        env.reset(EnvConfig(seed=7, domain_randomization={}))
        output_dir = Path(tmp_root)
        root_node = env.sofa_scene_root

        for step_index in range(steps):
            action = RobotCommand(
                timestamp_ns=1_000_000 + step_index,
                cycle_id=step_index,
                control_mode=ControlMode.CARTESIAN_TWIST,
                cartesian_twist=Twist(
                    linear=Vec3(x=0.05 * step_index, y=0.0, z=0.0),
                    angular=Vec3(x=0.0, y=0.0, z=0.0),
                ),
                enable=True,
                source="scripted",
            )
            env.step(action)
            screenshot_path = output_dir / f"frame_{step_index:03d}.png"
            payload = capture.capture(root_node=root_node, step_index=step_index)
            if payload.get("encoding") != "image/png" or "bytes" not in payload:
                raise RuntimeError(
                    "Native capture returned unsupported payload for "
                    f"step {step_index}: {payload!r}"
                )
            screenshot_path.write_bytes(payload["bytes"])
            assert screenshot_path.exists(), f"Screenshot path missing: {screenshot_path}"
            assert screenshot_path.stat().st_size > 0, f"Screenshot empty: {screenshot_path}"
            outputs.append(screenshot_path)

        return outputs


def main() -> None:
    run_dejavu_forceps_smoke(steps=2)


if __name__ == "__main__":
    raise SystemExit(
        "Direct execution of this module is deprecated. "
        "Use: uv run auto-surgery sofa-forceps-smoke."
    )
