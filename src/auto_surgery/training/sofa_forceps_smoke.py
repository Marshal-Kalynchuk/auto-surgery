"""Small end-to-end non-stub SOFA scene smoke script for the POC."""

from __future__ import annotations

import tempfile
from pathlib import Path

from auto_surgery.env.capture import SofaNativeRgbCapture
from auto_surgery.env.sofa import SofaEnvironment
from auto_surgery.env.sofa_tools import build_forceps_action_applier
from auto_surgery.schemas.commands import RobotCommand
from auto_surgery.schemas.manifests import EnvConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FORCEPS_SCENE = (
    _REPO_ROOT / "src" / "auto_surgery" / "env" / "sofa_scenes" / "brain_dejavu_forceps_poc.scn"
)


def run_dejavu_forceps_smoke(*, scene_path: str, steps: int = 2) -> list[Path]:
    if steps <= 0:
        raise ValueError("steps must be greater than 0 for screenshot capture.")

    action_applier = build_forceps_action_applier()
    capture = SofaNativeRgbCapture()
    outputs: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="sofa-forceps-poc-", dir="/tmp") as tmp_root:
        env = SofaEnvironment(
            sofa_scene_path=scene_path,
            fallback_to_stub=False,
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
                joint_positions={"j0": 0.05 * step_index},
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

        print(str(outputs[0]))
        return outputs


def main() -> None:
    scene_path = _DEFAULT_FORCEPS_SCENE
    run_dejavu_forceps_smoke(scene_path=str(scene_path))


if __name__ == "__main__":
    main()
