"""Small end-to-end non-stub SOFA scene smoke script for the POC."""

from __future__ import annotations

import tempfile
from pathlib import Path

from auto_surgery.env.capture import SofaNativeRgbCapture
from auto_surgery.env.sofa import SofaEnvironment
from auto_surgery.config import load_motion_config, load_scene_config
from auto_surgery.motion import SurgicalMotionGenerator
from auto_surgery.schemas.manifests import EnvConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_CONFIG_PATH = PROJECT_ROOT / "configs" / "scenes" / "dejavu_brain.yaml"
DEFAULT_MOTION_CONFIG_PATH = (PROJECT_ROOT / "configs" / "motion" / "default.yaml").resolve()
DEFAULT_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "sofa-forceps-smoke"


def _resolve_path(path: str | None, *, fallback: Path) -> str:
    if path is None:
        candidate = fallback
    else:
        candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate.resolve())


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="sofa-forceps-smoke-", dir=DEFAULT_OUTPUT_ROOT)).resolve()

    resolved = Path(output_dir).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def run_dejavu_forceps_smoke(
    *,
    scene_path: str | None = None,
    steps: int = 2,
    scene_config_path: str | None = None,
    motion_config_path: str | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    if steps <= 0:
        raise ValueError("steps must be greater than 0 for screenshot capture.")
    scene_config = load_scene_config(_resolve_path(scene_config_path, fallback=DEFAULT_SCENE_CONFIG_PATH))
    motion_config = load_motion_config(_resolve_path(motion_config_path, fallback=DEFAULT_MOTION_CONFIG_PATH)).model_copy(
        update={"seed": 7}
    )
    capture = SofaNativeRgbCapture()
    resolved_scene_path = scene_path
    if resolved_scene_path is None:
        resolved_scene_path = str(scene_config.tissue_scene_path)
    env_kwargs = dict(
        scene_config=scene_config,
        step_dt=0.01,
        pre_init_hooks=[capture.pre_init_hook],
    )
    env_kwargs["sofa_scene_path"] = resolved_scene_path
    outputs: list[Path] = []
    output_dir = _resolve_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SofaEnvironment(**env_kwargs)
    last_step = env.reset(EnvConfig(seed=7, scene=scene_config))
    generator = SurgicalMotionGenerator(motion_config, scene_config)
    command = generator.reset(last_step)
    root_node = env.sofa_scene_root

    for step_index in range(steps):
        last_step = env.step(command)
        command = generator.next_command(last_step)
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
    generator.finalize(last_step)

    return outputs


def main() -> None:
    run_dejavu_forceps_smoke(steps=2)


if __name__ == "__main__":
    raise SystemExit(
        "Direct execution of this module is deprecated. "
        "Use: uv run auto-surgery sofa-forceps-smoke."
    )
