"""CLI entry for Slurm array tasks: one `run_sofa_rollout_dataset` invocation per process."""

from __future__ import annotations

import json

import typer

from auto_surgery.env.capture import default_captures
from auto_surgery.schemas.manifests import SceneConfig
from auto_surgery.schemas.scene import ToolSpec
from auto_surgery.training.sofa_smoke import run_sofa_rollout_dataset

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    storage_root_uri: str = typer.Option(..., help="fsspec root URI ending with '/'."),
    case_id: str = typer.Option(...),
    session_id: str = typer.Option(...),
    sofa_scene_path: str | None = typer.Option(
        None,
        help="Optional `.scn` path; when omitted, `scene_id` selects the Python factory.",
    ),
    steps: int = typer.Option(64),
    seed: int = typer.Option(7),
    rgb: bool = typer.Option(False, help="Persist native SOFA RGB blobs."),
    scene_id: str = typer.Option("dejavu_brain"),
    tool_id: str = typer.Option("dejavu_forceps"),
) -> None:
    """Materialize one dataset manifest + optional RGB blobs."""

    captures = (
        default_captures(include_stereo_depth_stubs=False)
        if rgb
        else []
    )
    scene_cfg = SceneConfig(
        scene_id=scene_id,
        tool=ToolSpec(tool_id=tool_id),
    )
    ds = run_sofa_rollout_dataset(
        storage_root_uri=storage_root_uri,
        case_id=case_id,
        session_id=session_id,
        sofa_scene_path=sofa_scene_path,
        scene_config=scene_cfg,
        sofa_backend_factory=None,
        steps=steps,
        seed=seed,
        capture_modalities=captures if rgb else None,
    )
    typer.echo(json.dumps(ds.model_dump(), indent=2))


def cli() -> None:
    app()


if __name__ == "__main__":
    raise SystemExit(
        "Direct execution of this module is deprecated. "
        "Use: uv run auto-surgery run-one-episode."
    )
