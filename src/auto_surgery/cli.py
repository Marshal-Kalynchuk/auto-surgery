"""Typer CLI for smoke tests and bootstrap sanity jobs."""

from __future__ import annotations

import json

import typer

from auto_surgery.training.bootstrap import run_m1_tiny_overfit, run_m2_contrastive_stub

app = typer.Typer(no_args_is_help=True)


@app.command()
def smoke(
    skip_gpu: bool = typer.Option(False, help="Skip CUDA exercises (CI / laptops without GPU)."),
) -> None:
    """Run Blackwell/CUDA smoke gate from `training.smoke`."""

    from auto_surgery.training.smoke import run_blackwell_smoke

    typer.echo(json.dumps(run_blackwell_smoke(skip_gpu=skip_gpu), indent=2))


@app.command()
def bootstrap_m1() -> None:
    """Run tiny overfit loop (requires torch)."""

    loss = run_m1_tiny_overfit()
    typer.echo(json.dumps({"final_loss": loss}, indent=2))


@app.command()
def bootstrap_m2() -> None:
    """Run contrastive stub (requires torch)."""

    loss = run_m2_contrastive_stub()
    typer.echo(json.dumps({"final_loss": loss}, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
