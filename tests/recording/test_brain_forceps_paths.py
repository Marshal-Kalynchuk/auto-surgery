from __future__ import annotations

import argparse
from pathlib import Path

from auto_surgery.recording import brain_forceps


def _video_namespace(**overrides: object) -> argparse.Namespace:
    values = {
        "output": None,
        "output_dir": Path("artifacts"),
        "prefix": "brain_forceps_sample",
        "master_seed": 999,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_infer_video_output_path_prefers_explicit_output() -> None:
    output = Path("outputs") / "custom.mp4"
    args = _video_namespace(output=output)

    assert brain_forceps._infer_video_output_path(args) == brain_forceps._resolve_from_repo_root(
        output
    )


def test_infer_video_output_path_uses_output_dir_by_default() -> None:
    output_dir = Path("artifacts") / "samples"
    master_seed = 321
    args = _video_namespace(output_dir=output_dir, prefix="run", master_seed=master_seed)

    expected = (
        brain_forceps._build_run_output_root(brain_forceps._resolve_from_repo_root(output_dir), master_seed)
        / "run_video.mp4"
    )
    assert brain_forceps._infer_video_output_path(args) == expected
