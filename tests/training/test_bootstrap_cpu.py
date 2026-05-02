from __future__ import annotations

from auto_surgery.training.bootstrap import (
    run_m1_tiny_overfit,
    run_m2_contrastive_stub,
    run_m3_imitation_stub_from_frames,
)


def test_m1_runs_on_cpu() -> None:
    loss = run_m1_tiny_overfit(steps=20, device="cpu")
    assert loss == loss  # finite
    assert loss >= 0.0


def test_m2_runs_on_cpu() -> None:
    loss = run_m2_contrastive_stub(steps=10, device="cpu")
    assert isinstance(loss, float)


def test_m3_runs_on_cpu() -> None:
    frames = [[0.1] * 8, [0.2] * 8]
    actions = [[0.0] * 4, [1.0] * 4]
    loss = run_m3_imitation_stub_from_frames(frames, actions, steps=50, device="cpu")
    assert loss < 1.0
