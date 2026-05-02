from __future__ import annotations

from auto_surgery.training.smoke import run_blackwell_smoke


def test_smoke_skip_gpu_reports_torch() -> None:
    report = run_blackwell_smoke(skip_gpu=True)
    assert "torch" in report
    assert report["cuda_available"] in (True, False)
