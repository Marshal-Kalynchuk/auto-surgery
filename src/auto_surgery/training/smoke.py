"""Blackwell / CUDA smoke gate for training images."""

from __future__ import annotations

import sys
from typing import Any


def run_blackwell_smoke(*, skip_gpu: bool = False) -> dict[str, Any]:
    """Verify torch stack; exercise matmul, autocast bf16, and `torch.compile` when possible."""

    report: dict[str, Any] = {"python": sys.version}
    try:
        import torch
    except ImportError as e:
        raise ImportError("Smoke requires torch (train extra / training container).") from e

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if skip_gpu:
        report["skipped"] = "GPU exercises skipped"
        return report
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; run with --skip-gpu for import-only checks.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    report["gpu_name"] = props.name
    report["major"] = props.major
    report["minor"] = props.minor

    x = torch.randn(1024, 1024, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    z = (x @ y).sum()
    z.backward()
    report["fp32_matmul_backward_ok"] = True

    if torch.cuda.is_bf16_supported():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            a = torch.randn(512, 512, device=device)
            b = torch.randn(512, 512, device=device)
            _ = (a @ b).sum()
        report["bf16_autocast_ok"] = True
    else:
        report["bf16_autocast_ok"] = False

    try:
        m = torch.nn.Linear(64, 64).to(device)
        compiled = torch.compile(m)
        out = compiled(torch.randn(64, 64, device=device))
        report["torch_compile_ok"] = bool(out.numel())
    except Exception as exc:
        report["torch_compile_ok"] = False
        report["torch_compile_error"] = repr(exc)

    return report
