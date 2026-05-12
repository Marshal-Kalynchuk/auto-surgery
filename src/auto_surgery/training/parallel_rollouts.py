"""Multiprocessing helpers for SOFA rollouts (Slurm-friendly batching)."""

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import Pool
from typing import Any

from auto_surgery.schemas.manifests import DatasetManifest
from auto_surgery.training.sofa_smoke import run_sofa_rollout_dataset


def _episode_entry(args: tuple[dict[str, Any], int]) -> DatasetManifest:
    kwargs, _worker_index = args
    del _worker_index
    return run_sofa_rollout_dataset(**kwargs)


def run_rollouts_pool(
    jobs: Sequence[dict[str, Any]],
    *,
    processes: int | None = None,
    pool_factory: Any = None,
) -> list[DatasetManifest]:
    """Execute ``run_sofa_rollout_dataset`` jobs in parallel using ``multiprocessing.Pool``.

    Each element of ``jobs`` is a kwargs dict for :func:`run_sofa_rollout_dataset`.
    """

    if not jobs:
        return []
    factory = pool_factory or Pool
    indexed = [(dict(job), idx) for idx, job in enumerate(jobs)]
    with factory(processes=processes) as pool:
        return list(pool.map(_episode_entry, indexed))
