# Parallel SOFA rollouts (multiprocessing + Slurm)

## In-process pool

`src/auto_surgery/training/parallel_rollouts.py` exposes `run_rollouts_pool`, which maps each kwargs dict to `run_sofa_rollout_dataset` inside a `multiprocessing.Pool`.

Each job dict must match the keyword arguments of `run_sofa_rollout_dataset` (for example `storage_root_uri`, `case_id`, `session_id`, `steps`, `scene_config`, `capture_modalities`, …).

## Slurm array jobs

Use one array task per episode and invoke the Typer CLI:

```bash
uv run python -m auto_surgery.training.run_one_episode \
  --storage-root-uri file:///scratch/$USER/sofa-runs/ \
  --case-id brain_poc \
  --session-id ep_${SLURM_ARRAY_TASK_ID} \
  --sofa-scene-path /path/to/brain_dejavu_forceps_poc.scn \
  --stub false \
  --rgb \
  --steps 128
```

Tune `--processes` only for the multiprocessing helper; Slurm typically uses one CPU per array task instead of an inner pool.

## Environment

Always `source .env.sofa` (or export `SOFA_ROOT` consistently with `SOFA_HOME`) before native SOFA runs so plugin loading (for example `sofa-gl` and `SofaOffscreenCamera`) has a valid `SOFA_PLUGIN_PATH` and headless Qt is set with `QT_QPA_PLATFORM=offscreen` when `DISPLAY` is unavailable.
