# auto-surgery

Phase 0 Python infrastructure: typed schemas, `Environment` protocol (sim/real), append-only Parquet logging with manifests, training dataset/checkpoint manifests, and Blackwell smoke tooling.

## Setup

```bash
uv sync --frozen --all-groups --extra train
```

## Checks

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run pyright
uv run pytest
```

## SOFA Stage-0 runtime runbook

Discover what the local runtime loader can see before executing non-stub runs:

```bash
uv run python -c "from auto_surgery.env.sofa import discover_sofa_runtime_contract; print(discover_sofa_runtime_contract())"
```

Run a bounded rollout and dataset manifest output using the staged harness:

```bash
uv run python - <<'PY'
from auto_surgery.training.sofa_smoke import run_sofa_rollout_dataset

root_uri = "file:///tmp/auto-surgery-sofa-smoke/"
ds = run_sofa_rollout_dataset(
    storage_root_uri=root_uri,
    case_id="stage0_case",
    session_id="stage0_session",
    sofa_scene_path="file:///tmp/sofa_scenes/needle_env.scene",
    fallback_to_stub=False,
    steps=64,
    seed=7,
)
print("dataset", ds.model_dump_json())
PY
```

If SOFA bindings are not yet wired in your environment, keep local development running with stub fallback:

```bash
uv run python - <<'PY'
from auto_surgery.training.sofa_smoke import run_sofa_rollout_dataset

root_uri = "file:///tmp/auto-surgery-sofa-smoke/"
ds = run_sofa_rollout_dataset(
    storage_root_uri=root_uri,
    case_id="stage0_case_stub",
    session_id="stage0_session_stub",
    fallback_to_stub=True,
    steps=32,
)
print("dataset", ds.model_dump_json())
PY
```

Run the full smoke (rollout -> dataset -> IDM -> pseudo-action) once SOFA or stub is selected:

```bash
uv run python - <<'PY'
from auto_surgery.training.sofa_smoke import run_sofa_smoke_pipeline

run_sofa_smoke_pipeline(
    out_root_uri="file:///tmp/auto-surgery-sofa-smoke/",
    case_id="sofa_case",
    session_id="sofa_session",
    derived_case_id="sofa_case_derived",
    derived_session_id="sofa_session_derived",
    fallback_to_stub=False,
    steps=24,
)
PY
```

## Validation checklist (Stage-0 handoff)

1. Runtime contract check:
   - `uv run python -c "from auto_surgery.env.sofa import discover_sofa_runtime_contract; print(discover_sofa_runtime_contract())"`
2. Smoke gate:
   - `uv run auto-surgery smoke --skip-gpu`
3. Lint + type checks:
   - `uv run ruff check src tests`
   - `uv run ruff format src tests`
   - `uv run pyright`
4. Contract/integration coverage:
   - `uv run pytest tests/env/test_contract.py tests/integration/test_sim_to_training.py tests/integration/test_idm_stage0.py`
5. SOFA or stub smoke dataset evidence:
   - Run `run_sofa_rollout_dataset(...)`
   - Ensure `frame_count_estimate` equals the requested rollout length.
   - Confirm session manifest path appears in returned `DatasetManifest`.
6. Training handoff evidence:
   - Keep the `train_idm` output metrics dictionary and `CheckpointManifest` on disk.
   - Compare source and derived datasets: frame count parity + aligned `frame_index`.
   - Keep command vectors/`command_echo` parity for the same joint key set (`j0` at Stage-0).

## CLI

```bash
uv run auto-surgery smoke --skip-gpu   # CPU-only import checks
uv run auto-surgery smoke              # full CUDA smoke when torch+GPU present
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture.
