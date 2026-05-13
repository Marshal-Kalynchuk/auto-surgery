# auto-surgery

Phase 0 Python infrastructure: typed schemas, `Environment` protocol (sim/real), append-only Parquet logging with manifests, training dataset/checkpoint manifests, and Blackwell smoke tooling.

## Setup

```bash
uv sync --frozen --all-groups --extra train
```

If you update forceps collision geometry, regenerate the deterministic proxy first:

```bash
AUTO_SURGERY_DEJAVU_ROOT=/path/to/dejavu \
  uv run --group prep python scripts/prep_forceps_collision_meshes.py --force --print-hash
```

The generated artifact and contract live at `assets/forceps/shaft_tip_collision.obj` and
`assets/forceps/dejavu_default.yaml`.

If you need a real SOFA runtime, bootstrap it first via
[docs/SOFA_INSTALLATION.md](docs/SOFA_INSTALLATION.md) before non-stub stage-0 runs.

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

If needed, verify SOFA runtime and headless capture runtime:

```bash
source .env.sofa
uv run python -c "from auto_surgery.env.sofa_rgb_native import validate_native_capture_runtime; validate_native_capture_runtime(); print('offscreen camera available')"
```

The SOFA native path used by this repo for image capture is the `SofaOffscreenCamera` plugin
(bootstrapped via `infra/sofa/setup_sofa_conda.sh`) and configured by `.env.sofa`.

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
    steps=64,
    seed=7,
)
print("dataset", ds.model_dump_json())
PY
```

To validate with a different scene input during setup:

```bash
uv run python - <<'PY'
from auto_surgery.training.sofa_smoke import run_sofa_rollout_dataset

root_uri = "file:///tmp/auto-surgery-sofa-smoke/"
ds = run_sofa_rollout_dataset(
    storage_root_uri=root_uri,
    case_id="stage0_case_alt",
    session_id="stage0_session_alt",
    sofa_scene_path="file:///tmp/auto-surgery-sofa-scene-alternate.scn",
    steps=32,
)
print("dataset", ds.model_dump_json())
PY
```

Run the full smoke (rollout -> dataset -> IDM -> pseudo-action):

```bash
uv run python - <<'PY'
from auto_surgery.training.sofa_smoke import run_sofa_smoke_pipeline

run_sofa_smoke_pipeline(
    out_root_uri="file:///tmp/auto-surgery-sofa-smoke/",
    case_id="sofa_case",
    session_id="sofa_session",
    derived_case_id="sofa_case_derived",
    derived_session_id="sofa_session_derived",
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
5. SOFA smoke dataset evidence:
   - Run `run_sofa_rollout_dataset(...)`
   - Ensure `frame_count_estimate` equals the requested rollout length.
   - Confirm session manifest path appears in returned `DatasetManifest`.
6. Training handoff evidence:
   - Keep the `train_idm` output metrics dictionary and `CheckpointManifest` on disk.
   - Compare source and derived datasets: frame count parity + aligned `frame_index`.
   - Keep commanded/executed action parity for the same joint key set (`j0` at Stage-0).

## CLI entrypoints (single surface)

Preferred usage routes all scripts through `auto-surgery`:

```bash
uv run auto-surgery smoke --skip-gpu
uv run auto-surgery capture-brain-forceps-video --qgl-view <path> --ticks 180
uv run auto-surgery capture-brain-forceps-pngs --qgl-view <path> --ticks 180
uv run auto-surgery run-one-episode --storage-root-uri file:///tmp/auto-surgery-sofa-smoke/ --case-id demo --session-id s1
uv run auto-surgery train-idm --dataset-manifest-uri file:///tmp/auto-surgery-sofa-smoke/cases/demo/sessions/s1/manifest.json --out-ckpt-uri /tmp/idm.pt
uv run auto-surgery extract-pseudo-actions --dataset-manifest-uri ... --idm-ckpt-uri ... --out-root-uri file:///tmp/auto-surgery-sofa-smoke/ --out-case-id demo_derived --out-session-id s2
uv run auto-surgery render-rollout-preview --storage-root-uri file:///tmp/auto-surgery-sofa-smoke/ --case-id demo_derived --session-id s2 --output /tmp/preview.gif
uv run auto-surgery sofa-forceps-smoke --steps 2
```

Legacy `python -m` entrypoints are intentionally deprecated in favor of this unified surface.

## Captures

All capture commands require the SOFA conda environment. Use the helper script:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
if [[ -z "${AUTO_SURGERY_DEJAVU_ROOT:-}" ]]; then
  echo "AUTO_SURGERY_DEJAVU_ROOT is required for DejaVu capture."
  exit 1
fi

# Capture brain-forceps rollout as mp4 in one command
bash infra/sofa/with-sofa.sh uv run auto-surgery capture-brain-forceps-video \
  --qgl-view "$AUTO_SURGERY_DEJAVU_ROOT/scenes/brain/brain.scn.qglviewer.view" \
  --ticks 180 \
  --scene-config "$REPO_ROOT/configs/scenes/dejavu_brain.yaml" \
  --motion-config "$REPO_ROOT/configs/motion/default.yaml" \
  --fps 30 \
  --output "$REPO_ROOT/artifacts/brain_forceps.mp4"

bash infra/sofa/with-sofa.sh uv run auto-surgery capture-brain-forceps-pngs \
  --qgl-view "$AUTO_SURGERY_DEJAVU_ROOT/scenes/brain/brain.scn.qglviewer.view" \
  --ticks 180 \
  --scene-config "$REPO_ROOT/configs/scenes/dejavu_brain.yaml" \
  --motion-config "$REPO_ROOT/configs/motion/default.yaml" \
  --output-dir "$REPO_ROOT/artifacts" \
  --prefix brain_forceps_sample
```

The `with-sofa.sh` script handles conda activation and `.env.sofa` setup automatically.

For smoke testing or validation of the SOFA runtime itself:

```bash
bash infra/sofa/with-sofa.sh python - <<'PY'
from auto_surgery.env.sofa_rgb_native import validate_native_capture_runtime
validate_native_capture_runtime()
print("offscreen camera available")
PY
```

See [docs/library/ARCHITECTURE.md](docs/library/ARCHITECTURE.md) for system architecture.
