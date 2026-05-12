# SOFA Installation & POC Runtime Notes

This repository uses **SOFA** (Simulation Open Framework Architecture) via the adapter in `src/auto_surgery/env/sofa.py`.
The Stage-0 workflow steps SOFA scenes through the SOFA runtime for deterministic
dataset generation and IDM smoke coverage.

## 0. One-time SOFA environment setup

Install SOFA packages from Prefix.dev using a dedicated local `sofa-env` conda environment.
The project also requires headless rendering support from `SofaOffscreenCamera`, which is built from source because the default conda stack does not include it.

- Install Miniforge first if needed (for example, from https://github.com/conda-forge/miniforge).  
  The repo's runtime loader expects a Conda environment and will look for `CONDA_PREFIX` when setting `SOFA_HOME`.

From the repository root, create and activate a dedicated environment, then run your preferred SOFA bootstrap steps:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

conda create -n sofa-env -y
conda activate sofa-env
bash infra/sofa/setup_sofa_conda.sh
```

A bootstrap helper typically performs:

- `sofa-app`, `sofa-python3`, and `sofa-gl` from Prefix.dev
- `qt6-main`, `cmake`, `make`, and `pkg-config` into the active conda env
- this repo in editable mode (`pip install -e .`)
- `SofaOffscreenCamera` plugin/build artifacts into `$CONDA_PREFIX/plugins/SofaOffscreenCamera`
- `.env.sofa` exports (including `SOFA_HOME`, `SOFA_ROOT`, `SOFA_PLUGIN_PATH`, and Python path)

If needed, install the bundled plugin set used by the project POC runs:

```bash
conda install -y \
  --channel https://prefix.dev/sofa-framework \
  --channel conda-forge \
  sofa-app \
  sofa-python3 \
  sofa-stlib \
  sofa-modelorderreduction \
  sofa-beamadapter \
  sofa-softrobots \
  sofa-cosserat \
  sofa-gl
```

If available, this repo also includes a helper file `.env.sofa` to set runtime
environment variables (`SOFA_HOME`, `SOFA_ROOT`, `PYTHONPATH`, etc.) after activating
your conda environment.

The `sofa-gl` package and `SofaOffscreenCamera` are required for native rendering support used by the
`OffscreenCamera` capture path.

Recommended:

```bash
cd "$REPO_ROOT"
source .env.sofa
```

Quick smoke verification (SOFA 25.06+ defaults):

```bash
cd "$REPO_ROOT"
source .env.sofa
uv run python -c "from auto_surgery.env.sofa_rgb_native import validate_native_capture_runtime; validate_native_capture_runtime(); print('offscreen camera available')"
```

## 1. Verify SOFA runtime bindings are importable

Run:

```bash
cd "$REPO_ROOT"
uv run python -c "from auto_surgery.env.sofa import discover_sofa_runtime_contract; print(discover_sofa_runtime_contract())"
```

You should see a resolved module name (typically `Sofa`) and a non-empty resolved module path.

If this fails, ensure SOFA Python bindings are installed and your environment has them on `PYTHONPATH` (use `source .env.sofa`).

Verify the native capture stack is importable:

```bash
cd "$REPO_ROOT"
uv run python -c "from auto_surgery.env.sofa_rgb_native import validate_native_capture_runtime; validate_native_capture_runtime(); print('sofa offscreen capture ok')"
```

## 2. Acquire DejaVu scenes (brain)

For the Phase-1 / POC brain scene we use your DejaVu checkout (downloaded by you).

Set `AUTO_SURGERY_DEJAVU_ROOT` to your local DejaVu checkout:

```bash
if [[ -z "${AUTO_SURGERY_DEJAVU_ROOT:-}" ]]; then
  echo "AUTO_SURGERY_DEJAVU_ROOT is required for non-stub DejaVu runs."
  exit 1
fi
```

- `$AUTO_SURGERY_DEJAVU_ROOT/scenes/brain/brain.scn`
- `$AUTO_SURGERY_DEJAVU_ROOT/scenes/brain/brain.py` (requires SofaPython3 plugin support)

This repo also includes a local wrapper scene specifically for the forceps POC:

- `src/auto_surgery/env/sofa_scenes/brain_dejavu_forceps_poc.scn`

That wrapper is a regular SOFA XML scene and is the recommended entry point for non-stub POC runs.

## 3. Non-stub smoke: load + step + (optional) screenshot

### 3.1 Minimal load/step dataset generation (no RGB capture yet)

```bash
cd "$REPO_ROOT"
uv run python - <<PY
from auto_surgery.training.sofa_smoke import run_sofa_rollout_dataset

root_uri = "file:///tmp/auto-surgery-sofa-smoke/"
run_sofa_rollout_dataset(
    storage_root_uri=root_uri,
    case_id="dejavu_case",
    session_id="dejavu_session",
    sofa_scene_path="$REPO_ROOT/src/auto_surgery/env/sofa_scenes/brain_dejavu_forceps_poc.scn",
    steps=8,
    seed=7,
)
print("ok")
PY
```

### 3.2 Forceps RGB frame smoke (offscreen capture)

This smoke test writes **real PNG files** under `/tmp` by capturing native SOFA RGB frames via `OffscreenCamera`.

```bash
cd "$REPO_ROOT"
source .env.sofa
uv run python - <<PY
from auto_surgery.training.sofa_forceps_smoke import run_dejavu_forceps_smoke

run_dejavu_forceps_smoke(steps=1)
PY
```

If screenshot capture fails:

- Confirm native capture runtime:
  - `uv run python -c "from auto_surgery.env.sofa_rgb_native import validate_native_capture_runtime; validate_native_capture_runtime(); print('ok')"`
- Confirm `sofa-gl` and OpenGL dependencies are installed in the active environment.
- Confirm you can run an environment bootstrap:
  - `source .env.sofa`

## 4. Alternate smoke scene handles

To run against alternate scene handles during setup or experimentation:

```bash
cd "$REPO_ROOT"
uv run python - <<'PY'
from auto_surgery.training.sofa_smoke import run_sofa_rollout_dataset

root_uri = "file:///tmp/auto-surgery-sofa-smoke/"
run_sofa_rollout_dataset(
    storage_root_uri=root_uri,
    case_id="sofa_case_alt",
    session_id="sofa_session_alt",
    sofa_scene_path="file:///tmp/auto-surgery-sofa-scene-alternate.scn",
    steps=12,
)
print("ok")
PY
```

