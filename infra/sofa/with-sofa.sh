#!/usr/bin/env bash
# with-sofa.sh: Run a command inside the SOFA conda environment with proper setup.
#
# Usage:
#   bash infra/sofa/with-sofa.sh auto-surgery capture-brain-forceps-video ...
#   bash infra/sofa/with-sofa.sh python -c "from auto_surgery.env.sofa_rgb_native import ..."
#
# This script:
#   1. Activates the sofa-env conda environment
#   2. Sources .env.sofa to configure SOFA plugin paths
#   3. Runs your command in that context
#   4. Cleans up on exit

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly SOFA_ENV_NAME="${SOFA_ENV_NAME:-sofa-env}"

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Install Miniforge or activate your conda installation." >&2
  exit 1
fi

# Check if sofa-env exists
if ! conda env list | awk '{print $1}' | grep -qx "${SOFA_ENV_NAME}"; then
  echo "Conda environment '${SOFA_ENV_NAME}' not found." >&2
  echo "Run 'bash ${SCRIPT_DIR}/setup_sofa_conda.sh' first." >&2
  exit 1
fi

# Initialize conda (required for activation to work in non-interactive contexts)
eval "$(conda shell.bash hook)"

# Activate the environment and run the command
conda activate "${SOFA_ENV_NAME}"
source "${REPO_ROOT}/.env.sofa"

# Make uv treat the active sofa-env conda environment as the project venv,
# instead of building a parallel `.venv` under the repo (which would not have
# SOFA's Python or the offscreen camera plugin available).
#
# - UV_PROJECT_ENVIRONMENT points uv at the conda env directory.
# - UV_PYTHON pins the interpreter to the conda env's python so any tool that
#   asks uv to discover a python (e.g. `uv pip install`) picks the right one.
#
# pyproject.toml sets `[tool.uv] managed = false`, so `uv run` does not try to
# sync the env from a lockfile; deps are installed into sofa-env directly via
# `pip install -e .` in `setup_sofa_conda.sh`.
export UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX}"
export UV_PYTHON="${CONDA_PREFIX}/bin/python"

exec "$@"
