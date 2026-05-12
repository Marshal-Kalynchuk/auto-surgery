#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly SOFA_ENV_NAME="${SOFA_ENV_NAME:-sofa-env}"
readonly SOFA_CHANNEL="${SOFA_CHANNEL:-https://prefix.dev/sofa-framework}"
readonly CONDA_PACKAGES=(sofa-app sofa-python3 sofa-gl)

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH." >&2
  echo "Install Miniforge first: https://github.com/conda-forge/miniforge" >&2
  exit 1
fi

CONDA_EXE="$(command -v conda)"

eval "$("${CONDA_EXE}" shell.bash hook)"

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${SOFA_ENV_NAME}"; then
  "${CONDA_EXE}" create -n "${SOFA_ENV_NAME}" -y
fi

conda activate "${SOFA_ENV_NAME}"

conda install -y \
  -c "${SOFA_CHANNEL}" \
  -c conda-forge \
  "${CONDA_PACKAGES[@]}"

conda install -y \
  -c conda-forge \
  qt6-main \
  cmake \
  make \
  pkg-config

pip install -e "${REPO_ROOT}"

readonly PLUGIN_BUILD_ROOT="${PLUGIN_BUILD_ROOT:-$HOME/build}"
readonly PLUGIN_SRC="${PLUGIN_BUILD_ROOT}/SofaOffscreenCamera"
readonly PLUGIN_INSTALL_PREFIX="${CONDA_PREFIX}/plugins/SofaOffscreenCamera"
readonly PLUGIN_REPO="${SOFA_OFFSCREEN_CAMERA_REPO:-https://github.com/jnbrunet/SofaOffscreenCamera.git}"
readonly PLUGIN_REF="${SOFA_OFFSCREEN_CAMERA_REF:-v25.06}"
readonly PLUGIN_LIB="${PLUGIN_INSTALL_PREFIX}/lib/libSofaOffscreenCamera.so"
readonly PYTHON_VER="$("${CONDA_PREFIX}/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
readonly PLUGIN_PYTHON_SITE="${PLUGIN_INSTALL_PREFIX}/lib/${PYTHON_VER}/site-packages"
readonly CONDA_PYTHON_SITE="${CONDA_PREFIX}/lib/${PYTHON_VER}/site-packages"

if "${CONDA_PREFIX}/bin/python" - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("SofaOffscreenCamera") is not None else 1)
PY
then
  echo "SofaOffscreenCamera is already importable."
else
  mkdir -p "${PLUGIN_BUILD_ROOT}"
  if [ -d "${PLUGIN_SRC}/.git" ]; then
    git -C "${PLUGIN_SRC}" fetch --depth 1 origin "${PLUGIN_REF}"
    git -C "${PLUGIN_SRC}" checkout FETCH_HEAD
  else
    rm -rf "${PLUGIN_SRC}"
    git clone --branch "${PLUGIN_REF}" --depth 1 "${PLUGIN_REPO}" "${PLUGIN_SRC}"
  fi

  if [ ! -d "${PLUGIN_SRC}" ]; then
    echo "Failed to clone SofaOffscreenCamera plugin source." >&2
    exit 1
  fi

  readonly BUILD_DIR="${PLUGIN_SRC}/build"
  rm -rf "${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"

  (
    cd "${BUILD_DIR}"
    cmake .. \
      -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
      -DCMAKE_INSTALL_PREFIX="${PLUGIN_INSTALL_PREFIX}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE="${CONDA_PREFIX}/bin/python"
    cmake --build . -j"$(nproc)"
    cmake --install .
  )
fi

if [ ! -f "${PLUGIN_LIB}" ]; then
  echo "SofaOffscreenCamera library is still missing at ${PLUGIN_LIB}." >&2
  echo "The plugin build may have failed." >&2
  exit 1
fi

cat <<EOF > "${REPO_ROOT}/.env.sofa"
if [ -z "\${CONDA_PREFIX:-}" ]; then
  echo ".env.sofa requires an active conda env (CONDA_PREFIX is unset)." >&2
  return 1 2>/dev/null || exit 1
fi

export SOFA_HOME="\$CONDA_PREFIX"
export SOFA_ROOT="\$CONDA_PREFIX"
export SOFA_PLUGIN_PATH="\$CONDA_PREFIX/plugins/SofaOffscreenCamera/lib\${SOFA_PLUGIN_PATH:+:\$SOFA_PLUGIN_PATH}"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PATH="\$CONDA_PREFIX/bin\${PATH:+:\$PATH}"

if [ -z "\${QT_QPA_PLATFORM:-}" ] && [ -z "\${DISPLAY:-}" ]; then
  export QT_QPA_PLATFORM=offscreen
fi

export SOFA_GL_REQUIRED=1
EOF

cat <<'EOF'

SOFA Conda environment setup is complete.

Next:
  conda activate ${SOFA_ENV_NAME}
  source .env.sofa
  QT_QPA_PLATFORM=offscreen uv run python -c "from auto_surgery.env.sofa_rgb_native import validate_native_capture_runtime; validate_native_capture_runtime(); print('offscreen capture contract ok')"

EOF
