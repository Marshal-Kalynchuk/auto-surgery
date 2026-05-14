#!/usr/bin/env bash
# Run capture-brain-forceps-video for many master seeds via with-sofa.sh + uv.
#
# Usage:
#   bash infra/sofa/batch-capture-brain-forceps-video.sh --range START END [opts] -- [capture args...]
#   bash infra/sofa/batch-capture-brain-forceps-video.sh --seeds 1,5,10 [opts] -- [capture args...]
#
# Options before "--":
#   --continue-on-error  Run all seeds; exit 1 if any run failed.
#   --range START END    Inclusive integer seed range.
#   --seeds LIST         Comma-separated and/or whitespace-separated integers.
#
# Do not pass --master-seed in the capture args; this script sets it per iteration.
#
# Example:
#   bash infra/sofa/batch-capture-brain-forceps-video.sh --range 0 9 -- \
#     --ticks 200 --fps 10 --output-dir artifacts/eval --overwrite

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly WITH_SOFA="${REPO_ROOT}/infra/sofa/with-sofa.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  batch-capture-brain-forceps-video.sh (--range START END | --seeds SEEDS) [options] [-- CAPTURE_ARGS...]

Exactly one of:
  --range START END     Inclusive seed range (integers).
  --seeds SEEDS         Comma and/or whitespace-separated integer seeds.

Options:
  --continue-on-error   Run every seed even if one fails; exit non-zero if any failed.
  -h, --help            Show this help.

After an optional "--", remaining arguments are passed to each
  uv run auto-surgery capture-brain-forceps-video
invocation (do not include --master-seed; it is injected).

Example:
  bash infra/sofa/batch-capture-brain-forceps-video.sh --range 10 12 -- \
    --ticks 200 --fps 10 --output-dir artifacts/batch --overwrite
EOF
}

continue_on_error=false
range_start=""
range_end=""
seeds_literal=""
extra_args=()
seed_mode=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    --continue-on-error)
      continue_on_error=true
      shift
      ;;
    --range)
      if [[ $# -lt 3 ]]; then
        echo "error: --range requires START and END" >&2
        exit 1
      fi
      if [[ "${seed_mode}" == list ]]; then
        echo "error: use only one of --range or --seeds" >&2
        exit 1
      fi
      if [[ "${seed_mode}" == range ]]; then
        echo "error: --range was given more than once" >&2
        exit 1
      fi
      seed_mode=range
      range_start="$2"
      range_end="$3"
      shift 3
      ;;
    --seeds)
      if [[ $# -lt 2 ]]; then
        echo "error: --seeds requires a seed list" >&2
        exit 1
      fi
      if [[ "${seed_mode}" == range ]]; then
        echo "error: use only one of --range or --seeds" >&2
        exit 1
      fi
      if [[ "${seed_mode}" == list ]]; then
        echo "error: --seeds was given more than once" >&2
        exit 1
      fi
      seed_mode=list
      seeds_literal="$2"
      shift 2
      ;;
    *)
      echo "error: unexpected argument: $1 (use -- before capture flags)" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${seed_mode}" ]]; then
  echo "error: specify --range START END or --seeds LIST" >&2
  usage
  exit 1
fi

seeds=()
if [[ "${seed_mode}" == range ]]; then
  if ! [[ "${range_start}" =~ ^-?[0-9]+$ && "${range_end}" =~ ^-?[0-9]+$ ]]; then
    echo "error: --range START and END must be integers (got: ${range_start} ${range_end})" >&2
    exit 1
  fi
  if ((range_start > range_end)); then
    echo "error: --range START must be <= END (got: ${range_start} ${range_end})" >&2
    exit 1
  fi
  for ((s = range_start; s <= range_end; s++)); do
    seeds+=("${s}")
  done
else
  normalized="${seeds_literal//,/ }"
  read -r -a raw_tokens <<< "${normalized}" || true
  if [[ ${#raw_tokens[@]} -eq 0 ]]; then
    echo "error: --seeds produced an empty list" >&2
    exit 1
  fi
  for tok in "${raw_tokens[@]}"; do
    if [[ -z "${tok}" ]]; then
      continue
    fi
    if ! [[ "${tok}" =~ ^-?[0-9]+$ ]]; then
      echo "error: invalid seed token (expected integer): ${tok}" >&2
      exit 1
    fi
    seeds+=("${tok}")
  done
  if [[ ${#seeds[@]} -eq 0 ]]; then
    echo "error: --seeds produced an empty list" >&2
    exit 1
  fi
fi

total=${#seeds[@]}
failed_seeds=()
idx=0

for seed in "${seeds[@]}"; do
  idx=$((idx + 1))
  echo "=== seed ${seed} (${idx}/${total}) ===" >&2
  if [[ "${continue_on_error}" == true ]]; then
    if ! bash "${WITH_SOFA}" uv run auto-surgery capture-brain-forceps-video \
      --master-seed "${seed}" \
      "${extra_args[@]}"; then
      failed_seeds+=("${seed}")
      echo "warning: capture failed for seed ${seed}" >&2
    fi
  else
    bash "${WITH_SOFA}" uv run auto-surgery capture-brain-forceps-video \
      --master-seed "${seed}" \
      "${extra_args[@]}"
  fi
done

if [[ "${continue_on_error}" == true && ${#failed_seeds[@]} -gt 0 ]]; then
  echo "error: failed seeds (${#failed_seeds[@]}): ${failed_seeds[*]}" >&2
  exit 1
fi
