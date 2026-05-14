# Bound forceps motion — visual smoke (seeds 8 & 9)

## Automated run (this workspace)

`capture-brain-forceps-video` was **not** executed successfully here: native SOFA Python bindings are absent (`SofaNativeRenderError` from `validate_native_capture_runtime()`).

## Local acceptance (when SOFA is available)

From repo root (after `infra/sofa/setup_sofa_conda.sh` and `source .env.sofa` per the error message):

```bash
uv run auto-surgery capture-brain-forceps-video --master-seed 8 \
  --output-dir artifacts/bf_seed8_postfix --num-episodes 1 --ticks 240
uv run auto-surgery capture-brain-forceps-video --master-seed 9 \
  --output-dir artifacts/bf_seed9_postfix --num-episodes 1 --ticks 240
```

**Checks**

1. At least one tick with `safety.command_blocked == True` and envelope-related blocking (e.g. `tip_outside_workspace_envelope`) in recorded safety / control artifacts.
2. Per-tick tip displacement from `forceps_contract_trace.parquet` (or equivalent): Euclidean step \(\le \texttt{max\_linear\_mm\_s} \times dt \times 1.05\) (defaults in `motion_shaping_defaults.yaml` for `dejavu_brain`).
3. Visual: no off-frame teleport; tissue may still clip until tissue collision is implemented.
