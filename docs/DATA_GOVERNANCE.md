# Data governance (Phase 0 gate)

This repository processes surgical robotics data that may include **PHI**. Phase 0 implements **manifest-level classification** so training pipelines refuse PHI workloads unless explicitly cleared.

## Manifest fields

- `DataClassification`: `synthetic` | `simulation` | `lab_non_phi` | `phi_restricted` | `deidentified_derived`
- `RetentionTier`: operational labels for lifecycle policy ([ARCHITECTURE.md](ARCHITECTURE.md) §8.1 / logging manifests)

## Training gate

`training.datasets.assert_training_gate` raises `PhiDatasetBlockedError` for `PHI_RESTRICTED` datasets unless `phi_training_allowed=True`.

**Policy:** keep `phi_training_allowed=False` everywhere by default (including CI). Enable only on compliant workstations after:

1. **RBAC** on bucket/NAS paths holding raw telemetry.
2. **Encryption at rest** for PHI tiers.
3. **Audit logging** of dataset exports and checkpoint promotions.
4. Documented **de-identification method** or legal basis on manifests (`deidentification_method`, `access_control_decision_ref`).

## Synthetic / simulation path

Development may proceed on `SYNTHETIC` or `SIMULATION` classifications without PHI controls.
