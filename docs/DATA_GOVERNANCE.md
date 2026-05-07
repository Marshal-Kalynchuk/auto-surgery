# Data Governance (Phase 0 Gate)

**Status:** v1.0 — applies to ARCHITECTURE.md v2.5 and later

Repository processes surgical robotics data that may include **PHI**. All training tiers (TRAINING_STRATEGY.md) implement manifest-level classification so pipelines refuse PHI workloads unless explicitly cleared.

## Manifest Fields

- `DataClassification`: `synthetic` | `simulation` | `lab_non_phi` | `phi_restricted` | `deidentified_derived`
- `RetentionTier`: operational labels for lifecycle policy
- `clock_source`: (for real-robot phases) `ptp` | `ntp` | `monotonic` — see TIME_SYNC.md

## Training Gate

Training pipeline (TRAINING_STRATEGY.md Tiers 0–4) refuses `phi_restricted` datasets unless `phi_training_allowed=True`.

**Policy:** Keep `phi_training_allowed=False` everywhere by default (including CI). Enable only on compliant workstations after:

1. **RBAC** on bucket/NAS paths holding raw telemetry.
2. **Encryption at rest** for PHI tiers.
3. **Audit logging** of dataset exports and checkpoint promotions.
4. Documented **de-identification method** or legal basis on manifests (`deidentification_method`, `access_control_decision_ref`).

## Synthetic / simulation path

Development may proceed on `SYNTHETIC` or `SIMULATION` classifications without PHI controls.
