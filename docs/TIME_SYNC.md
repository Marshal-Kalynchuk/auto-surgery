# Time synchronization (capture infrastructure)

Architecture requirement ([ARCHITECTURE.md](ARCHITECTURE.md) §5.1): multimodal streams share a common clock with **≤1 ms drift** and are logged continuously.

## Recommended deployment

1. Prefer **PTP (IEEE 1588)** on capture PCs, robot controller hosts, and network switches when hardware supports it.
2. If PTP is unavailable, use **disciplined NTP** with local stratum-1 or low-latency LAN servers—not default public pools—for intraoperative capture.
3. Record `clock_source` (`ptp` | `ntp` | `monotonic`) on every `SensorBundle` and on `SessionManifest`.

## Verification procedure

1. Log **paired timestamps** from each modality into a dedicated calibration session (e.g., GPIO pulse, LED flash, or shared sample clock).
2. Compute **empirical skew** between modalities over the session; archive the stats next to the session manifest.
3. If skew exceeds 1 ms, document mitigations (interpolation policy, resampling, or relaxed guarantees for that rig) **before** claiming compliance.

## Simulator / lab shortcut

Deterministic stubs may use `clock_source="monotonic"` for engineering-only datasets; production OR capture **must** record the real synchronization stack used.
