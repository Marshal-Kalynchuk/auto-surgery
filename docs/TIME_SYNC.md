# Time Synchronization (Capture Infrastructure)

**Status:** v1.0 — applies to ARCHITECTURE.md v2.5 and later

Architecture requirement: Multimodal capture streams (when available) share a common clock with **≤1 ms drift** and are logged continuously. Applies primarily to real-robot phases (Phase 4+). For mono-video-only phases (Phase 0–3), this is not a constraint.

## Recommended Deployment

1. Prefer **PTP (IEEE 1588)** on capture PCs, robot controller hosts, and network switches when hardware supports it.
2. If PTP is unavailable, use **disciplined NTP** with local stratum-1 or low-latency LAN servers—not default public pools.
3. Record `clock_source` (`ptp` | `ntp` | `monotonic`) on every capture session.

**Early phases (Phase 0–3):** Mono video only; no multimodal synchronization needed. Use `monotonic` clocks.

**Real-robot phases (Phase 4+):** When stereo cameras, force/torque sensors, and kinematics are available, enforce synchronization per above.

## Verification Procedure

For multimodal datasets (Phase 4+):

1. Log **paired timestamps** from each modality in a dedicated calibration session (GPIO pulse, LED flash, or shared sample clock).
2. Compute **empirical skew** between modalities over the session; archive the stats next to the session manifest.
3. If skew exceeds 1 ms, document mitigations (interpolation policy, resampling, relaxed guarantees) **before** claiming compliance.

## Simulator / Lab Shortcut

Deterministic stubs (Phase 0–3) may use `clock_source="monotonic"` for engineering-only datasets. 

**Production capture (Phase 4+) must record the real synchronization stack used.**
