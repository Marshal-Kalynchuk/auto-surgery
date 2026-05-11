# Target Robot: KUKA KR 6-2

**Purpose:** Single reference for the **KUKA KR 6-2** six-axis arm used by this project for simulation, control, and integration planning.  
**Scope:** Public, machine-level summaries and pointers to manuals. **Joint torques, speed limits, wrist payloads, and load diagrams** must come from KUKA documentation for the **exact type and serial** on your nameplate.

**Designation:** This repository targets the **KUKA KR 6-2** (also written **KR 6 2** or **KR6-2** in vendor and simulation libraries).

---

## 1. Representative technical data

Values below are from the **RoboDK** robot library for quick cell sizing and simulation asset choice. They are **not** a substitute for KUKA’s specification for your machine.

**Source:** [RoboDK — KUKA KR 6 2](https://robodk.com/robot/KUKA/KR-6-2)

| Topic | Value |
|--------|--------|
| Axes | 6 |
| Payload | 6.0 kg |
| Reach | 1611 mm |
| Repeatability | 0.05 mm |
| Weight | 235 kg |

Confirm **controller** and **KSS** version against your cabinet and KUKA system documentation.

---

## 2. Documentation and files

1. **Operating instructions (example host):** [ManualsLib — Kuka KR 6-2 Operating Instructions Manual](https://www.manualslib.com/manual/3420907/Kuka-Kr-6-2.html) — prefer a copy from **my.KUKA / KUKA Xpert** for controlled work.
2. **KUKA Xpert / my.KUKA:** Use [KUKA Xpert](https://xpert.kuka.com/) and a [my.KUKA](https://my.kuka.com/s/signup) account for CAD, specs, and software. The public Download Center may not match the literal string `KR 6-2`; use the **commercial type** on the nameplate when searching ([Download Center](https://www.kuka.com/en-us/services/downloads?terms=Language%3Aen%3A1%3Bproduct_name%3AKR%206-2)).
3. **Simulation geometry:** [RoboDK library file](https://cdn.robodk.com/downloads-library/library-robots/KUKA-KR-6-2.robot) — verify against your CAD before relying on it for clearance or calibration studies.

---

## 3. Implications for this project

- **Workspace:** Plan for a **medium-reach** (~1.6 m class) arm and a **heavy** manipulator base compared with compact small-payload designs.
- **Simulation parity:** Label assets with the **exact KUKA type** from the nameplate (e.g. library string `KUKA-KR-6-2` vs order code). Re-run validation when controller or software generation changes.
- **Safety:** Industrial KUKA robots require **risk assessment**, **safeguarding**, and compliance with **applicable machinery regulations**. This document is **not** a safety manual.

---

## 4. Revision

| Date | Change |
|------|--------|
| 2026-05-04 | **KR 6-2** reference: specs, docs pointers, project notes. |
