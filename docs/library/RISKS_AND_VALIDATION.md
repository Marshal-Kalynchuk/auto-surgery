# Risks and Validation Plan

**Status:** v0.1 — companion to `ARCHITECTURE.md`
**Purpose:** Catalog the architecture's load-bearing bets, failure modes, and the first-month experiments that validate or refute them before downstream work commits.

This document exists because `ARCHITECTURE.md` is a research roadmap, not a deployable system. Several of its assumptions are unproven for surgery and need empirical validation early. This document names them and specifies what to do about each.

---

## 1. The Load-Bearing Bet

**Everything in `ARCHITECTURE.md` depends on one assumption: that an Inverse Dynamics Model trained in SOFA simulation can extract pseudo-actions from real human-surgeon video that are accurate enough to train a useful policy via behavioral cloning.**

If this is false, the policy story collapses and the architecture needs to be redesigned before any other component is built.

The bet is plausible because:

- VPT (OpenAI, 2022) demonstrated this approach for Minecraft from internet video
- The action representation (delta tool pose, gripper state) is at the right level of abstraction for transfer
- Pretrained foundation features (DINOv2, V-JEPA 2) reduce sim-to-real domain gap on observations

The bet is risky because surgery is the opposite of Minecraft on every relevant axis:

| Axis | Minecraft (where VPT worked) | Surgery (where this is unvalidated) |
|---|---|---|
| Action space | Small, discrete (movement + click) | Rich and continuous (delta pose + force + duty cycle + grip + pivoting) |
| Observation space | Clean, well-lit, deterministic rendering | Blood, smoke, motion blur, instrument occlusion, fluid, varying camera |
| Surgeon intent | Visible from frames (build a house, mine a block) | Often invisible from frames (retraction vs. exploration vs. correction) |
| Sim-real gap | Render gap only; physics is irrelevant | Tissue dynamics, tool-tissue interaction, fluid behavior all differ |
| Action-frame coupling | Low latency, deterministic | Variable latency, soft tissue compliance modulates effective action |

**Honest prior probability that a SOFA-trained IDM produces clinically useful pseudo-actions from real surgical video: 30–50%.**

This must be validated first. See §3.1.

---

## 2. Catalog of Failure Modes

Listed in roughly the order they will probably manifest if not addressed. Each has an associated mitigation or validation experiment.

### 2.1 Foundation features partially useless on surgical video

**Mode.** DINOv2 / V-JEPA 2 features are trained on natural video. Surgical video is out-of-distribution: smoke, blood, lighting variance, surgical-specific visual statistics. Slot decomposition may be unstable on real surgery; per-patch features may not carry surgically actionable information.

**Mitigation.** Surgical fine-tuning. Quantity unknown until validated.

**Validation.** §3.2.

### 2.2 Object permanence under cautery smoke

**Mode.** Cauterizer fires produce 1–5 seconds of obscuring smoke. Slot attention loses bindings during this period. Hopfield can pattern-complete, but only if the entity was well-encoded *before* smoke started. Long smoke events produce false "new entities" when objects re-emerge in slightly different configurations.

**Mitigation.** Pre-smoke entity state preserved in Hopfield; entry-on-emergence matched by similarity not coordinate. Tracker tolerance for long unobserved gaps.

**Validation.** Targeted evaluation on smoke-heavy clips after Phase 1.

### 2.3 IDM action-vocabulary mismatch

**Mode.** Real surgeons use techniques (pivoting through trocar fulcrum, gravity-assisted positioning, technique-specific compound motions) that a SOFA-trained IDM does not have in its training distribution. Pseudo-actions decode incorrectly.

**Mitigation.** SOFA scene generation must include the relevant action vocabulary. Domain randomization helps but doesn't substitute for representative dynamics.

**Validation.** §3.1 IDM transfer experiment must include diverse surgeon technique samples, not just trivial tool motions.

### 2.4 Cumulative pseudo-action drift in BC

**Mode.** Errors in IDM extraction compound when behavioral cloning trains on long sequences. Policy output diverges from demonstrator behavior past short horizons.

**Mitigation.** Train BC with short horizons + chunked replay; use DAgger when real teleop becomes available; consider noise-aware loss.

**Validation.** Standard BC evaluation on simulated trajectories with injected pseudo-action noise of known magnitude.

### 2.5 Forward-model sim-to-real failure

**Mode.** Forward model trained in SOFA gives wrong implausibility signals on real tissue (different compliance, different optical flow under contact). Safety gate becomes noisy or misses real failures.

**Mitigation.** Fine-tune on real data once available; calibrate implausibility thresholds with conformal prediction; treat early-deployment forward-model signals as *advisory*, not vetoing.

**Validation.** Phase 4 fine-tune; quantify false-positive and false-negative rates.

### 2.6 No reward signal beyond imitation

**Mode.** Imitation caps the policy at the worst surgeon in the training data minus IDM noise. There is no mechanism in v0 to do better. The architecture has no path to improvement past Phase 4 without RL or DAgger.

**Mitigation.** Phase 9+ adds Dreamer-style RL. DAgger is in §7.4 of `ARCHITECTURE.md` for surgeon corrections during real cases.

**Open.** This is a real architectural gap for v0–v8. The plan to close it needs a concrete spec — it is currently underspecified.

### 2.7 Predictive-coding features may not carry surgical task signal

**Mode.** Self-supervised features are general-purpose; surgical decisions are specific. The system may learn to predict pixels well while learning representations that don't carry surgically actionable information.

**Mitigation.** Behavioral cloning on pseudo-actions provides task signal at the policy level. The world model and slot attention remain self-supervised; if their features are insufficient, they can be augmented with task-specific auxiliary heads.

**Validation.** §3.3 measures whether the predictive-coding features support downstream classification of surgically meaningful events.

### 2.8 Hopfield capacity at case length

**Mode.** Modern Hopfield gives exponential capacity in theory. Surgical cases last hours and contain many entity-event associations. Practical capacity at this load is unvalidated.

**Mitigation.** Empirical capacity measurement; if insufficient, fall back to SSM-state or learned key-value memory with the same interface.

**Validation.** §3.4.

### 2.9 Distribution shift between training video and clinical surgery

**Mode.** Available surgical video is selected (teaching cases, well-recorded, well-lit). Real intraoperative variability is much wider — atypical anatomy, equipment failures, complications, surgeon style differences.

**Mitigation.** Diverse training data, OOD detection in policy substrate, escalation to surgeon under low confidence.

**Open.** Quantifying clinical distribution shift requires real-case sampling that is currently outside the data plan.

### 2.10 Brain-alignment may not buy what we hope

**Mode.** The "brain-aligned" framing in earlier drafts (now in `research/ARCHITECTURE_BRAIN_ALIGNED.md`) assumes connectivity-differentiated transformer modules outperform a monolithic transformer with the same parameter count. This is unproven.

**Mitigation.** The canonical spec (`ARCHITECTURE.md`) does not commit to brain-region modularization. The architecture is implementable as a monolithic backbone with multiple heads.

**Validation.** When module organization is empirically tested, compare against a same-parameter-count monolithic baseline.

---

## 3. First-Month Validation Experiments

Before committing to Phase 1+ training, run these experiments. Each is cheap relative to the cost of building the wrong architecture.

### 3.1 IDM Transfer Validation (highest priority)

**Goal.** Quantify how accurately a SOFA-trained IDM extracts pseudo-actions from real surgical video.

**Setup.**

1. Identify any available robot-collected surgical sessions where ground-truth actions exist (KUKA teleop logs, neuroArm research data, dVRK datasets, RoboNet surgical subsets).
2. Train an IDM on SOFA renders with action labels. Use the action representation that will go to the policy substrate (delta Cartesian pose + gripper).
3. Run the IDM on the real-data video. Compare extracted pseudo-actions to ground-truth actions frame-by-frame.

**Metrics.**

- RMSE on delta translation (mm), delta rotation (rad), gripper state (binary)
- Per-frame correlation with ground truth
- Sequence-level dynamic time warping distance

**Acceptance gates.**

- **Strong (proceed):** Per-frame RMSE within 2× the inter-surgeon variance baseline; correlation > 0.6 on translation
- **Moderate (proceed with risk):** RMSE 2–4× inter-surgeon variance; correlation 0.3–0.6
- **Weak (redesign required):** RMSE > 4× variance or correlation < 0.3 — the architecture's policy story is broken; redesign before further work

**Time budget.** 2–3 weeks.

**Fallback if weak.** Investigate alternatives: action-conditioned VLAs trained directly on surgical video, RL from contract-based rewards in sim, or restrict the architecture's claims to in-sim training only.

### 3.2 Foundation Feature Adaptation Quality

**Goal.** Quantify how much surgical fine-tuning is needed before backbone features are useful for slot attention and downstream tasks.

**Setup.**

1. Run frozen DINOv2 + V-JEPA 2 over a sample of surgical video clips
2. Train a simple slot attention head on top
3. Qualitatively inspect slot decompositions; quantitatively measure temporal consistency of slot identities
4. Repeat with light surgical fine-tuning of the backbone (1–2 GPU-days)
5. Compare slot quality before and after fine-tuning

**Metrics.**

- Slot identity coherence over time (fraction of slots maintaining spatial overlap > threshold across consecutive frames)
- Visual coherence of slot regions (subjective, scored by human reviewer)
- Slot-to-tool / slot-to-tissue alignment on a small annotated sample (one-time annotation, not training data)

**Acceptance gates.**

- **Strong:** Frozen features produce coherent slots → minimal fine-tuning needed
- **Moderate:** Light fine-tuning fixes slot coherence → planned investment scales
- **Weak:** Heavy fine-tuning required, possibly with from-scratch components → reassess training compute budget

**Time budget.** 1 week.

### 3.3 Predictive-Coding Feature Surgical Specificity

**Goal.** Verify that self-supervised features carry surgically actionable information.

**Setup.**

1. Extract features from the (fine-tuned) backbones over surgical video
2. Train small classifier heads on a small labeled sample for: (a) tool-in-frame, (b) bleeding-event, (c) cautery-active, (d) surgical-phase
3. Measure classification accuracy

**Acceptance gates.** Each task must reach reasonable accuracy on held-out clips (task-dependent thresholds; use literature baselines where available).

**Failure interpretation.** If predictive-coding features fail these classification tasks, augment with task-specific auxiliary heads during training rather than relying on pure self-supervision.

**Time budget.** 1–2 weeks. Requires a small annotated sample (not part of the no-annotation-budget guarantee — this is *evaluation*, not training).

### 3.4 Hopfield Capacity at Case Scale

**Goal.** Verify modern Hopfield can store and recall case-length episode loads.

**Setup.**

1. Synthetic experiment: encode N entity-event embeddings into Hopfield over a simulated case timeline (N scales: 100, 500, 2000, 5000)
2. Issue partial-cue queries; measure recall accuracy

**Acceptance gates.**

- Recall > 90% at N = 2000 entries → adequate for medium cases
- Below this → either reduce per-case write rate or switch to a learned key-value memory

**Time budget.** 3–5 days.

### 3.5 Forward-Model Sim-to-Real Smoke Test

**Goal.** Early estimate of how much the SOFA-trained forward model misfires on real video.

**Setup.**

1. Train forward model in SOFA with domain randomization
2. Run on real surgical video (no actions; passive observation)
3. Observe prediction-error distribution; note where errors are large

**Acceptance gates.** Error distribution distinguishable between expected-physics frames and visually-anomalous frames. False-positive rate on routine surgical motion < 10% at the threshold that catches injected synthetic anomalies.

**Time budget.** 1 week.

---

## 4. Architectural Gaps

The canonical spec has known gaps. Ordered by severity.

### 4.1 No reward signal until Phase 9

The architecture has no path to improvement past expert demonstrations until Dreamer-style RL is added. Imitation alone produces a system bounded by demonstration quality minus IDM noise.

**Action.** Spec the bridge: which contract-based rewards exist in SOFA, when DAgger becomes available with real teleop, how the slow planner uses world-model rollouts to identify low-confidence states for surgeon escalation.

### 4.2 No evaluation framework

`ARCHITECTURE.md` Phase 0 → Phase 9 has phase descriptions but no metrics, gates, or acceptance criteria. "Phase 4 complete" is not currently definable.

**Action.** Define per-phase metrics:

- Phase 1: slot coherence + feature quality (§3.2, 3.3)
- Phase 2: IDM accuracy (§3.1) + forward-model error distribution (§3.5)
- Phase 3: BC validation accuracy + cumulative drift bounds (§2.4)
- Phase 4: real-KUKA peg-transfer success rate, action-error budgets
- Phase 5+: Hopfield recall (§3.4), suturing benchmarks
- Phase 7: cadaveric task completion + safety violation rate

### 4.3 No data contract

The architecture assumes "mono surgical video" as input but does not size:

- Hours of video available
- Procedures covered
- Camera and quality variance
- Patient consent and de-identification status
- Annotation budget for evaluation samples (not training)

**Action.** A separate doc, `DATA_CONTRACT.md`, listing actual data on hand and required acquisition.

### 4.4 Cross-component gradient flow underspecified

Are modules trained jointly, in stages, or end-to-end fine-tuned? Each choice has different stability properties. The current spec is silent.

**Action.** Spec a training schedule per phase: which losses are active, which weights are frozen, which gradients flow.

### 4.5 Uncertainty calibration beyond "conformal prediction"

The safety architecture cites conformal prediction. Conformal prediction is not turnkey for high-dimensional safety surfaces. Specifically:

- Calibration set composition
- Coverage guarantees under distribution shift
- Sliding-window updates during a case

**Action.** A separate doc, `SAFETY_CALIBRATION.md`, with concrete protocol.

### 4.6 Surgeon language directive interface

VL-JEPA was deferred from the critical path. The architecture currently has no language-grounding mechanism. If surgeon directives during real cases become a requirement, this needs to be reactivated.

**Action.** Track as deferred; revisit when language input becomes load-bearing.

---

## 5. Validation Gates Between Phases

Each phase transition requires explicit pass criteria. Failure to meet criteria triggers redesign, not a forward push.

| From | To | Gate |
|---|---|---|
| 0 → 1 | Infrastructure → adaptation | §3.1 IDM validation reaches at least *moderate* acceptance |
| 1 → 2 | Adaptation → world model + IDM + forward model | §3.2 feature-quality gate; §3.3 predictive-coding gate |
| 2 → 3 | WM/IDM/forward model → policy training | IDM transfer validated on diverse surgeon technique (§3.1 expanded); forward-model error distribution acceptable (§3.5) |
| 3 → 4 | Policy in sim → KUKA real | Sim peg-transfer success rate above threshold; cumulative BC drift bounded |
| 4 → 5 | KUKA bring-up → Hopfield activation | Real-data fine-tuning improves component metrics; §3.4 Hopfield capacity validated |
| 5 → 6 | Hopfield active → consolidation | Cross-case knowledge transfer measurable on benchmark task |
| 6 → 7 | Suturing → cadaveric narrow autonomy | Surgeon-judged suturing quality on phantom; safety violation rate below clinical threshold |
| 7 → 8 | Cadaveric → neuroArm port | KUKA → neuroArm policy fine-tune converges; control-mode mapping validated |
| 8 → 9 | neuroArm port → procedural autonomy | Autonomy on multiple narrow tasks; RL refinement infrastructure in place |

If a gate fails, the response is **redesign, not retry**. A failed gate means an assumption upstream is broken.

---

## 6. Honest Probability Estimates

These are subjective priors, not computed probabilities. They are documented to keep expectations calibrated.

| Milestone | 12-month probability | 24-month probability |
|---|---|---|
| Phase 1 complete (slot attention + adapted backbones working on surgical video) | High (70–85%) | Very high |
| Phase 2 complete (DINO-WM + IDM + forward model trained, IDM transfer validated) | Moderate (50–65%) | High |
| Phase 3 complete (imitation policy producing reasonable peg-transfer in sim) | Moderate (40–55%) | High |
| Phase 4 complete (KUKA real peg-transfer working) | Low-Moderate (30–45%) | Moderate-High |
| Phase 6 complete (suturing on phantom with Hopfield-active recall) | Low (15–25%) | Moderate (40–55%) |
| Phase 7 complete (cadaveric narrow-task autonomy) | Very low (<10%) | Low (15–30%) |
| Clinically useful neuroArm autonomy from this architecture | Negligible | Low (10–20%) without major rework |

Most of the risk concentrates in §1 (IDM transfer) and §2.1 / §2.2 (foundation features + occlusion). These are the components to validate first. The §3 first-month experiments are designed to bring the §1 and §2.1 estimates above water before the rest of the plan commits to them.

---

## 7. Recommended Sequencing

The next four weeks should run in roughly this order:

| Week | Work |
|---|---|
| 1 | §3.1 IDM transfer validation setup (acquire real-data benchmark, train SOFA IDM) |
| 1 | §3.4 Hopfield capacity synthetic experiment (cheap, in parallel) |
| 2 | §3.1 IDM transfer experiment runs and gate evaluation |
| 2 | §3.2 foundation feature quality (slot attention on surgical video) |
| 3 | §3.3 predictive-coding feature classification probes |
| 3 | §3.5 forward-model sim-to-real smoke test |
| 4 | Synthesis: report on all five experiments; gate decision for Phase 1 commitment |

If the IDM transfer gate fails at week 2, halt downstream planning and reconvene on architecture redesign. Do not start Phase 1 component training until the gate passes.

---

## 8. Final Position

`ARCHITECTURE.md` is a principled research roadmap with several unproven bets. This document names those bets, specifies how to validate them cheaply, and gates downstream work on the validation outcomes.

**The single most important action is the §3.1 IDM transfer experiment.** Until it runs and produces a result, the architecture's policy story is a hypothesis, not a plan.

**The second most important action is closing the §4.1 reward-signal gap and the §4.2 evaluation-framework gap.** Without these, even successful component training cannot demonstrate progress.

This document should be revisited after the first-month experiments complete and again at every phase gate.
