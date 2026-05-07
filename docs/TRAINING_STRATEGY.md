# Training Strategy

**Status:** v1.0 — aligned with ARCHITECTURE.md v2.5
**Last updated:** 2026-05-06
**Parent:** ARCHITECTURE.md §7 (Training Strategy)
**Scope:** Training pipeline for autonomous surgical robotics, organized by tier. Emphasizes self-supervised learning on foundation models, imitation from surgical video via inverse-dynamics pseudo-actions, and three-timescale memory.

---

## 0. Design Principle: Local Learning Rules at Different Timescales

The architecture combines self-supervised foundation models, imitation learning from surgical video (via inverse-dynamics-extracted pseudo-actions), and three-timescale memory. A naive end-to-end training approach would face circular dependencies, long credit assignment horizons, and representation collapse.

**Solution: local learning rules at different timescales.** Each component learns from signals available to it directly without waiting for the entire loop to close. Interconnections carry information at inference time; learning is driven by local objectives.

| Principle | Implementation |
|---|---|
| **Perception learns from self-supervision** | DINOv2, V-JEPA 2, slot attention train on reconstruction and temporal consistency without downstream signals |
| **World model learns from prediction error** | DINO-WM and fast forward model optimize next-state prediction independently |
| **Policy learns from behavior cloning** | Substrate trains on pseudo-actions extracted via IDM, independent of world model quality |
| **Three-timescale memory** | Activations (working), Hopfield fast weights (case-long), slow weights (cross-case). No retrieval buffers. |
| **Downstream consumers adapt** | Planner and policy learn to use richer representations as they improve |

---

## 1. Training Tiers

Training is organized into four tiers by timescale and dependency:

```
Tier 0: Foundation models (use existing checkpoints)
  └── DINOv2, V-JEPA 2 (already trained; fine-tune on surgical video)

Tier 1: Surgical adaptation (self-supervised, per-frame)
  └── Slot attention, backbone fine-tuning, entity binding

Tier 2: Dynamics & policy (short-horizon, imitation + RL)
  └── DINO-WM, IDM, fast forward model, policy substrate

Tier 3: Memory encoding (case-long, self-supervised)
  └── Hopfield encoder/decoder bridge; Hopfield inference-time updates

Tier 4: Memory policy (cross-case, offline RL — deferred to Phase 9)
  └── Consolidation policy (not needed for initial deployment)
```

Tiers 0–3 run in parallel or sequential dependency. Tier 4 is deferred until RL refinement becomes a bottleneck (Phase 9+).

---

## 2. Tier 0 — Foundation Models (Pretrained Checkpoints)

No training required. Use existing checkpoints:

| Model | Source | Role |
|---|---|---|
| **DINOv2** | Meta, 2023 | Per-patch visual features for action and semantic streams |
| **V-JEPA 2** | Meta, 2025 | Self-supervised video features; bootstrap policy substrate |
| **DINO-WM** | NYU + Meta, 2025 | World model for planner search |
| **Modern Hopfield** | Ramsauer et al., 2020 | Fast-weight memory (inference-time updates, no training) |

**Action:** Load checkpoints, verify compatibility, set up fine-tuning pipelines.

---

## 3. Tier 1 — Surgical Adaptation (Self-Supervised)

Fine-tune foundation models on surgical video without labels. All losses are self-supervised.

### 3.1 Slot Attention

**Data:** Mono surgical video (no annotations required)

**Method:** Reconstruction + temporal consistency loss

```
backbone_features → K slots → reconstruct features + temporal_consistency(slot_t, slot_{t+1})
```

**Objective:** Slots emerge as object-like representations (tools, tissue, blood pools, anatomy) without labels.

**Training data construction:**
- Extract 5–10 second clips from surgical video
- Run frozen DINOv2 + V-JEPA 2 backbones to get features
- Train slot attention with reconstruction loss + temporal consistency

**Success metric:** Slot coherence over time (frame-to-frame overlap > threshold; tracks persist under small deformations).

### 3.2 V-JEPA 2 Fine-Tuning

**Data:** Same mono surgical video

**Method:** Masked feature prediction (frozen backbone, train prediction head)

**Objective:** Predict masked patch features given surrounding context

**Success metric:** Prediction accuracy on held-out clips; qualitative inspection of learned features (should capture surgical semantics).

### 3.3 Backbone Adaptation (Optional)

If frozen backbones + fine-tuned heads don't achieve acceptable performance (measured in §3.3.1 below), unfreeze backbone layers and fine-tune end-to-end with a small learning rate.

---

## 4. Tier 2 — Dynamics and Policy (Short-Horizon)

Train on paired (observation, action, next_observation) tuples. For surgical video, actions come from inverse dynamics (see §4.1).

### 4.1 Inverse Dynamics Model (IDM) Training

**This is load-bearing.** The IDM bridges the no-action-label problem.

**Setup:**

1. **Train in SOFA simulation.** Generate paired (frame_t, frame_t+1, action_t) with ground-truth actions.
   - Domain randomization: tissue properties, lighting, camera intrinsics, tool geometry
   - Action representation: delta Cartesian pose + gripper state (same as policy substrate will use)
   
2. **Train supervised with MSE loss:**
   ```
   IDM(backbone_features_t, backbone_features_t+1) → predicted_action_t
   Loss = MSE(predicted_action_t, ground_truth_action_t)
   ```

3. **Validate on real data (if available).** Compare IDM pseudo-actions against ground-truth teleop actions.
   - See RISKS_AND_VALIDATION.md §3.1 for validation gates

**Success metric:** RMSE on delta translation and rotation matches inter-surgeon variance at acceptable level (see RISKS_AND_VALIDATION.md).

### 4.2 Apply IDM to Surgical Video

Once IDM is trained in sim, apply it frame-by-frame to surgical video to extract pseudo-actions.

**Data preprocessing:**
1. Extract overlapping windows of N frames from surgical video
2. Run backbone features through IDM
3. Produce (state, pseudo_action) pairs for behavioral cloning

### 4.3 Fast Forward Model

**Data:** SOFA sim with ground-truth next states

**Method:** Supervised prediction

```
forward_model(sensory_t, motor_command_t) → predicted_sensory_{t+1}
Loss = MSE(predicted_sensory_{t+1}, actual_sensory_{t+1})
```

**Role:** Fast controller refinement + safety gate implausibility detection

**Training:** Sim-only initially. Fine-tune on real data once available.

### 4.4 DINO-WM World Model

**Data:** Surgical video transitions + pseudo-actions from IDM

**Method:** Self-supervised prediction on foundation-model features

**Objective:** Predict future patch features given action sequences

**Training:**
1. Use frozen DINOv2 features as prediction targets
2. Train DINO-WM to predict features T steps ahead
3. Measure prediction accuracy on held-out video

**Success metric:** Prediction RMSE on unseen surgical videos; comparison against a no-action baseline (does action information improve prediction?).

### 4.5 Policy Substrate (Behavioral Cloning)

**Data:** Surgical video + pseudo-actions from IDM (Tier 2 IDM must be validated first)

**Method:** Behavioral cloning with language directives and behavior contracts

**Objective:** Predict pseudo-actions given (active entity state, surgeon directive, behavior contract)

**Training:**
```
substrate(entity_embedding, directive, contract) → predicted_pseudo_action_t
Loss = MSE(predicted_pseudo_action_t, ground_truth_pseudo_action_t)
```

**Curriculum:**
1. Start with simple directives (e.g., "grasp this tool") and rigid objects (no tissue deformation)
2. Gradually introduce complex directives (e.g., "resect tumor avoiding this vessel") and soft-tissue dynamics
3. Measure performance on peg-transfer benchmarks in sim

**Success metric:** Sim peg-transfer success rate; qualitative inspection of policies on held-out video clips.

---

## 5. Tier 3 — Memory Encoding (Case-Long, Self-Supervised)

This tier is deferred until Phase 5 (Hopfield activated at inference). For v0–v4, active entity embeddings + slot tracking are sufficient (no Hopfield consolidation).

**Placeholder for future work:** When memory beyond a single case becomes important, train Hopfield encoder/decoder bridge with a reconstruction objective: given case-long entity interaction sequences, the bridge should encode them such that Hopfield can retrieve them via pattern completion.

---

## 6. Tier 4 — Memory Policy (Cross-Case, Offline RL)

This tier is deferred until Phase 9 (Dreamer-style RL). Not needed for initial deployment.

**Placeholder for future work:** Train consolidation policy via offline RL on procedure logs with hindsight rewards. See RISKS_AND_VALIDATION.md for detailed design.

---

## 7. Integration with ARCHITECTURE.md

### 7.1 Tier 0 — Checkpoints

ARCHITECTURE.md §5 specifies which foundation models are critical path. Load and verify them at Phase 0.

### 7.2 Tier 1 — Surgical Adaptation

ARCHITECTURE.md §6.1 (Perception). Run Tier 1 training at Phase 1.

Validation gate (RISKS_AND_VALIDATION.md §3.2): Foundation features must support coherent slot decomposition on surgical video.

### 7.3 Tier 2 — Dynamics & Policy

ARCHITECTURE.md §6.3–6.5. Run at Phase 2–3.

**Critical gate:** IDM transfer validation (RISKS_AND_VALIDATION.md §3.1). Do not proceed to policy training until IDM accuracy passes acceptance.

### 7.4 Tier 3 — Hopfield Activation

ARCHITECTURE.md §6.2.2. Activate at Phase 5 (no consolidation yet; Hopfield inference-time updates only).

Validation gate (RISKS_AND_VALIDATION.md §3.4): Hopfield capacity at case length must be empirically validated.

### 7.5 Tier 4 — Consolidation Policy

Deferred to Phase 9. See RISKS_AND_VALIDATION.md §6 for offline RL specification.

---

## 8. Per-Component Summary Table

| Component | Tier | Training Data | Method | Local Objective | Phase |
|---|---|---|---|---|---|
| Slot attention | 1 | Surgical video | Reconstruction + temporal consistency | Slot coherence over time | 1 |
| V-JEPA 2 fine-tune | 1 | Surgical video | Masked feature prediction | Prediction accuracy | 1 |
| IDM | 2 | SOFA sim + real teleop (validation) | Supervised | Next-action prediction RMSE | 2 |
| Fast forward model | 2 | SOFA sim | Supervised | Next-sensory prediction accuracy | 2 |
| DINO-WM | 2 | Surgical video + pseudo-actions | Self-supervised prediction | Feature prediction accuracy | 2 |
| Policy substrate | 2 | Surgical video + IDM pseudo-actions | Behavioral cloning | Pseudo-action match, sim peg-transfer success | 3 |
| Slow planner | 2 | Teleop goal decomposition | Imitation + supervised | Directive quality (domain-specific) | 3 |
| Fast controller | — | Pre-built | Classical (MPC) | Trajectory tracking (not ML) | 0 |
| Safety evaluator | — | Simulation + labeled safety data | Conformal calibration | Coverage at nominal levels | 0 |
| Hopfield bridge | 3 | Case-long entity sequences (deferred) | Reconstruction (deferred) | Retrieval accuracy (deferred) | 5 |
| Consolidation policy | 4 | Procedure logs with hindsight reward (deferred) | Offline RL (deferred) | Hindsight consolidation reward (deferred) | 9 |

---

## 9. Data Sources and Availability

**Required for v0 training:**

1. **Mono surgical video** — primary training data source. Assumption: available in quantity (100+ hours of surgical footage).
2. **SOFA simulation** — for IDM and fast forward model training. SOFA scenes with domain randomization.
3. **Sim peg-transfer benchmarks** — evaluation, not training data.

**Not required for v0 training (deferred):**

- Multimodal sensor ground truth (kinematics, forces, etc.)
- Annotated segmentation masks
- Explicit action labels (recovered via IDM instead)
- Real-robot teleop logs (used for validation and Phase 4 fine-tuning)

**Validation experiments require small annotated samples:**

- RISKS_AND_VALIDATION.md §3.1 (IDM transfer): a few real-robot + human-surgeon video pairs with ground-truth actions (100–1000 pairs)
- RISKS_AND_VALIDATION.md §3.3 (feature quality): small labeled sample for surgical-event classification (100–500 clips)

---

## 10. Failure Modes and Mitigations

### 10.1 IDM Transfer Fails

**Symptom:** Pseudo-actions extracted from surgical video are inaccurate (high RMSE, low correlation with inter-surgeon variance).

**Mitigation:** 
- Increase diversity of SOFA scenes (tissue parameters, tool geometry, camera intrinsics)
- Investigate action-vocabulary mismatch (see RISKS_AND_VALIDATION.md §2.3)
- If transfer accuracy remains too low, consider alternative: train a small VLA on surgical video + sim trajectory pairs instead of relying on IDM

**Gate:** RISKS_AND_VALIDATION.md §3.1. Do not proceed to policy training if IDM gate fails.

### 10.2 Slot Attention Unstable Under Occlusion

**Symptom:** Slots lose identity when tissue is occluded (cautery smoke, blood) and spawn "new entities" when objects re-emerge.

**Mitigation:**
- Increase temporal consistency loss weight
- Improve tracker to handle long unobserved gaps
- Early experiments: measure slot coherence on smoke-heavy clips; if poor, iterate on architecture before downstream work commits to this binding mechanism

### 10.3 Forward Model Sim-to-Real Gap

**Symptom:** Forward-model predictions on real surgical video are inaccurate; safety gate produces false positives.

**Mitigation:**
- Domain randomization during SOFA training
- Fine-tune on real data once available
- Calibrate implausibility threshold with conformal prediction
- Treat early-deployment signals as advisory, not vetoing

### 10.4 Behavioral Cloning Drift

**Symptom:** Policy performance degrades over long sequences (cumulative pseudo-action error).

**Mitigation:**
- Train with short horizons (5–10 frame windows)
- Use chunked replay and importance weighting
- Introduce DAgger (surgeon corrections) during real deployment
- Measure drift in sim before real deployment

---

## 11. Roadmap and Validation Gates

| Milestone | Validation Gate | Blocks |
|---|---|---|
| Tier 1 complete (slot attention + V-JEPA 2 fine-tune) | RISKS_AND_VALIDATION.md §3.2–3.3 (feature quality and predictive coding) | Tier 2 start |
| IDM trained and validated | RISKS_AND_VALIDATION.md §3.1 (IDM transfer accuracy) | Policy substrate training |
| Tier 2 complete (DINO-WM, IDM, forward model) | IDM gate + forward-model smoke test (§3.5) | Policy substrate training |
| Policy substrate trained in sim | Sim peg-transfer benchmark | Real KUKA deployment |
| Phase 4 complete (KUKA real) | Real peg-transfer + action-error budget | Phase 5 (Hopfield active) |
| Phase 5 complete (Hopfield inference) | RISKS_AND_VALIDATION.md §3.4 (Hopfield capacity) | Phase 6+ |

---

## 12. Summary

The training strategy is four-tier and local-learning-first:

1. **Tier 0:** Use existing foundation models (DINOv2, V-JEPA 2, DINO-WM) without modification.
2. **Tier 1:** Self-supervised fine-tuning on surgical video (slot attention, masked prediction).
3. **Tier 2:** Imitation learning on pseudo-actions extracted via IDM, validated against real-robot data.
4. **Tier 3 & 4:** Deferred (Hopfield and consolidation policy not needed for v0).

**Load-bearing bet:** IDM transfer (§3.1 and RISKS_AND_VALIDATION.md §3.1). Until this gate passes, downstream policy work is speculative.

**No annotations required.** No supervised perception labels. All training is self-supervised or imitation-based.
