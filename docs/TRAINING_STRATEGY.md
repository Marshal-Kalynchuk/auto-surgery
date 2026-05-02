# Training Strategy

**Status:** Draft v0.1
**Last updated:** 2026-04-29
**Parent:** ARCHITECTURE.md §8 (Data, Training, Simulator)
**Scope:** Complete training strategy for the autonomous surgery stack, with particular depth on the entity knowledge store components (digest update network, consolidation policy) and the brain-inspired principles that make training a deeply interconnected system tractable.

---

## 0. Design Principle: Local Learning Rules at Different Timescales

The architecture is deeply interconnected at inference time — entity embeddings flow to the world model, which generates prediction errors that flow to the planner, which gates consolidation back into entity state, which changes the embeddings that the world model reads. A naive end-to-end training approach (one loss, one backward pass through the whole loop) would face circular dependencies, extremely long credit assignment horizons, and representation collapse.

The brain faces the same problem at far greater scale and solves it with **local learning rules operating at different timescales**. Each neural system learns from signals available to it directly, without waiting for the entire loop to close. Interconnections carry information at inference time, but learning is driven by local objectives.

This training strategy applies the same principle:

| Principle | Brain mechanism | Our implementation |
|---|---|---|
| **Perception learns from self-supervision** | Visual cortex learns from local predictive coding, not downstream utility | Perception streams, slot attention, and joint embedding train on contrastive/reconstruction objectives independent of downstream components |
| **Memory writes first, evaluates later** | Hippocampus encodes promiscuously; consolidation sorts out value during sleep | Digest update network learns self-supervised compression; consolidation policy is trained separately via offline replay |
| **Consolidation policy learns from replay with hindsight** | Sleep replay with dopaminergic reward tags | Offline RL over the event archive with hindsight reward computed from downstream prediction accuracy |
| **Downstream consumers adapt to available representations** | Motor cortex works with whatever PFC signal is available and co-adapts | World model and policy train with digest optionally available (dropout on digest pathway); gradually learn to exploit it as it improves |
| **Different timescales break circular dependency** | Synaptic plasticity (ms), Hebbian (s), consolidation (hours), cortical learning (weeks) | Four training tiers with decreasing frequency and increasing temporal scope |

The remainder of this document specifies each component's training in detail, organized by tier.

---

## 1. Training Tiers

Training is organized into four tiers by timescale and dependency structure. Each tier's components have local objectives that can be trained without waiting for later tiers to converge.

```
Tier 1: Foundation (no temporal state, self-supervised + supervised)
  └── Perception streams, slot attention, joint embedding, safety calibration

Tier 2: Dynamics & Policy (short-horizon temporal, imitation + RL)
  └── World model, policy substrate, online adapters, planner reasoning

Tier 3: Memory Encoding (medium-horizon, self-supervised compression)
  └── Digest update network

Tier 4: Memory Policy (long-horizon, offline RL with hindsight)
  └── Consolidation policy

Co-training refinement: all tiers active, soft coupling
```

Tiers 1 and 2 correspond to existing training in ARCHITECTURE.md §8.2. Tiers 3 and 4 are new, introduced by the entity knowledge store. The co-training phase replaces the rigid freeze/unfreeze staging with a softer co-adaptation approach (see §5).

---

## 2. Tier 1 — Foundation Components

These components learn from local objectives with no dependency on downstream consumers. They provide stable representations that all later tiers build on.

### 2.1 Action Stream Encoder

| Property | Value |
|---|---|
| **Data** | Sim + real teleop |
| **Method** | Self-supervised (next-frame prediction, contrastive across views) + supervised on tracking labels |
| **Objective** | Reconstruct/predict geometric features; tracking accuracy on labeled sequences |
| **Timescale** | Per-frame |
| **Brain analogue** | Primary visual cortex — learns features from local prediction error |

No dependency on entity knowledge store. Trained first or in parallel with other Tier 1 components.

### 2.2 Semantic Stream Encoder

| Property | Value |
|---|---|
| **Data** | Public surgical video + real teleop, fine-tuned on segmentation labels |
| **Method** | Foundation model fine-tune (SAM 2 / MedSAM2) |
| **Objective** | Segmentation IoU, classification accuracy |
| **Timescale** | Per-frame |
| **Brain analogue** | Inferotemporal cortex — learns categories from labeled examples |

No dependency on entity knowledge store.

### 2.3 Slot Attention

| Property | Value |
|---|---|
| **Data** | Real teleop + sim with slot ground truth |
| **Method** | Supervised on sim slot assignments, semi-supervised on real |
| **Objective** | Slot assignment accuracy, tracking persistence, spawn/despawn recall |
| **Timescale** | Per-frame |

Depends on Tier 1 encoders. Can be co-trained with them when using a joint loss, or trained sequentially with frozen encoders.

### 2.4 Joint Embedding Model

| Property | Value |
|---|---|
| **Data** | Multi-modal pairs across all logged modalities (vision, geometry, kinematics, force, language) |
| **Method** | Multi-modal contrastive (CLIP-style InfoNCE), cross-instance similarity, action-conditioned similarity, language alignment |
| **Objective** | Cross-modal retrieval accuracy, type/affordance probe accuracy, action-conditioned similarity |
| **Timescale** | Per-frame / per-pair |
| **Brain analogue** | Associative cortex — learns cross-modal binding from co-occurrence |

**Key detail:** The joint embedding defines the representation space that the entity knowledge store operates in. The interaction digest lives in a subspace of (or adjacent to) this embedding space. The embedding model must be trained and stabilized before Tier 3 begins, because the digest update network needs a meaningful target space.

**Compatibility-preserving retraining.** Per ARCHITECTURE.md §6.4.3: anchor-based fine-tuning with regularization against embedding drift of canonical reference slots. All downstream components declare compatible embedding versions.

### 2.5 Type / Affordance Heads

| Property | Value |
|---|---|
| **Data** | Labeled slot crops; affordance labels from sim and teleop |
| **Method** | Supervised classification/regression probes over the frozen embedding |
| **Objective** | Classification accuracy, affordance prediction accuracy |
| **Timescale** | Per-slot |

Probes on top of a frozen embedding. Retrained whenever the embedding model is retrained.

### 2.6 Safety Calibration

| Property | Value |
|---|---|
| **Data** | Held-out calibration sets across distribution-shifted scenarios |
| **Method** | Conformal prediction; conformal thresholds on embedding distances |
| **Objective** | Calibrated coverage at nominal levels (90% at 95% nominal) |
| **Timescale** | Post-training calibration pass |

Recalibrated after any Tier 1 or Tier 2 retrain. Independent of entity knowledge store.

---

## 3. Tier 2 — Dynamics and Policy

These components learn from interaction data. They consume entity embeddings (from Tier 1) and, once the entity knowledge store is active, evolved entity state (which includes the interaction digest). During initial training, they train without the digest. The digest pathway is introduced in the co-training phase (§5).

### 3.1 World Model

| Property | Value |
|---|---|
| **Data** | Sim + real teleop with logged scene + actions + outcomes |
| **Method** | Hybrid: sim physics prior + learned residual conditioned on entity embeddings |
| **Objective (local)** | Next-state prediction accuracy (pose, force, deformation, contact) with calibrated uncertainty |
| **Timescale** | Per-step (5–20 Hz) |
| **Brain analogue** | Cerebellum — forward model calibration from climbing-fiber error signals |

**Local learning signal.** The world model's loss is prediction error against observed next-state. This is the most local and cheapest signal in the system. It does not require downstream reward, planner decisions, or procedure-level outcomes — just "I predicted X, I observed Y."

**Two-phase training:**

1. **Without digest (Tier 2 baseline).** Train the world model on entity embeddings from perception only (current observation, no interaction history). This establishes the baseline: how well can dynamics be predicted from what's visible right now?
2. **With digest (co-training, §5).** Train with digest-augmented entity embeddings. The world model learns to exploit interaction history when it's available. Prediction error on re-engagement episodes (where the entity was interacted with before, left, and returned to) provides the signal for whether the digest contains useful information.

**Per-slot dynamics adapters.** Online few-shot adaptation from observed interactions. Initialized from embedding nearest-neighbor priors. The adapter is a local learning mechanism (cerebellar analogue) that operates at inference time — it does not require offline training but does require a training-time protocol that includes online adaptation in the inner loop so the base model learns to support it.

### 3.2 Policy Substrate

| Property | Value |
|---|---|
| **Data** | All teleop + sim rollouts + DAgger corrections, conditioned on language directives + contracts |
| **Method** | Imitation → sim RL (contract-based reward) → DAgger |
| **Objective** | Match expert behavior (imitation); maximize contract satisfaction (RL); absorb corrections (DAgger) |
| **Timescale** | Per-step to per-episode |
| **Brain analogue** | Primary motor + premotor cortex — motor skill from practice + reinforcement + coaching |

**Staged training (per ARCHITECTURE.md §6.5.4):**

1. **Imitation pretraining.** Surgeon teleop + sim rollouts. The substrate learns to predict expert actions conditioned on scene + directive + contract.
2. **RL refinement in simulation.** Reward: contract satisfaction scores (soft slots) + safety invariant preservation (hard slots). Used after imitation to discover behaviors imitation can't reach and to refine quality.
3. **DAgger fine-tuning.** Once the substrate is assistive, surgeon corrections during real cases are recorded and used as correction data. The long-term flywheel.

**Relationship to entity knowledge store.** The substrate consumes entity embeddings. Like the world model, it initially trains without the digest, then learns to exploit digest-augmented embeddings in the co-training phase. The key signal: does the substrate make better decisions when it has access to interaction history?

**Replay buffers for coverage preservation.** Per ARCHITECTURE.md §6.5.6: when fine-tuning for Phase N+1, replay buffers preserve Phase 1..N coverage to prevent catastrophic forgetting.

### 3.3 Planner — Reasoning Module

| Property | Value |
|---|---|
| **Data** | Teleop demonstrations of goal decomposition, paired with surgeon language |
| **Method** | Imitation + supervised on decomposition labels |
| **Objective** | Correct directive generation, appropriate contract specification |
| **Timescale** | Per-planner-step (0.5–2 Hz) |

### 3.4 Planner — Modulation Policy

| Property | Value |
|---|---|
| **Data** | Teleop sessions with risk events, paired with surgeon behavioral changes |
| **Method** | Supervised on event-to-modulation mappings |
| **Objective** | Correct caution adjustment, attention targeting, time pressure estimation |

### 3.5 Planner — Value Head

| Property | Value |
|---|---|
| **Data** | Sim rollouts + outcome labels |
| **Method** | Supervised regression on rollout returns |
| **Objective** | Accurate value estimation for directive search (MCTS/CEM) |

### 3.6 Substrate Online Fine-Tunes (LoRA)

| Property | Value |
|---|---|
| **Data** | Surgeon-flagged teaching cases |
| **Method** | LoRA-style adapter training, hours-to-overnight cadence |
| **Gate** | Per-directive evaluation suite must pass before deployment |
| **Merge** | Into base substrate at next periodic full retrain |

---

## 4. Tier 3 — Memory Encoding (Digest Update Network)

This is the first entity-knowledge-store-specific training tier. The digest update network learns *how to encode* interaction events into the interaction digest. It has a **local self-supervised objective** that does not require the planner, the consolidation policy, or downstream evaluation.

### 4.1 Architecture Recap

The digest update network is a gated recurrent model (GRU-style or transformer write mechanism) that takes:
- Current interaction digest for an entity
- New signal to consolidate (prediction error, risk assessment, event record, or planner conclusion)
- Entity embedding context

And produces an updated interaction digest.

### 4.2 Local Training Objective: Predictive Compression

The digest update network is trained with a **self-supervised predictive compression** objective inspired by hippocampal encoding. The digest should encode information about past interactions such that it enables prediction of future interactions with the same entity.

**Formal objective.** Given a sequence of interaction episodes with entity `e`:

```
episode_1, episode_2, ..., episode_T  (chronologically ordered)
```

After processing episodes 1..k into the digest, the network should predict:
- The entity's dynamics on re-engagement at episode k+1 (compliance, force response, deformation)
- The world model's prediction error when re-engaging this entity (lower is better)
- Key interaction features of past episodes (reconstruction signal)

This is a three-part loss:

```
L_digest = λ_pred * L_predictive + λ_recon * L_reconstruction + λ_reg * L_regularization
```

**L_predictive (primary).** After processing episodes 1..k, predict the entity's dynamics at the next contact episode. Measured against ground-truth observed dynamics. This is the key signal: does the digest carry useful forward-looking information?

**L_reconstruction (auxiliary).** From the current digest, reconstruct summary statistics of past episodes (mean force, contact count, compliance estimate, bleeding events). This prevents the digest from collapsing to a trivial representation. Analogous to hippocampal replay enabling episodic recall.

**L_regularization.** Digest dimensionality and capacity regularization. Prevent the digest from growing unbounded information content. Fixed-size representation with information bottleneck.

### 4.3 Training Data Construction

Training episodes are extracted from the event archive (or, before the system is deployed, from sim rollouts and teleop logs):

1. Select a procedure and an entity that was interacted with multiple times.
2. Extract the chronological sequence of interaction episodes for that entity.
3. For each prefix of the sequence, compute the target: what happened at the next interaction?
4. Train the digest update network to process the prefix and predict the next interaction.

**Sim-heavy.** Early training is dominated by sim data because it provides:
- Complete ground-truth dynamics at every point
- Controllable diversity of tissue behaviors
- Arbitrary-length procedures with many re-engagement events
- The ability to create "counterfactual" episodes (what if we'd re-engaged with a different action?)

### 4.4 Training Protocol

1. **Freeze all Tier 1 and Tier 2 components.** The digest update network trains with a fixed embedding space, fixed world model, and fixed perception.
2. **Generate training sequences.** From sim + logged real data, extract per-entity interaction episode sequences.
3. **Train the digest update network** with the predictive compression objective (§4.2).
4. **Evaluate.** On held-out procedures, measure: does the digest improve next-interaction prediction vs. a no-digest baseline? If not, the encoding is not useful — iterate on architecture/objective.

### 4.5 No Planner Dependency

The digest update network at this stage processes **all** interaction events for an entity, not a planner-selected subset. The planner's consolidation gating is a separate mechanism trained in Tier 4. At Tier 3, we learn the best possible encoding assuming all events are consolidated. Tier 4 then learns which events to select.

This separation (hippocampus encodes everything; PFC decides what's worth remembering) means:
- Tier 3 can be trained without a planner
- Tier 3 provides an upper bound on digest quality (all events encoded)
- Tier 4 learns to approximate this upper bound with a subset of events

### 4.6 Brain Analogue

Hippocampal encoding. The hippocampus forms episodic memories quickly, with high fidelity, from local synaptic signals (Hebbian/BTSP). It does not wait for the PFC to evaluate whether the memory is worth forming. The quality of encoding is evaluated later during consolidation (sleep replay). Similarly, the digest update network learns to encode interaction events well — the question of which events to encode is deferred to Tier 4.

---

## 5. Co-Training Refinement: Digest in the Loop

After Tier 3 produces a functioning digest update network, the downstream consumers (world model, policy substrate, planner) need to learn to *read* the digest. This is not a hard freeze/unfreeze boundary — it's a gradual co-adaptation inspired by how cortical systems adapt to improving inputs from other regions.

### 5.1 Mechanism: Digest Dropout

Introduce the digest into downstream consumers' inputs with **stochastic dropout**:

- With probability `p_digest` (starting at 0, annealed to ~0.8 over training), the entity embedding includes the interaction digest.
- With probability `1 - p_digest`, the entity embedding is perception-only (no digest).

This accomplishes three things:
1. **Backward compatibility.** Consumers never become fully dependent on the digest — they maintain the ability to function with perception alone (important for new entities with no interaction history).
2. **Gradual exploitation.** As `p_digest` increases, consumers learn to exploit digest information when available.
3. **Diagnostic signal.** The gap in performance between digest-present and digest-absent tells you whether the digest is carrying useful information.

### 5.2 Co-Training Losses

Each consumer keeps its original loss (world model: prediction accuracy; policy: imitation + RL; planner: directive quality). No new losses are introduced. The consumers simply see richer input when the digest is present and learn to use it if it helps.

### 5.3 What to Watch For

**Representation collapse.** If performance with digest present equals performance with digest absent across all consumers, the digest is not encoding useful information. Go back to Tier 3 and improve the encoding objective.

**Consumer override.** If consumers learn to ignore the digest (e.g., by learning a near-zero attention weight on digest dimensions), the information isn't useful in its current form. Check: is the predictive compression objective (Tier 3) well-aligned with what consumers need?

**Catastrophic interference.** If consumer performance *degrades* when the digest is introduced, the digest is providing confusing rather than helpful information. Reduce `p_digest`, investigate which digest dimensions cause interference, and iterate on Tier 3.

### 5.4 Brain Analogue

Motor cortex co-adapting with PFC. When prefrontal representations improve (over weeks of learning), motor cortex doesn't need to be retrained from scratch — it adjusts to the richer input through its own local plasticity. Our consumers (world model, policy) similarly adjust to digest-augmented embeddings through their own local losses. The dropout mechanism ensures no consumer becomes brittle to digest availability, just as no motor program becomes inoperable when PFC is temporarily unavailable (lesion studies show degraded but not absent motor function).

---

## 6. Tier 4 — Memory Policy (Consolidation via Offline RL)

The consolidation policy learns *what's worth remembering*: which events should be consolidated into the interaction digest, at what priority, and when. This is the planner's memory-gating role, and it is the hardest training problem in the system.

### 6.1 Why Not Supervised?

There is no ground-truth label for "this event should be consolidated." The value of consolidation is defined by downstream outcomes — did remembering this information help? This makes it fundamentally a decision problem, not a classification problem. RL is the right framing.

### 6.2 Why Offline RL?

The consolidation policy cannot be trained online during actual surgery. The feedback loop is too slow (the value of consolidation manifests minutes later), the stakes are too high, and the environment is non-repeatable. Instead, we train offline on logged procedures, using hindsight information that was not available at consolidation time.

### 6.3 Hindsight Reward Design

After a procedure completes (or a sim episode ends), we have information that was unavailable at decision time:

- **Which entities were re-engaged, and when.** If entity E was last interacted with at T=5min and re-engaged at T=30min, we know the temporal gap and the context of re-engagement.
- **World model prediction accuracy at re-engagement.** How surprised was the world model when re-engaging entity E? If the digest contained useful calibration, prediction error should be lower.
- **Policy quality at re-engagement.** Did the policy make better decisions when re-engaging an entity whose digest was well-maintained? Measured by contract satisfaction, surgeon corrections needed, safety events.
- **What the digest contained vs. didn't.** We can compare the actual digest (from whatever consolidation policy was active) against counterfactual digests (what if we'd consolidated more? less? different events?).

**Hindsight reward for a consolidation decision at time T about entity E:**

```
R(consolidate_event_X_for_entity_E_at_T) =
    Σ over future re-engagements of E at T' > T:
        γ^(T'-T) * [
            α * ΔPredictionAccuracy(E, T')   # world model was more accurate
          + β * ΔPolicyQuality(E, T')         # policy made better decisions
          - κ * ConsolidationCost              # compute/capacity cost of this write
        ]
```

where:
- `ΔPredictionAccuracy` = prediction error without this consolidation minus prediction error with it (positive means consolidation helped)
- `ΔPolicyQuality` = contract satisfaction score with this consolidation minus without it
- `ConsolidationCost` = a small fixed cost per consolidation, encoding the preference for sparser memory writes
- `γ` = temporal discount (the further away the re-engagement, the less this consolidation decision mattered)
- `α, β, κ` = weighting hyperparameters

**The counterfactual.** Computing `ΔPredictionAccuracy` and `ΔPolicyQuality` requires comparing "digest with this event consolidated" vs. "digest without it." Because the digest update network is a differentiable recurrent model (Tier 3), we can run two forward passes through the same procedure: one with the event consolidated, one without. The difference in downstream metrics is the hindsight reward.

### 6.4 RL Formulation

| Element | Design |
|---|---|
| **State** | Planner state + accumulated signals since last consolidation cycle + entity digests |
| **Action** | For each pending signal per entity: consolidate at priority P ∈ {high, medium, low, drop} |
| **Reward** | Hindsight reward (§6.3), computed post-episode |
| **Episode** | One full procedure, replayed from event archive or sim |
| **Horizon** | Full procedure length (tens of minutes to hours) |
| **Algorithm** | Offline RL — conservative Q-learning (CQL) or decision transformer on logged trajectories |
| **Training data** | Event archive + logged consolidation decisions + downstream metrics |

### 6.5 Practical Training Protocol

1. **Collect procedure logs.** Run the system (sim or real, with Tiers 1–3 active and the digest update network encoding all events, i.e., no selective consolidation). Log everything: all signals, all digests, all world model predictions, all policy decisions, all outcomes.

2. **Compute hindsight rewards.** For each procedure, for each entity that was interacted with multiple times:
   - Identify all consolidation-eligible events
   - For each event, compute the counterfactual reward: what was the downstream impact of having this event in the digest?
   - This produces a dataset of (state, action=consolidate/drop, reward) tuples

3. **Train the consolidation policy.** Using offline RL on the hindsight-labeled dataset. The policy learns: given the current planner state and accumulated signals, which events for which entities should be consolidated?

4. **Evaluate.** Run the full system with the learned consolidation policy active. Compare against:
   - Baseline: no digest (perception only)
   - Upper bound: consolidate everything (Tier 3 alone, no gating)
   - Heuristic: consolidate on prediction error > threshold

   The learned policy should approach the upper bound's quality while consolidating far fewer events (matching the sparsity of the heuristic but with better selection).

### 6.6 Curriculum for Consolidation Policy

The consolidation policy faces a cold-start problem: early procedures may not have enough diversity for the policy to learn meaningful selection. Training curriculum:

1. **Sim procedures with forced re-engagement.** Design sim scenarios where the robot must interact with the same entities multiple times over varying gaps. This maximizes the signal for whether consolidation helped.
2. **Varied tissue diversity.** Some entities should be "boring" (standard compliance, no surprises) and some "interesting" (unusual compliance, bleeding tendency, surgeon corrections). The policy should learn to consolidate more for interesting entities.
3. **Strategic context variation.** Same entity interactions under different surgical goals. The policy should learn that the same prediction error matters more when the planner is about to re-engage that entity.
4. **Increasing procedure length.** Start with short episodes (5 minutes) where credit assignment is easy, then extend to full procedure length as the policy stabilizes.

### 6.7 Brain Analogue

Sleep replay with reward-tagged episodes. During sleep, the hippocampus replays episodic memories. Replay is biased toward episodes that were associated with reward prediction errors (dopaminergic tagging at encoding time). The PFC updates its consolidation policy based on which replayed episodes turned out to be important for future behavior. Our offline RL is functionally identical: replay logged procedures, evaluate which consolidation decisions mattered (hindsight reward), and update the consolidation policy.

The key insight from neuroscience: the brain's consolidation policy improves over a lifetime but starts with simple heuristics (novelty, arousal, reward proximity). Our training curriculum mirrors this: start with simple scenarios where consolidation value is obvious, then generalize to complex strategic contexts.

---

## 7. Event Archive as Training Infrastructure

The entity knowledge store plan describes the event archive as serving "rare queries, post-hoc analysis, audit, and training data pipeline." This section specifies the training role in detail.

### 7.1 What the Event Archive Must Log

For the training strategy to work, the event archive must record more than raw interaction events. Required fields per event record:

| Field | Purpose in training |
|---|---|
| **Raw event** (contact, force transient, state change, etc.) | Tier 3 input (digest update network training) |
| **Entity ID (slot ID)** | Episode construction for Tier 3/4 |
| **Timestamp** | Temporal ordering and gap computation |
| **World model prediction at this timestep** | Hindsight reward computation (Tier 4) |
| **World model prediction error** | Consolidation signal and reward baseline |
| **Active planner state** | State for consolidation policy (Tier 4) |
| **Consolidation decision (if any)** | Logged action for offline RL (Tier 4) |
| **Digest state before and after (if consolidated)** | Counterfactual computation (Tier 4) |
| **Downstream metrics window** | Policy quality, contract satisfaction, surgeon corrections in a temporal window after this event |

### 7.2 Event Archive vs. Case Log

The case log (ARCHITECTURE.md §8.1.3) logs per-frame system state for surgeon UI, audit, and training-data curation. The event archive is a *superset* in temporal resolution for significant events and a *subset* in frame coverage (it doesn't log every perception frame). They can share storage infrastructure but serve different consumers:

- **Case log** → training for Tier 1 (perception) and Tier 2 (imitation, RL) components
- **Event archive** → training for Tier 3 (digest encoding) and Tier 4 (consolidation policy)

### 7.3 Sim-Generated Training Events

The event archive for training need not come only from real procedures. The simulator (SOFA-backed, ARCHITECTURE.md §8.3) can generate synthetic procedures with:
- Controlled re-engagement patterns (entity interacted at T, left, re-engaged at T+Δ)
- Known ground-truth dynamics (the simulator knows exact compliance, bleeding rate, etc.)
- Diverse tissue behaviors (domain randomization of material parameters)
- Forced diversity of events (normal, surprising, high-risk)

Sim-generated events are the primary training data source for Tiers 3 and 4 before enough real procedures are logged. Domain randomization ensures the digest update network and consolidation policy don't overfit to sim dynamics.

---

## 8. Updated Per-Component Training Matrix

This extends ARCHITECTURE.md §8.2 with entity knowledge store components and tier assignments.

| Component | Tier | Training data | Method | Local objective |
|---|---|---|---|---|
| Action stream encoder | 1 | Sim + real teleop | Self-supervised + supervised tracking | Tracking accuracy, next-frame prediction |
| Semantic stream encoder | 1 | Public surgical video + real teleop | Foundation model fine-tune | Segmentation IoU |
| Slot attention | 1 | Real teleop + sim | Supervised (sim), semi-supervised (real) | Slot assignment accuracy |
| **Joint embedding model** | 1 | Multi-modal pairs, all modalities | Multi-modal contrastive (CLIP-style) | Cross-modal retrieval, probe accuracy |
| Type / affordance heads | 1 | Labeled slot crops | Supervised probes over embedding | Classification accuracy |
| **World model** | 2 | Sim + real teleop with scene + actions | Physics prior + learned residual | Next-state prediction accuracy |
| Per-slot dynamics adapter | 2 | Online interaction observations | Online few-shot adaptation | Per-entity prediction accuracy |
| **Policy substrate** | 2 | Teleop + sim + DAgger | Imitation → RL (contract reward) → DAgger | Expert match / contract satisfaction |
| Substrate LoRA fine-tunes | 2 | Surgeon-flagged cases | Fast adapter training | Per-directive evaluation pass |
| Substrate confidence head | 2 | Substrate predictions vs. outcomes | Self-supervised + calibration | Calibration accuracy |
| Planner — reasoning module | 2 | Teleop goal decomposition pairs | Imitation + supervised | Directive quality |
| Planner — modulation policy | 2 | Risk event sessions | Supervised on event-to-modulation | Modulation accuracy |
| Planner — value head | 2 | Sim rollouts + outcomes | Supervised regression | Value estimation accuracy |
| Risk / conformal | 1 | Held-out calibration set | Conformal calibration | Coverage at nominal levels |
| OOD detector | 1 | Embedding distances on training data | Conformal thresholds | Detection accuracy |
| **Digest update network** | 3 | Per-entity interaction sequences (sim + real) | Self-supervised predictive compression | Next-interaction prediction, reconstruction |
| **Consolidation policy** | 4 | Full procedure logs with hindsight reward | Offline RL (CQL / decision transformer) | Hindsight consolidation reward |

---

## 9. Training Integration with Phasing

Each project phase (ARCHITECTURE.md §10) introduces entity knowledge store capabilities incrementally. The digest and consolidation policy are not needed for early phases and should not gate them.

| Phase | EKS training status | Notes |
|---|---|---|
| 0 | No EKS. Train Tier 1 + Tier 2 only. | Foundation + dynamics + policy without interaction memory |
| 1 (peg transfer) | No EKS needed. Rigid objects have trivial dynamics that don't benefit from interaction memory. | Focus on substrate coverage and world model for rigid bodies |
| 2 (suturing in sim) | **Begin Tier 3.** Sim suturing provides rich re-engagement data (needle driven through tissue multiple times). Train digest update network on sim data. | First real test: does digest improve needle–tissue interaction prediction? |
| 3 (suturing on phantom) | **Tier 3 validation on real data.** Does the sim-trained digest transfer to phantom? | Co-training (§5) begins: world model and policy learn to read the digest |
| 4 (resection in sim) | **Begin Tier 4.** Soft-tissue resection creates diverse re-engagement patterns. Train consolidation policy on sim resection procedures. | This is where interaction memory matters: tissue compliance varies, bleeding tendency must be tracked |
| 5 (cadaveric narrow-task) | **Tier 4 validation on real data.** Real tissue diversity tests the consolidation policy's generalization. | Event archive begins accumulating real procedure data |
| 6+ (composition) | **Full EKS active.** Multiple entities, cross-directive composition, long procedures. | The consolidation policy's ability to manage memory across many entities under diverse conditions is stress-tested |

**Rollback path at each phase:** If the EKS components don't help (digest doesn't improve downstream metrics), the system still functions — it reverts to perception-only entity embeddings (the Tier 2 baseline). The EKS is an enhancement, not a dependency. This is enforced by the digest dropout mechanism (§5.1): consumers are always trained to work without the digest.

---

## 10. Failure Modes and Mitigations

### 10.1 Digest Representation Collapse

**Symptom:** Digest converges to a constant vector regardless of interaction history.
**Cause:** Predictive compression objective too weak; insufficient diversity in training interactions.
**Mitigation:** Increase weight on reconstruction loss (L_reconstruction in §4.2). Add diversity-promoting regularization (maximize mutual information between digest and interaction sequence). Ensure training data includes entities with genuinely varied dynamics.

### 10.2 Consolidation Policy Learns "Consolidate Everything"

**Symptom:** The policy assigns high priority to all events, providing no useful selection.
**Cause:** Consolidation cost κ too low; or the digest capacity is large enough that consolidating everything doesn't hurt.
**Mitigation:** Increase κ. Reduce digest dimensionality to force capacity constraints. The whole point of the consolidation policy is to work within a capacity budget — if the budget is unconstrained, the policy trivially consolidates everything.

### 10.3 Consolidation Policy Learns "Consolidate Nothing"

**Symptom:** The policy drops all events, digest remains at initialization.
**Cause:** Hindsight reward too noisy or too sparse (re-engagements are too rare in the training data for the reward signal to be meaningful).
**Mitigation:** Increase sim training scenarios with frequent re-engagement. Use reward shaping: intermediate reward for world model accuracy improvement even without full re-engagement. Curriculum starts with short episodes (§6.6).

### 10.4 Sim-to-Real Gap in Consolidation

**Symptom:** Consolidation policy trained on sim doesn't generalize to real tissue dynamics.
**Cause:** Sim dynamics don't capture the diversity of real tissue behavior, so the policy learns sim-specific significance patterns.
**Mitigation:** Domain randomization of material parameters in sim. As real data accumulates (Phase 5+), mix sim and real data in Tier 4 training. The digest update network (Tier 3) may be more robust to sim-to-real gap than the consolidation policy (Tier 4) because predictive compression is a more universal objective.

### 10.5 Temporal Credit Assignment Failure

**Symptom:** Hindsight reward is too noisy for the consolidation policy to learn from (too many confounding factors between consolidation and re-engagement).
**Cause:** Long temporal gaps between consolidation and evaluation, combined with many other system components changing in between.
**Mitigation:** Start with short episodes where the gap is minutes, not hours. Use per-entity attribution (the reward for consolidating event X for entity E only considers re-engagements of entity E, not global metrics). If necessary, use a learned value function that predicts consolidation value from local features (entity state, event type, planner context), bootstrapped from the hindsight reward, rather than relying on raw hindsight reward alone.

### 10.6 Digest Corruption

**Symptom:** A bad consolidation corrupts the digest, causing downstream failures.
**Mitigation per ARCHITECTURE.md open question #5:** Maintain a digest checkpoint from before each consolidation. If downstream metrics (world model error, policy confidence) degrade sharply after a consolidation, roll back the digest to the checkpoint. The event archive enables re-consolidation with a corrected policy. At inference time, this is a simple state management mechanism (store previous digest, compare metrics, revert if needed).

---

## 11. Open Training Questions

These are specific to the training strategy and complement the open design questions in the entity knowledge store plan.

1. **Digest dimensionality.** How large should the interaction digest be? Too small → information bottleneck limits what can be encoded. Too large → consolidation policy has no incentive to be selective. Initial estimate: same dimensionality as the entity embedding (d ≈ 512), revisable based on Tier 3 experiments.

2. **Offline RL algorithm choice.** CQL, IQL, decision transformer, or something else for the consolidation policy? The trajectory is long (full procedure), the action space is structured (per-entity, multi-priority), and the reward is computed in hindsight. Decision transformer may be natural since it conditions on desired return.

3. **Counterfactual computation cost.** Computing the hindsight reward requires counterfactual forward passes through the digest update network and (optionally) world model. For long procedures with many events, this could be expensive. Approximations: sample a subset of events for counterfactual evaluation; use influence functions instead of full forward passes; batch counterfactual computation.

4. **Digest initialization for new entities.** When a new entity is first perceived, what is its initial digest? Training distribution average? Zero vector? Nearest-neighbor in embedding space? This affects both Tier 3 training (what's the starting point for the recurrence?) and runtime behavior (how does the system handle entities it's never interacted with?).

5. **Digest transfer across procedures.** Should interaction history from previous procedures on the same patient carry over? If yes, the digest becomes a cross-procedure patient model — powerful but raises questions about staleness, versioning, and consent. If no, the digest resets per procedure and patient-specific learning is limited to within-procedure. Initial recommendation: no cross-procedure transfer; revisit once the within-procedure mechanism is validated.

6. **Co-training schedule.** How to schedule the digest dropout annealing (§5.1)? Linear annealing? Staged? Conditioned on digest quality metrics? Needs empirical tuning.

7. **Online consolidation policy adaptation.** The offline-trained consolidation policy may encounter novel situations at inference time. Should there be an online adaptation mechanism (meta-RL)? Or is the policy stable enough that offline retraining with accumulated data is sufficient? Initial recommendation: no online adaptation for the consolidation policy; update through the standard retraining flywheel.

---

## 12. Relationship to Existing Training Documents

This document extends and is consistent with:

- **ARCHITECTURE.md §8.2** — the per-component training table, which this document extends with EKS components (§8 above)
- **ARCHITECTURE.md §6.5.4** — substrate training, which this document references and does not replace
- **ARCHITECTURE.md §6.4.2** — embedding training objectives, which this document treats as Tier 1
- **ARCHITECTURE.md §10** — phasing, which this document maps to EKS training phases (§9 above)
- **ARCHITECTURE_CONDENSED.md §4.10** — training flywheel, which this document extends with event archive as training infrastructure (§7)
- **TRAINING_PIPELINE.mmd** — training pipeline diagram, which should be updated to include Tier 3/4 components and the event archive → offline RL pathway

---

## 13. Summary

The system trains like the brain: each component learns from local signals at its own timescale, and interconnections emerge at inference time. The entity knowledge store introduces two new trainable components (digest update network, consolidation policy) and one new training mechanism (offline RL with hindsight reward over the event archive). The existing modular training approach is preserved and extended, not replaced.

The critical architectural decision: **separate the encoding mechanism from the encoding policy.** The digest update network learns *how to write* (Tier 3, self-supervised). The consolidation policy learns *what to write* (Tier 4, offline RL). This mirrors the hippocampus/PFC split in the brain and breaks the otherwise-circular training dependency into two tractable local problems.
