# Autonomous Surgery Architecture Specification

**Status:** Draft v0.8
**Last updated:** 2026-05-02
**Scope:** Reference architecture for the autonomous neurosurgery program at Neuroarm
**Audience:** Engineers building the system, leadership setting expectations, researchers identifying contribution surfaces

**Executive summary:** [ARCHITECTURE_CONDENSED.md](ARCHITECTURE_CONDENSED.md)

**Safety terminology (normative, v0.8):** Use **safety evaluator** for the dedicated safety path as a whole (Layer 1 physical invariants + Layer 2 async assessment + sync command gate; §5.10). **Async safety assessment** publishes the **SafetySurface** from entity state (~10–20 Hz). The **sync command gate** checks every command against that surface plus Layer 1 (100–500 Hz). The legacy term **safety filter** means the **sync command gate** / final command gate only; avoid using it for the full evaluator. Surgeon **constraint types** and contract **hard slots** remain deterministic kinds; entity-informed geometry and envelopes come from the SafetySurface, not from bypassing hard commitments.

---

## 0. Changelog and rationale

### v0.8.1 (2026-05-02) — Documentation alignment with two-phase safety evaluator (no architectural change)

**Change.** Brought §2–§4, §5.2 and related layer text, §5.9 system-attention diagrams, §9 status table, §11 open-question wording, and §12 glossary into alignment with the v0.8 safety evaluator: updated §4 diagram (SafetySurface → planner, entity knowledge store, no standalone risk-system box, no RAG in planner bundle), reframed §3.2–§3.3 and §3.7 table, replaced stale “safety filter only / unchanged from v0.2” claims in normative sections, and annotated v0.7 changelog text superseded by v0.8.

**Motivation.** The v0.8 changelog and §5.10 were authoritative, but older paragraphs and figures still described a monolithic safety filter and separate risk system.

### v0.8 (2026-04-30) — Safety evaluator absorbs risk system; entity-informed assessment replaces formal independence claim

**Change.** The safety filter (claimed to be formally independent from learned representations) and the risk system (separate async monitor) are unified into a two-phase safety evaluator that honestly acknowledges its dependence on entity state while remaining dedicated to safety.

**Motivation.** The current architecture contains a contradiction: section 5.10 claims the safety filter "does not read learned signals," but the same section says "brain-shift compensation continuously warps no-go regions to current anatomy." This warping necessarily passes through perception encoders, entity binding, and world model deformation predictions. The filter's geometric inputs (poses, no-go regions after brain shift) are model-derived, not ground-truth independent observations. This makes the formal-independence claim false and prevents the safety system from being informed by safety-relevant context (bleed history, cumulative tissue stress, spatial learnings about critical regions) that only exists in entity embeddings and interaction digests.

**New architecture: Two-phase safety evaluator.**

**Layer 1 (Physical invariants):** Genuinely formal, model-independent. Joint limits, workspace bounds, hardware force limits, singularity avoidance, communication watchdog. Small, trivially verifiable, unchanged from current design.

**Layer 2 (Entity-informed assessment):** Two execution phases:

- **Async assessment phase (~10-20 Hz):** Absorbs current risk system capabilities (conformal prediction, per-entity risk scoring, escalation triggers, direct override path). Reads entity embeddings, interaction digests, world model deformation predictions, surgeon tags. Computes and publishes a **safety surface** — the pre-computed, entity-informed constraint landscape: deformation-compensated no-go geometry, context-aware force envelopes (entity that bled → tighter threshold from interaction digest), per-entity risk scores, anomaly flags. Exposes interpretable signals to planner: per-entity safety margins, safety gradient, which entities constrain current action and why, calibrated confidence. Retains ≤50 ms direct override path for critical threshold crossings.

- **Sync command gate phase (100-500 Hz):** Fast and thin. Reads pre-computed safety surface + Layer 1 invariants. Checks each command against deformation-compensated no-go SDFs, context-aware envelopes, risk thresholds. Decides pass/project/veto. No heavy inference per tick—just geometric/envelope checks against surfaces pre-computed by async phase. Retains fail-closed default, checks every command regardless of source, immediate surgeon override.

The **safety surface** is the inspectable intermediate artifact: what gets logged for audit, what the planner reads for reasoning, what the surgeon UI visualizes.

**Verification story.** Layer 1 and sync gate logic are formally verifiable. The safety surface is produced by calibrated learned assessment, evaluated per phase, and conservative by design (uncertain → tighter constraints, unknown → veto). System safety depends on async assessment calibration quality, itself a safety-critical evaluation surface with phase-gating and dedicated evaluation story separate from policy training.

**What changes.** Safety filter → two-phase evaluator. Risk system absorbed into async assessment. Safety surface published as intermediate artifact. Bidirectional planner communication (evaluator reports landscape, planner reasons over it). Honest acknowledgment of model dependency.

**What stays.** Safety is a dedicated path separate from policy. Fail-closed default. Surgeon override unconditionally authoritative. Checks every command. Both layers check every command path.

**Sections affected:** §3.5 (one-way door revised—dedication/separation, not formal independence), §3.7 (soft/hard separation reframed—discipline is dedication + interpretability + conservatism + training separation), §4 (diagram updated—safety evaluator box, entity_store edge, safety_surface → planner edge), §5.8 (risk system absorbed; brief note + redirect to 5.10), §5.10 (full rewrite—two-phase evaluator with Layer 1 + async + sync architecture, SafetySurface interface, interpretable signals, new verification story, performance contracts), §5.9 (system attention references updated from risk system to safety evaluator), §11 (new open questions: safety calibration protocol, safety head training separation from policy, surface update rate tuning, async/sync boundary decisions), §12 (glossary: safety evaluator, safety surface, physical invariant layer, async safety assessment, sync command gate).

**Backward compatibility.** The external API is stable: every command still passes through a safety gate, surgeon still has immediate override, safety surface replaces the decoupled constraint surface as a more informative intermediate. Code depending on the safety filter's interface changes minimally; the time scale of input to decision-logic changes (some reasoning moves from per-command to per-cycle async), but the semantics are preserved.

---

### v0.7 (2026-04-29) — Entity Knowledge Store replaces snapshot scene graph; planner-gated consolidation

**Change.** The scene graph (a snapshot data structure overwritten each perception cycle) is replaced with an evolving entity knowledge store that accumulates patient-specific interaction history through planner-gated consolidation. Each persistent entity (identified by slot ID) now carries current observation (refreshed every cycle), interaction digest (distilled understanding of past interactions), entity embedding (fuses observation + digest), and pre-computed priors (consumer-specific derivations).

The system uses a two-path input model inspired by how the prefrontal cortex modulates hippocampal memory encoding:
- **Direct path (ungated):** Perception refreshes current observations (~10-20 Hz); surgeon corrections write unconditional overrides.
- **Consolidated path (planner-gated):** World model prediction errors, risk assessments, and event records flow to the planner as input signals. The planner decides what gets consolidated into each entity's interaction digest, at what priority, and with what framing. The digest update network (a gated recurrent model) executes planner-selected consolidations at 0.5-2 Hz.

All learned components read entity embeddings, which carry distilled interaction history, dynamics calibration, and strategic context. No component re-attends to raw interaction history; the digest absorbs significant information once.

**Why.** Surgery requires both high-resolution short-term memory (milliseconds for controlling active bleeding) and low-resolution long-horizon memory (minutes+ for tumor state tracking and strategic reasoning). A fixed history window forces a false tradeoff. Patient-specific calibration in existing architecture is scattered (adapter weights, SSM recurrent state, risk heads) and lossy. The entity knowledge store makes calibration explicit and shared: when the world model learns "this tissue is unusually compliant," that learning persists where all consumers can benefit. When the planner decides "this region matters because it's adjacent to tagged critical anatomy," that strategic context persists where the policy substrate can use it.

The planner's role as memory gate is principled: it has strategic context, goals, and full situational awareness. It decides significance far better than threshold-based event detection alone. This also gives the planner natural attention-budget modulation—denser consolidation during uncertain situations, sparser during routine execution.

**Complementary learning systems (neuroscience inspiration).** The design maps onto hippocampal-cortical learning in the brain:
- Event archive = hippocampal fast episodic storage (high resolution, every detail)
- Interaction digest = neocortical slow consolidated understanding
- Planner = prefrontal strategic modulation of what gets remembered
- World model prediction errors to planner ≈ dopaminergic prediction-error signaling
- Risk signals to planner ≈ amygdala arousal/salience
- Surgeon corrections ≈ external authoritative teacher (no brain analogue)

**Safety decoupling preserved (v0.7 wording; superseded by v0.8).** *The mechanism described in the next sentences applied before the two-phase safety evaluator. In v0.8, **async safety assessment** reads entity embeddings and interaction digests to publish a **SafetySurface**; the **sync command gate** enforces it plus Layer 1. Learned signals may *tighten* constraints or *veto*; they must not *relax* surgeon-authored hard commitments — that separation principle remains (see ARCHITECTURE_CONDENSED §4.3, §5.10 here).* At v0.7 we stated: the safety filter did not read the entity knowledge store (contains learned representations and strategic context). It read a separate, decoupled surface: current poses from perception, surgeon-authored constraints, force/velocity envelopes, formal predicates. Learned signals flowed through the risk system (now absorbed into async assessment, §5.10) to *raise* hard stops, never to *relax* them.

**Three new components / mechanisms:**
1. **Digest update network.** Gated recurrent model (GRU or transformer-based write mechanism) that consolidates selected signals into updated interaction digests. Runs ~0.5-2 Hz, only when planner signals significant events.
2. **Event archive.** Append-only store of raw high-resolution interaction records indexed by slot ID + event type + timestamp. Not routine consumed at inference; exists for rare queries, audit, post-hoc analysis, and training-data pipeline.
3. **Planner-gated consolidation controller.** Logic layer in the planner that receives signals, decides consolidation priority, and triggers digest updates. Deterministic, not learned.

**Sections affected:** §3.1-3.2 (system overview rewritten around entity knowledge store), §4 (diagrams updated), §5.2 (entity knowledge store detailed design), §5.3 (planner gains memory-gating role), §5.4 (world model outputs to planner for consolidation), §5.8 (risk system outputs to planner for consolidation), §6.5-6.6 (substrate consumes evolved entity embeddings), §8 (case log is event archive for offline training), §10 (per-entity patient-specific calibration now explicit), §11 (open questions: consolidation policy, digest architecture, surgeon override scope), §12 (glossary: entity knowledge store, interaction digest, digest update network, event archive, consolidated path, direct path).

**What stays.** Dual-stream perception, slot attention (object binding), dedicated command gate / safety path (v0.8: two-phase safety evaluator), policy substrate (SSM-Transformer), multi-loop control hierarchy, all off-the-shelf components, safety/learning separation principle. All remain unchanged in intent.

**What changes (deep):** Scene-graph concept replaced with entity knowledge store. Planner gains explicit memory-gating role. Perception becomes direct-path writer (not via planner). World model and risk-related assessments (v0.8: inputs to async assessment and planner) report to planner (not direct writes to digests). Entity embedding now carries three stacked components: current observation (perception-driven), interaction digest (planner-consolidated), pre-computed priors (consumer-specific). This stack enables per-entity dynamics calibration that persists and compounds across interactions within a procedure.

**Backward compatibility.** The external API (what components produce and consume, what the safety evaluator exposes, what the surgeon sees) changes minimally. What changes internally is the representation of shared state and the temporal accumulation mechanism. Code consuming entity embeddings sees richer state; code writing poses/tags sees the same interface.

---

### v0.6 (2026-04-29) — System attention coordination of modulation channels

*Post-v0.8:* References below to a standalone **risk system** describe the pre-v0.8 layout; risk-modulation outputs are produced by **async safety assessment** (§5.10), which publishes `RiskState` into this aggregation.

**Change.** Four independent modulation signals — planner modulation, voluntary attention, risk override, and adaptive halting — are now coordinated through a single "system attention" module that aggregates them into a coherent per-tick modulation state. This replaces independent consumer reading of each signal with a unified interface and single inspection surface.

**The four channels being coordinated:**

1. **Planner modulation.** Bias signals from the slow planner (caution_level, time_pressure, attention_targets) that vary more frequently than full directive re-steps.
2. **Voluntary attention.** Soft cues to the action-stream encoder's compute about which slots to prioritize ("look here").
3. **Risk override.** Direct intervention from **async safety assessment** (safety evaluator; §5.10) when risk exceeds hard thresholds; can escalate or request modulation changes. *(Pre-v0.8 wording referred to a standalone “risk system.”)*
4. **Adaptive halting.** Internal halting signals from the recurrent reasoning module and online monitors (substrate confidence, OOD detection, contract drift), used to vary loop speeds or trigger re-planning.

Previously, each consumer (substrate runtime, planner, fast controller, and the **pre-v0.8 standalone risk monitor** — **now** **async safety assessment**, §5.10) read these signals independently, leading to potential inconsistencies: the substrate could be executing at high confidence while the risk path had raised caution, or the planner could be issuing a new modulation while the reasoning module was halting to re-deliberate.

**System attention module (new in v0.6):**

A lightweight coordination component runs once per tick and produces a unified **SystemAttentionState** that aggregates:

- **Aggregated caution level.** Max of planner caution and **async-assessment** caution (`RiskState`; fail-closed: higher caution wins).
- **Merged attention targets.** Union of planner attention targets and **async-assessment** flagged slots.
- **Time pressure.** Planner-set, but modulated down by **async safety assessment** signals if escalation is near.
- **Halting priority.** Boolean flag set if reasoning or confidence monitoring requests adaptive halting.
- **Escalation reason (if any).** Aggregates reasons from **async safety assessment**, contract violation, or OOD detection into a single escalation event, routed to the planner immediately.

All other components (substrate runtime, fast controller, **async-assessment outputs feeding system attention**) consume this unified state instead of reading signals independently. This reduces:

- **Signal inconsistency:** Components act on the same modulation snapshot.
- **Debugging surface:** One place to inspect "why did the system choose this behavior at this moment?" instead of tracing four independent signal sources.
- **Latency analysis:** All modulation updates are synchronous within a tick; no race conditions between planner and **async safety assessment**.

**Mechanism (not learned):** The system attention module is deterministic, not ML-based. It is a small coordination layer that implements aggregation logic (max, union, priority rules) and routes escalation events. The planner retains full deliberative authority — it decides whether to re-condition, modulate further, or escalate to the surgeon. This is consistency and inspectability, not automation.

**What stays.** The planner, **safety evaluator** (async assessment publishes `RiskState` into this aggregation), substrate, and fast controller retain their responsibilities. The planner still produces PlannerModulation; **async assessment** still publishes risk modulation; substrate runtime still monitors contracts. The system attention module is purely a coordination layer, not a replacement for any of those.

**Sections affected:** §5.5 (planner produces modulation; planner notes system attention dependency), §5.6.3 (contract monitoring now references system attention state), §5.8 (**former** risk-system capabilities absorbed into async assessment; signals route through system attention), new §5.11 (System attention module), §9 (implementation status updated), §11 (open questions: modulation conflict resolution), §12 (glossary: SystemAttentionState).

### v0.5 (2026-04-28) — Episodic memory removed; subtractive simplification

**Change.** The "episodic memory" architectural layer (RAG over past surgical cases, with retrieved cases entering the substrate's conditioning bundle as in-context examples) is removed. It is not replaced by a learned-memory mechanism inside the substrate or by anything else architectural. The system's memory is fully accounted for by:

1. **Substrate weights** — semantic / general knowledge, updated by training.
2. **Substrate recurrent state + persistent slot-level state (entity knowledge store)** — within-procedure working memory.

That's it. There is no inference-time retrieval of past cases. There is no "episodic memory" as a separate architectural component.

**Why.** Inference-time RAG is a substitute for training. Anything in the training set is already in the substrate's weights — RAG-presenting it back is redundant. The cross-procedure use cases that motivated RAG (patient-specific recall, surgeon teaching, rare-case influence) are better served by other mechanisms that already exist or are simpler to add:

- Patient-specific context flows in via the *conditioning bundle* (pre-op imaging, surgical plan, surgeon-tagged anatomy) at procedure start. Not memory; data flow.
- Surgeon teaching is better served by *fast online fine-tuning* (LoRA-style adapters, hours-not-weeks cadence). Better generalization than in-context learning, evaluable against per-directive suites before deployment.
- Rare cases are absorbed by *training-data curation* (the next fine-tune batch) rather than at-inference retrieval.

The case *log* still exists — it's an operational artifact for surgeon UI, audit, training-data curation, and failure analysis. But it lives in the data pipeline (§8), not as an architectural memory layer.

**What's lost.** Imagined benefits of RAG-into-substrate that we walk away from: explicit auditable retrieval traces at inference time (replaced by post-hoc embedding-similarity queries on the case log); zero-latency surgeon teaching (replaced by overnight LoRA fine-tunes — slower but more reliable); cross-procedure substrate recall (substrate weights are the right place; retraining cadence is the right knob). None of these losses is load-bearing.

**Sections affected:** §7 (collapsed to two short subsections), §5.5 (`PlannerOutput` loses `retrieved_cases`; planner components updated), §3.3 (substrate input list updated), §6.5 (substrate inputs and compositional-behavior mechanisms updated), §8.1 (case log added as operational artifact), §8.2 (online fine-tuning added as training mode), §9 (episodic memory row removed), §11, §12.

### v0.4 (2026-04-28) — Soften remaining hard-set sets where safety doesn't require them

**Change.** Four interfaces that were closed-enum or closed-list in v0.3 are softened to continuous / open / mixed structures. Hard guarantees still come from explicit, surgeon-authored, formally specified inputs to the safety filter — that boundary is unchanged. What's softened is everything *outside* that boundary that was unnecessarily constrained.

The four revisions:

1. **Affordance space, not affordance list.** Affordances are now a continuous learned subspace of the slot embedding (affordance space). Named affordance predicates (`graspable_along_axis`, `has_cutting_edge`, etc.) are *labeled regions* of that space — readout heads, not the source of truth. Behavior contracts can reference either named affordances or affordance-space neighborhoods.
2. **Tag schema as closed constraint types + open natural-language content.** Surgeon tags now have two layers: a closed set of *constraint types* (`no_go_region`, `force_limit`, `proximity_alert`, `do_not_action`) that the safety filter consumes deterministically, and open *natural-language content* (e.g., "the proximal segment of vessel_018") for surgeon communication and case-log / training-curation lookup (UI, audit). Surgeons interact in language; the system anchors language to slots; constraints carry both layers.
3. **Directive distribution, not directive vocabulary.** Directives are not an enumerated list. The substrate is trained on a continuous *distribution* of natural-language directives. Phasing measures this distribution's coverage and reliability, not the count of supported "skills."
4. **Mixed behavior-contract schema.** Contracts now have two slot classes: *hard slots* (formally verifiable predicates over interpretable state, consumed by the safety filter) and *soft slots* (free-form natural-language success criteria and quality measures, consumed by the substrate as conditioning and by the runtime via learned heads).

A fifth minor refinement (v0.4 only): episodic memory stores cases as multi-view records keyed on the joint embedding, with the index — not the data — being the fixed schema. *Superseded by v0.5: episodic memory removed entirely.*

**Why.** The same logic that drove v0.2 (embedding-first slots) and v0.3 (substrate-as-policy) applied to layers of the architecture I had left as closed enums when I shouldn't have. Hard-coding closed enums for things that are not safety-load-bearing throws away continuous structure for no defensive benefit. The principle: **hard-set only what formal verification or physical interface requires; learn or open everything else.**

**What stays.** The safety filter's constraint *types* remain a closed, formally specified set. The robot interface, sensor modalities, multi-loop rate structure, and operational modes remain hard-set. The Environment schema and slot schema skeletons remain fixed for engineering tractability. None of these change.

**Sections affected:** §3.1 (slot description updated), §3.7 (extended), §5.2.4 (affordance heads → readouts on affordance subspace), §5.3.1 (slot data model fields restructured), §5.5 (BehaviorContract gains hard/soft slots, directive distribution language), §5.6.3 (contract monitoring across hard/soft slots), §5.10.3 (constraint types vs. tag content), §6.5.6 (bootstrap recast as distribution coverage), §7.2 (multi-view records), §10 (phasing recast), §11 (open questions updated), §12 (glossary updated).

### v0.3 (2026-04-28) — Substrate-as-policy (skills are language, not code)

**Change.** The discrete skill library is removed as an architectural artifact. The action source is now a single compositional policy substrate — a learned neural network that consumes the full context bundle (scene, embeddings, episodic retrievals, language directive, risk state, planner modulation, recent events) and produces commanded actions or short-horizon target trajectories. Skill names persist as language-level conditioning interfaces, as retrieval keys for episodic memory, and as anchors for behavioral contracts. They are not code modules. There is no skill registry, no per-skill state machine, no per-skill implementation, no skill-migration path from classical to learned.

*Superseded by v0.5:* The bundle above described the pre–v0.5 substrate input list. Episodic retrievals and inference-time retrieved cases were removed; skill names remain language conditioning, contract anchors, and **case-log index keys** (audit, curation), not substrate-conditioning retrieval keys.

**Why.** The brain's motor system is not a discrete registry. It is a continuous reinforced policy space where similar patterns share substrate, attention modulates execution at any granularity, and novel situations are handled by composing learned primitives across contexts. A surgeon trained separately on suturing and cauterization can produce a competent suture-during-bleed response without ever having been trained on that specific composition — because the underlying motor system composes natively. A discrete skill library, multi-level or otherwise, cannot do this: novel compositions require new skills, which require new training and validation, which is exactly the brittleness we want to avoid.

This also aligns with where foundation-model robotics is heading (RT-2 → OpenVLA → Pi-zero / π0.5 lineage): one learned policy substrate conditioned on language and context, not N separately-engineered skill modules.

**What stays.** Skill names persist as natural-language tokens the surgeon uses, as conditioning keys for the substrate, and as index keys for case-log queries (audit, curation). Skill contracts (pre/post-conditions, required affordances, forbidden tags, safety invariants) persist as behavioral specifications consumed by the planner for goal monitoring and by the safety filter for hard-constraint envelope selection. The safety filter operates on commanded actions; whether those came from a "skill" or from the substrate's compositional output is invisible to it.

**What's lost (honestly).** Step-by-step debuggability is harder; behavioral testing replaces structural testing; predictability for surgeons requires building a different kind of trust; data demands are substantially higher; the substrate is one giant research effort rather than N parallel skill-engineering efforts.

**Sections affected:** §3.3 (rewritten), §4 (diagram updated), §5.5 (planner output reshaped), §5.6 (full rewrite as substrate runtime), §5.7 (interface to substrate), §6 (substrate section expanded with action policy), §8.2 (training regime), §9 (implementation status), §10 (phasing), §11 (open questions), §12 (glossary).

### v0.2 (2026-04-28) — Embedding-first slot representation

**Change.** The scene graph's slots are now embedding-first: each slot's primary data is a learned multi-modal embedding vector. Type labels, affordances, and per-instance dynamics parameters are *derived* from the embedding (and from observation), not primary. Surgeon-authored explicit tags remain authoritative for hard safety constraints.

**Why.** A closed-enum typed scene graph throws away the continuous structure of the world: within-instance variation (different forceps behave differently), cross-instance similarity (scalpels and osteotomes share affordances), soft membership (a half-resected tumor is partway between intact and void), and per-instance dynamics adaptation (each cottonoid saturates at its own rate). Typed-only systems either ignore this information or proliferate per-type schemas to capture it badly. Embedding-first captures it natively, scales the same code from N types to N+k types without rewriting, and matches how every successful modern foundation-model system works (CLIP, DINOv2, SAM 2, OpenVLA, AlphaFold all use continuous embeddings as substrate with discrete labels as derived projections).

**What stays explicit.** Hard safety guarantees still require deterministic predicates over surgeon-authored tags (e.g., "this specific vessel must not be touched"). Embeddings provide priors and defaults; tags provide guarantees. Both layers are kept; their roles are separated.

**Sections affected:** §3.1, §3.7 (new), §5.2 (new sub-component), §5.3 (slot data model rewritten), §5.4 (world model becomes embedding-conditioned), §5.6 (skills consume affordances + tags, not types), §5.10 (safety filter splits hard vs. soft constraints), §6.4 (new), §7.2, §8.2, §9, §11, §12.

### v0.1 (2026-04-28) — Initial draft

Initial reference architecture: object-centric typed scene graph, multi-loop control, skill library, sim/real parity, formally specified safety filter, bottom-up contract design.

---

## 1. Purpose and scope

This document specifies the end-to-end architecture for an autonomous surgical robotics stack, with neuroArm as the target platform. It is the primary reference for what we build, in what order, with which off-the-shelf components, and where targeted research investments are required.

It is **not**:
- A research paper. Citations are minimal; concepts are stated as design decisions.
- A clinical or regulatory document. Those are downstream.
- A timeline. A separate roadmap document covers phasing and milestones.
- A justification document. Rationale is summarized; deeper reasoning lives in design discussions.

The architecture is designed to support a progression of capabilities — from a manual-dexterity benchmark (FLS peg transfer at human-competitive level) through cadaver-validated narrow-task autonomy (e.g., autonomous retraction-hold, probe-to-target navigation) to autonomous surgical resection over multi-year horizons. The same architecture instantiates all of them by expanding training data, directive distribution coverage, embedding-space coverage, and the **deformable / interaction world-model reliability frontier** — not by rebuilding the stack.

---

## 2. Goals and non-goals

### 2.1 Architectural goals

1. **Single architecture across tasks.** Peg transfer, suturing, resection, retraction-hold all run on the same stack with different object types, skills, and constraints.
2. **Clean separation between learned and engineered components.** Components that benefit from learning are learned; components that require formal guarantees are engineered. No hybrid or hand-wavy boundaries.
3. **Multi-timescale control.** Different decisions happen at different rates and have different architectures, all sharing scene state.
4. **Sim/real parity.** The same code runs against the simulator and the real robot. No bespoke sim-only or real-only logic above the sensor/actuator layer.
5. **Forward-compatibility with foundation-model improvements.** Components like the perception encoder are swappable as the field improves; the architecture does not depend on any one model.
6. **Verifiable safety.** Every commanded action passes through the **safety evaluator** (§5.10): **Layer 1** (physical invariants) and **sync command gate** logic are formally specified and runtime-verified; **async safety assessment** produces a calibrated **SafetySurface** (phase-gated evaluation, conservative when uncertain). Fail-closed defaults and unconditional surgeon override are non-negotiable.

### 2.2 Non-goals

- **End-to-end vision-language-action models.** Pure VLAs lack the structural commitments this architecture requires (multi-loop, formal safety, object-centric state).
- **Replacing Transformer/SSM ML.** We use modern ML components as building blocks. We do not invent new attention mechanisms or backbone architectures.
- **Solving open problems in foundational ML.** We architect around current ML limitations and contribute targeted research only where the architecture itself reveals gaps.
- **Frontier-lab-scale data ambitions.** This stack is designed to work with the data we can plausibly collect (teleop logs, phantom, cadaver, eventually clinical), not internet-scale corpora.

---

## 3. Architectural commitments (one-way doors)

These are decisions that must be locked down before code is written. Reversing any of them after implementation begins is expensive.

### 3.1 Object-centric, embedding-first scene representation

The world state is a graph of object slots — not a global field, not an image, not a single latent vector. Each slot represents a discrete entity in the surgical field (a tool, a tissue region, a cottonoid, a fluid pool, a resection void, etc.) with persistent identity across frames and across occlusion.

**Each slot is embedding-first.** Its primary data is a learned multi-modal embedding vector that captures its perceptual, geometric, and functional properties continuously. Type labels (`Tool`, `IntactTissue`, etc.) are *derived* from the embedding and from observation, with confidences. **Affordances live in a continuous learned affordance subspace** of the slot embedding; named affordance predicates (`graspable_along_axis`, `has_cutting_edge`, etc.) are *labeled regions* of that space, exposed as readout heads — not separate enumerated properties. Per-instance dynamics parameters are *adapted online* per slot. Pose and geometry are tracked explicitly.

**Surgeon-authored explicit tags are kept separately and remain authoritative for hard safety constraints, with two layers:**

- **Constraint type** (closed enum, formally specified): `no_go_region`, `force_limit`, `proximity_alert`, `do_not_action(action_class)`, etc. These are the formal *kinds* of constraint the **sync command gate** enforces (using the SafetySurface and Layer 1; §5.10).
- **Tag content** (open, natural-language, slot-anchored): the surgeon's natural-language reference for what this constraint applies to ("the proximal segment of vessel_018"; "tissue that bled at minute 12"; "the small branch I flagged earlier"). The system anchors language to a specific slot or scene region using the embedding model and recent context; the surgeon visually confirms the anchor before the constraint becomes active.

The constraint type is what the sync gate enforces (with SafetySurface parameterization where applicable); the content is what the surgeon, the planner's reasoning module, the case log (for surgeon UI), and the audit log use. The two-layer tag is how surgeons interact in language while the gate still gets a deterministic constraint kind to enforce.

Spawn, despawn, contact, and tag changes are first-class events.

**Why one-way:**
- Every other component (perception writes to slots, world model reads from them, planner reasons over them, safety evaluator consumes them) depends on this representation. Changing it later requires touching every layer.
- The embedding-first vs. typed-first distinction in particular cascades: an embedding-conditioned world model is a fundamentally different module from N per-type dynamics modules; affordance-based skill matching is a different control flow from type-matching; embedding-keyed **case indexing** (case log, training curation — not substrate conditioning at inference) is a different schema from type-keyed lookup. Switching directions later means rewriting all of these.

### 3.2 Multi-loop, multi-timescale control

Three control loops run at different rates against shared entity/scene state, plus the **safety evaluator** (§5.10): **async safety assessment** (~10–20 Hz) publishes a SafetySurface and modulation-related signals; the **sync command gate** (100–500 Hz) checks every command. Former standalone **risk system** capabilities live inside async assessment (§5.8).

| Loop / path | Rate | Role |
|---|---|---|
| Slow planner | ~0.5–2 Hz | Directive composition, deliberative reasoning, memory consolidation, escalation; reads SafetySurface |
| Substrate runtime | ~5–20 Hz | Substrate forward pass + contract / OOD monitoring + re-conditioning |
| Fast controller | ~100–500 Hz | MPC + visual servoing + force-aware control |
| Safety evaluator | async ~10–20 Hz + sync 100–500 Hz | SafetySurface from entity state; pass/project/veto on every command; Layer 1 invariants; absorbs former risk monitoring (§5.8, §5.10) |

**Why one-way:** A single-loop architecture cannot meet both the latency requirements of fine motor control and the deliberation requirements of directive selection. Retrofitting multi-loop into a single-loop design requires rewriting the controller and planner.

### 3.3 Compositional policy substrate as the action source

The action source is a single compositional policy substrate — a learned neural network that consumes the full context bundle (scene state, slot embeddings, language directive, behavior contract, risk state, planner modulation, surgeon directives, recent events) and produces commanded actions (or short-horizon target trajectories that the fast controller executes). The substrate composes behavior in real time across patterns it has learned during training; it can recombine primitives across what would conventionally be considered separate "skills" (e.g., aspirate-while-watching-for-bleed) without those compositions having been pre-programmed.

**Skills are language, not code.** The names surgeons and the system use (`Aspirate`, `Cauterize`, `Suture`, `Retract`, `MoveToPose`, ...) are *conditioning labels* that bias the substrate's output, *index keys* for the case log (when surgeons or auditors want to find related past cases), and *anchors* for behavioral contracts (which carry hard slots — formally verifiable safety invariants, force envelopes, forbidden constraint-type tags, abort conditions — and soft slots — free-form success criteria, learned quality measures, affordance requirements; see §5.5). They are not function pointers. There is no per-skill code module, state machine, or implementation. A skill is what the substrate *does* when conditioned on its label and its contract, evaluated by whether it satisfies that contract's hard slots and progresses on its soft slots.

**Directives are drawn from a continuous distribution, not an enumerated vocabulary.** The substrate is trained on a continuous distribution of natural-language directives. There is no fixed list of supported directives — the substrate's *reliability frontier* (the region of language space where its behavior is reliable, measured by per-directive evaluation suites and OOD detection) defines what's currently supported. Phases (§10) expand this frontier; they do not extend a registry.

**Safety boundary survives.** The **sync command gate** still sits between every commanded action and the robot: fail-closed, substrate-output-agnostic, immediate surgeon override. v0.8 adds **async assessment** that reads entity state and publishes a **SafetySurface** the gate enforces (heavy reasoning off the hot path). The gate’s logic stays small and verifiable; the SafetySurface is calibrated and phase-evaluated (§5.10). Whether a command came from teleoperation, the planner, or the substrate is invisible to the gate. The substrate is the action source; the safety evaluator is the action gate; training and dedication stay separate from policy optimization.

**Why one-way:**
- The substrate's training data, conditioning interfaces, and architecture are all designed around this commitment. Reverting to a discrete skill library means splitting the substrate into N individually-trained policies, throwing away the compositional generalization that motivated this choice in the first place.
- The planner output schema and the surgeon-directive grounding both change shape under this commitment. Reverting requires re-architecting them.
- This is the foundation-model bet for the action layer. It is the highest-uncertainty architectural commitment in the document. We make it deliberately, with eyes open about what's lost (step-by-step debuggability, structural verifiability of skills, predictable per-skill behavior). The mitigations are output-level safety and behavioral contract monitoring; see §5.6 and §5.10.

### 3.4 Sim and real implement the same `Environment` interface

The simulator and the real robot expose the same observation space, action space, entity-knowledge-store / slot schema, directive distribution, and contract schema. Code above the `Environment` boundary runs unchanged on both.

**Why one-way:** Without this parity, every component is implemented twice (once for sim, once for real) and drifts. Sim becomes useless for development.

### 3.5 Dedicated safety evaluator outside the policy substrate

Safety is a dedicated system path separate from the policy substrate. The safety evaluator checks every commanded action and is subject to formal verification (Layer 1 and sync command gate) and calibrated evaluation (Layer 2 async assessment). Learned entity state informs the safety evaluator's assessment; what is prohibited is the policy substrate influencing the safety evaluator's training or relaxing its constraints.

The evaluator's Layer 1 (physical invariants) is engineered code with formal semantics. Layer 2's async assessment phase reads entity embeddings and publishes an interpretable safety surface; the sync command gate checks commands against this surface and has small, verifiable logic.

**Why one-way:** Clinical autonomy cannot rely on the policy substrate to police itself. Embedding safety reasoning inside the policy collapses the safety/task separation, makes formal verification of safety impossible, and incentivizes the policy to relax constraints for task completion. A dedicated path prevents this. The commitment is to a separate path with clear verification scope — not to the false claim that the safety system is formally independent of all learned representations. Real systems (including this one) depend on learned perception; the discipline is dedication and conservatism, not pretended independence.

### 3.6 Bottom-up contract design

Layers are specified in this order: action space, safety constraints, controller, world model, perception. Each layer's contract (input/output types, latency, accuracy) is engineered first; its implementation is chosen to meet the contract.

**Why one-way:** Top-down design (perception-first) produces components that controllers cannot use. Once perception is built without a controller's contract in mind, it has to be redone.

### 3.7 Separation of safety dedication from task commitment

The system distinguishes safety reasoning (dedicated, interpretable, conservative) from task execution (policy-driven, learned). The separation runs through the safety evaluator's design:

| Source | Soft (probabilistic, learned) | Hard (deterministic, surgeon-authored) |
|---|---|---|
| Slot properties | embedding, affordance space, type distribution, dynamics adapter, learned uncertainty | surgeon tags (constraint type + content), surgeon labels |
| Behavior contract slots | success criteria, quality measures, surgeon-stated intent | safety invariants, abort conditions, force/velocity envelopes, forbidden constraint-type tags |
| Risk inputs | substrate confidence, OOD scores, async-assessment per-slot scores (SafetySurface) | surgeon override, hard-veto path |
| Safety evaluator | Learned entity state informs **async assessment** (SafetySurface); may **tighten** or **veto**; must **not relax** hard commitments or substitute policy optimization for safety training | **Sync gate** enforces deterministic constraint kinds, SafetySurface geometry/envelopes, contract hard slots, Layer 1 |

**Soft priors** (embeddings, affordance vectors, type estimates, substrate confidence, OOD scores, learned risk scores) inform planner reasoning, modulation, substrate conditioning, and contract monitoring. They can flow through the safety evaluator to *raise* caution or trigger veto, but they cannot *relax* the safety evaluator's assessment.

**Hard commitments** (surgeon tags with constraint types, behavior contract hard slots, force/velocity limits, formal no-go geometry) are the explicit inputs the safety evaluator uses. These are deterministic predicates over interpretable state.

**Why one-way:** If the policy substrate could influence what the safety evaluator considers safe, or if safety training optimized for policy-convenience rather than real safety, the system would be unsafe-by-design. The separation is not about preventing learned signals from entering the safety system — entity embeddings must enter to assess real anatomy. The separation is about dedicating safety to safety, training safety separately, preventing the policy from relaxing safety, and keeping safety interpretable and auditable.

---

## 4. System overview

```
                   SURGEON INTERFACE (goal / permission / override)
                                  │
                    SafetySurface │
                   (interpretable)│
                          ▲       │
                          │       ▼
                   ┌──────┴───────────────────────────┐
                   │   SLOW PLANNER  (~0.5–2 Hz)       │
                   │   directive, contract, modulation │
                   │   (+ reads SafetySurface)         │
                   └──────────────┬───────────────────┘
                                  │ conditioning bundle
                                  ▼
                   ┌──────────────────────────────────┐
                   │   POLICY SUBSTRATE RUNTIME (~5–20 Hz)│
                   │   Compositional policy network +  │
                   │   contract monitoring             │
                   └──────────────┬───────────────────┘
                                  │ target trajectories
                                  ▼
                   ┌──────────────────────────────────┐
                   │   FAST CONTROLLER  (~100–500 Hz)  │
                   │   MPC + visual servo + force      │
                   └──────────────┬───────────────────┘
                                  │ RobotCommand
                                  ▼
                   ┌──────────────────────────────────┐
                   │   SAFETY EVALUATOR (v0.8)         │
                   │   Async assessment (~10–20 Hz):   │
                   │     entity state → SafetySurface  │
                   │   Sync command gate (100–500 Hz): │
                   │     surface + Layer 1 invariants  │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                              ┌────────┐
                              │ ROBOT  │
                              │neuroArm│
                              └────┬───┘
                                   │ (sensors)
                                   ▼
                   ┌──────────────────────────────────┐
                   │   PERCEPTION  (dual-stream)       │
                   │   Action stream + Semantic stream │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────────┐
                   │ ENTITY KNOWLEDGE STORE            │
                   │ (embedding-first slots)           │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────────┐
                   │   WORLD MODEL  (embedding-conditioned)│
                   └──────────────────────────────────┘

Entity knowledge store is read by planner, substrate runtime, world model,
and safety evaluator async assessment. World model is read by planner
and feeds consolidation and SafetySurface context. Every commanded
action passes through the sync command gate; surgeon override is
unconditional. (Inference-time RAG / retrieved cases are not in the
runtime bundle; case log is operational only — §8.)
```

---

## 5. Layer specifications

### 5.1 Sensors

| Modality | Rate | Interface | Notes |
|---|---|---|---|
| Stereo microscope | 60–120 Hz | RGB + depth (or stereo pair) | Primary scene perception |
| Tool kinematics | 500 Hz | Joint encoder readings → SE(3) end-effector pose | Direct from neuroArm |
| Force/torque | 1 kHz | 6-axis F/T at end-effector | Contact-aware control |
| Audio | 48 kHz | Mono or stereo | Cautery sounds, surgeon speech |
| Intraop ultrasound | sparse, on-demand | Volumetric | Brain-shift correction |
| Pre-op MRI/CT | static | Volumetric + plan annotations | Anatomy prior |

**Contract:** All streams synchronized to a common clock with ≤1 ms drift. All streams logged continuously to disk.

### 5.2 Perception

Two parallel encoders feeding a unified slot-attention head.

#### 5.2.1 Action stream encoder

**Purpose:** Low-latency geometric perception for the fast control loop.

**Inputs:** stereo microscope frames, kinematics, force.

**Outputs:** tool poses (with covariance), per-tool contact state, tissue surface motion field, force estimates.

**Architecture:** SE(3)-equivariant attention layers (SE(3)-Transformer or Equiformer class). Lightweight; latency-budgeted.

**Relationship to soft tissue.** SE(3) layers encode **rigid-frame** geometry: tool pose, contact frames, and spatial relations transform consistently under global rotation and translation. They do **not** replace a constitutive model of deformable tissue, plastic damage, topology change, or bleeding. Those behaviors are predicted by the embedding-conditioned world model (§5.4), which fuses these geometric features with stereo shape and motion, force/torque, and contact history. Axial tool roll is in SE(3) when kinematics or asymmetric appearance makes it observable; symmetric appearance alone can leave roll weakly determined from vision — kinematics and wrench then carry the signal.

**Latency budget:** ≤15 ms p99 from frame ingestion to entity-knowledge-store / slot update.

**Status:** Off-the-shelf + integration. Streaming adaptation is modest research.

#### 5.2.2 Semantic stream encoder

**Purpose:** Slower, higher-resolution semantic perception for the slow planner and safety evaluator async assessment (context for SafetySurface; §5.10).

**Inputs:** stereo microscope frames.

**Outputs:** per-pixel object segmentation, classification, anatomy labels.

**Architecture:** Foundation model (SAM 2 / MedSAM2 / surgical VLA backbone), fine-tuned on surgical data. Treated as a swappable commodity.

**Latency budget:** ≤150 ms p99.

**Status:** Off-the-shelf, fine-tuned on collected data.

#### 5.2.3 Slot Attention head

**Purpose:** Integrate dual-stream output into object **slots** (slot-level scene state inside the **entity knowledge store**, §5.2). Maintains slot identity across frames and emits spawn/despawn events.

**Inputs:** action stream features, semantic stream features.

**Outputs:** **Scene state** updates — slot creation, slot pose/geometry updates, slot deletion. Each new or updated slot is annotated with an embedding (produced by the joint embedding head, §5.2.4) and pose/geometry from the action stream.

**Architecture:** Variable-cardinality slot attention. Each slot is tracked over time; identity is maintained across occlusion via temporal continuity in the embedding space and pose history.

**Physically overlapping entities must remain separate slots.** A vein draped over a tumor occupies overlapping space and is mechanically coupled to it, but the two have different dynamics (pressurized vessel vs. bulk tissue), different affordances (preserve vs. resect), different risk profiles (catastrophic hemorrhage vs. margin management), and different surgeon constraints (hard no-go vs. active removal target). Merging them would prevent the system from expressing "remove this, preserve that." The per-slot decomposition gives the planner, safety evaluator, and surgeon UI separate handles; the relationship between co-located entities is captured in the right places: interaction coupling in the world model (§5.4), spatial structure in SE(3)-equivariant features (§5.2.1), and strategic reasoning in the planner (§5.5). The dual-stream fusion is critical here — the semantic stream (SAM 2 / MedSAM2) provides the fine-grained anatomical segmentation needed to distinguish a vessel from underlying tissue when the action stream's geometry alone cannot separate them.

**Entity splitting is a known failure mode.** The competitive softmax in slot attention encourages each perceptual feature to be predominantly explained by one slot, but does not strictly enforce one-to-one entity-to-slot mapping. A large or heterogeneous entity (multi-lobed tumor, long vessel) could have different spatial regions claimed by different slots if those regions look sufficiently different in feature space. Mitigations: persistent identity tracking makes sporadic splits unstable across frames; entity-level training supervision penalizes gratuitous splitting; the planner would observe two slots with near-identical embeddings and overlapping spatial extent. Risk factors remain: genuinely ambiguous boundaries (partially resected tissue), under-training, distribution shift. Custom streaming adaptations should include explicit merge/split detection — slot-merging heuristics or a learned merge/split head after the binding pass.

**Latency budget:** ≤30 ms p99.

**Status:** Off-the-shelf base; streaming variable-cardinality is a research investment.

#### 5.2.4 Joint embedding head

**Purpose:** Produce a learned, multi-modal embedding vector for each slot. This is the slot's primary representation; types, affordances, and per-instance dynamics adapters are computed from it.

**Inputs:** per-slot crop of action-stream features, semantic-stream features, recent kinematic history (when the slot is a tool), recent contact/force history (when the slot is involved in contact), surgeon language references (when the surgeon refers to a slot by name).

**Outputs:** for each slot, an embedding vector `e_slot ∈ R^d` (initial `d ≈ 512`) plus per-dimension uncertainty.

**Architecture:** A multi-modal encoder trained with contrastive objectives (CLIP-style) across pairs of (visual feature, geometric feature, kinematic trace, force trace, surgeon language). The same encoder produces embeddings for slots, for **case-log / curation similarity queries** (ops and training pipeline — not substrate conditioning at inference), and for skill-required-affordance matching, ensuring all live in the same space.

**Derived heads (small networks on top of the embedding):**
- *Type classifier head:* outputs a confidence distribution over the open type hierarchy. Used for surgeon communication, debugging, and prior selection. Not used by the sync command gate for hard constraint types (surgeon tags / contract hard slots are authoritative; §5.10).
- *Affordance space projection:* projects the slot embedding into a learned affordance subspace `R^d_aff`. This is the *primary* affordance representation — a continuous vector capturing functional properties. Trained jointly with the rest of the embedding through contrastive objectives that group functionally-similar slots and through supervised signals from labeled affordance examples.
- *Named affordance readout heads:* one classifier per *labeled region* of the affordance subspace (`graspable_along_axis`, `has_cutting_edge`, `is_rigid`, `absorbs_fluid`, `is_vessel_like`, ...). Each outputs a confidence by measuring proximity of the slot's affordance vector to a learned anchor for that region. New named affordances can be added by labeling a few examples and training a new readout head — the affordance subspace itself doesn't change. Behavior contracts can also reference *unnamed neighborhoods* in affordance space directly (e.g., "any slot whose affordance vector is within 0.3 of canonical_aspirator_target") for cases where no named predicate suffices.
- *Per-slot dynamics adapter:* a small parameter set (or hypernetwork-conditioned weights) initialized from the embedding's nearest neighbors and refined online from observed interactions. Cerebellar analog.

**Latency budget:** ≤30 ms p99 for embedding computation; ≤5 ms p99 for derived head inference.

**Training:** see §6.4 and §8.2.

**Status:** **Component-level research investment.** A surgical multi-modal embedding model trained at scale on (vision, kinematics, force, language) pairs does not currently exist; building it is one of the project's primary research efforts.

### 5.3 Scene state

The shared blackboard. All other components read from and (where appropriate) write to it.

#### 5.3.1 Slot data model

The slot is structured in three layers: **primary** (the embedding-first substrate), **derived** (computed from primary, used as priors), and **explicit** (surgeon-authored, used for hard guarantees).

```python
class Slot:
    # ── Identity ───────────────────────────────────────────
    id: SlotId                         # stable across frames; reassigned on despawn
    last_updated: Timestamp
    history_ref: HistoryHandle         # to retrieve recent slot history

    # ── Primary representation (embedding-first) ───────────
    embedding: Vector                  # learned multi-modal embedding, R^d
    embedding_uncertainty: Vector      # per-dimension uncertainty
    pose: SE3                          # rigid pose (or anchor pose for deformable types)
    pose_uncertainty: SE3Cov           # 6x6 covariance in se(3)
    geometry: Geometry                 # mesh, SDF, parametric, or point cloud

    # ── Derived from embedding + observation ───────────────
    type_distribution: list[(TypePath, float)]   # open hierarchy + confidences
    affordance_vector: Vector                    # slot's position in affordance subspace (R^d_aff)
    affordance_uncertainty: Vector               # per-dimension uncertainty
    named_affordance_readouts: list[(Affordance, float)]  # readout head outputs (named regions)
    dynamics_adapter: SlotDynamicsAdapter        # online-adapted residual params
    derived_state: dict                          # type-conditional best-guess state

    # ── Explicit (surgeon-authored, authoritative) ─────────
    surgeon_tags: list[SurgeonTag]               # constraint-type + natural-language content
    surgeon_label: Optional[str]                 # human-readable name surgeon assigned
```

**SurgeonTag schema** — the two-layer tag from §3.1:

```python
class SurgeonTag:
    constraint_type: ConstraintType      # closed enum: NO_GO_REGION, FORCE_LIMIT,
                                         #   PROXIMITY_ALERT, DO_NOT_ACTION, ...
    constraint_params: dict              # type-specific deterministic params
                                         #   (e.g., NO_GO_REGION needs margin_mm)
    natural_language_content: str        # surgeon's free-form description
    anchored_to: SlotId                  # primary slot this tag applies to
    surgeon_confirmed: bool              # surgeon has visually confirmed the anchor
    created_at: Timestamp
    created_by: SurgeonId
```

Only `constraint_type`, `constraint_params`, and `anchored_to` are consumed by the safety evaluator’s sync gate (with SafetySurface; §5.10). `natural_language_content` is preserved for surgeon communication, case-log retrieval (surgeon UI, audit, training-data curation), and audit. `surgeon_confirmed` must be `True` before the tag becomes hard-constraint-active; until then the constraint is provisional and triggers an escalation if violated.

**Layer roles:**
- *Primary* is the source of truth. Perception writes here. The embedding captures everything the system has learned about this slot's perceptual and functional identity.
- *Derived* is recomputed from primary + observation. Type confidences, affordance vectors, and named affordance readouts drive substrate conditioning and provide priors to the world model. None of this is used for hard guarantees.
- *Explicit* is set by the surgeon (or by pre-op planning). Hard safety constraints reference the deterministic part of explicit tags (constraint type + params + slot anchor). The sync command gate does not use the derived layer or free-form natural-language content as a substitute for those deterministic fields.

**Type hierarchy.** `TypePath` is an open hierarchical path through an extensible ontology (e.g., `"Tool/Cutting/Scissors/Mayo"`). The ontology can grow at runtime: a slot whose embedding doesn't match any leaf is assigned a fresh `Unknown_*` leaf, which surgeon registration or accumulated observation can later refine. Multiple paths can have non-zero confidence simultaneously.

**Affordance representation (v0.4).** Affordances live in a continuous learned subspace `R^d_aff` of the joint embedding. Each slot has an `affordance_vector` (its position in that space) plus per-dimension uncertainty. Named affordances like `graspable_along_axis` are *labeled regions* exposed as readout heads — they yield a confidence in [0, 1] but are not the primary representation. This means:
- Two slots that are functionally similar (e.g., a known forceps and a novel forceps-like tool) have nearby affordance vectors, even before any named readout fires confidently.
- New named affordances can be added without retraining the affordance subspace — just label a few examples and train a new readout head.
- Behavior contracts can reference either named affordances (`graspable_along_axis ≥ 0.7`) or affordance-space neighborhoods (`distance(affordance_vector, canonical_aspirator_target) ≤ 0.3`). The latter handles cases where no named predicate captures what the surgeon needs.

**Dynamics adapter.** A small per-slot parameter set (or hypernetwork-conditioned weights) that adapts the world model's predictions for *this specific slot*. Initialized from the embedding's nearest neighbors in the training set; refined online from observed interactions.

**Surgeon tags (v0.4).** Two-layer (see `SurgeonTag` schema above). The *constraint type* is from a closed, formally specified enum the sync command gate enforces (with SafetySurface; §5.10). The *natural-language content* is the surgeon's free-form description, used for surgeon communication and case-log retrieval (surgeon UI, audit, training-data curation) but not in place of the deterministic constraint fields at the gate. The two layers are bound together in a single tag, anchored to a specific slot, with surgeon visual confirmation gating the tag's activation as a hard constraint.

Examples:
- `SurgeonTag(constraint_type=NO_GO_REGION, constraint_params={"margin_mm": 2.0}, natural_language_content="the proximal segment of vessel_018, the bigger branch", anchored_to=vessel_018)`
- `SurgeonTag(constraint_type=DO_NOT_ACTION, constraint_params={"action_class": "aspirate"}, natural_language_content="fragile-looking tissue near the anterior margin", anchored_to=tissue_034)`
- `SurgeonTag(constraint_type=FORCE_LIMIT, constraint_params={"max_force_N": 0.15}, natural_language_content="the granulation tissue", anchored_to=tissue_022)`

#### 5.3.2 Scene state (`SceneGraph`)

Slot-level blackboard inside the **entity knowledge store** (§5.2). The `SceneGraph` type name is historical; it denotes embedding-first **slots**, contact topology, and events — not the pre–v0.7 snapshot-only structure.

```python
class SceneGraph:
    slots: dict[SlotId, Slot]
    contact_graph: list[ContactEdge]      # which slot pairs are touching
    events: list[SceneEvent]              # spawn, despawn, contact, surgeon override, ...
    surgeon_state: SurgeonState           # in-control / out / observing / requesting
    timestamp: Timestamp
```

#### 5.3.3 Per-type derived state schemas

These schemas describe the structured `derived_state` populated when the slot's most-confident type path implies them. They are computed from the embedding plus observed history; they are not the slot's source of truth. A slot may have multiple plausible type paths and therefore multiple partially-populated derived-state schemas, each with its own confidence.

| Type path | Derived state fields |
|---|---|
| `Tool/*` | `tool_kind` (string from type path), `gripper_state`, `tip_engaged_with: SlotId?`, `estimated_force` |
| `Tissue/Intact/*` | `region_label` (anatomical), `estimated_material_params`, `deformation_field_handle`, `attached_to: list[SlotId]` |
| `Tissue/RemovedVoid/*` | `volume`, `boundary_mesh`, `created_by_skill`, `created_at` |
| `Cottonoid/*` | `saturation`, `pose_in_field` |
| `Fluid/*` | `density_field_handle`, `source_rate`, `total_volume` |

When a slot's type confidence is low across all paths, derived state is sparsely populated and downstream consumers (planner, world model) rely more heavily on the embedding directly.

#### 5.3.4 Invariants

- Every slot has a stable ID across frames until its despawn event.
- Every spawn/despawn event is observable and logged.
- Contact graph is consistent with slot poses (no contact edges between slots whose geometries don't intersect).
- Surgeon override always writes to scene state synchronously; all loops observe it within their next cycle.

### 5.4 World model

Embedding-conditioned forward dynamics, exposed in two interfaces. A single unified dynamics model handles all slots; per-slot behavior is conditioned on the slot's embedding and adapted by its dynamics adapter. Per-type defaults are the high-density regions of the embedding space, not separate code paths.

#### 5.4.1 Embedding-conditioned dynamics

The unified model has the form:

```
predict(slot_state, slot_embedding, slot_dynamics_adapter, action, scene_context, dt)
    → (next_slot_state, next_slot_state_uncertainty)
```

Internally it composes:
- A **physics prior** (analytic + simulator-backed) — rigid-body kinematics for tool-like slots, hybrid SOFA hyperelastic for tissue-like slots, scalar-field source/sink for fluid-like slots, etc. The prior used is selected by the embedding's region in the space (effectively a soft mixture-of-physics).
- A **learned residual** conditioned on the embedding and the dynamics adapter, capturing instance-specific behavior the prior misses (compliance, friability, saturation rate, individual anatomy).
- A **contact / interaction term** that takes pairs of slot embeddings and predicts contact forces and resulting state changes.

**Multimodal fusion for deformable dynamics.** Learned “elasticity,” compliance, and interaction coupling are inferred by **relating** SE(3)-structured action geometry to stereo-evolved surface state, applied wrenches, and temporal history — not by SE(3) equivariance alone. The result is a **predictive** model (residual over the physics prior) with calibrated uncertainty on its readouts; it is **not** a formal guarantee that the patient matches a particular material law. Hard safety remains surgeon-tagged constraints and the safety evaluator (§3.5, §5.10).

**Scene-integrated dynamics.** Conditioning on slot embeddings **and** **scene state** (slot poses, contact topology, events from §5.3) lets a single dynamics pass couple many slots. The joint embedding objective is intended so that **geometrically or anatomically neighboring regions** often lie in **nearby regions of embedding space**; then interaction history on one tissue slot acts as a **prior** for dynamics on adjacent slots, and **bleed / fluid risk** can propagate through coupled predictions and interaction terms—informing planner rollouts and risk readouts. This is an **empirical** property to measure (embedding neighborhood vs. spatial adjacency, transfer after local contact), not a theorem guaranteed by the architecture alone.

**Why embedding-conditioned (vs. per-type modules).** Per-type modules:
- Force a closed type system that doesn't generalize across similar types or to novel ones.
- Require N×M pairwise interaction modules for N tool types and M tissue types.
- Cannot transfer learning across instances of nominally-the-same type that actually differ.
- Fragment the implementation into N codebases that drift independently.

Embedding-conditioned is one model that:
- Generalizes by similarity in embedding space.
- Handles N×M interactions implicitly through the joint embedding of contact pairs.
- Transfers per-instance learning via the dynamics adapter.
- Lives in one codebase, validated end-to-end.

Per-type **defaults** still exist — they're the values predicted for prototypical embeddings of each type. They serve as initialization for the dynamics adapter when no prior observation exists for a specific slot.

#### 5.4.2 Dynamics interface

```python
class WorldModel:
    def predict(
        self,
        slot: Slot,
        action: Action,
        scene: SceneGraph,
        dt: Duration,
    ) -> tuple[SlotState, SlotStateUncertainty]: ...

    def predict_interaction(
        self,
        slot_a: Slot,
        slot_b: Slot,
        contact: ContactState,
        action: Action,
        dt: Duration,
    ) -> tuple[ContactState, list[SlotStateUpdate], InteractionUncertainty]: ...

    def rollout(
        self,
        scene: SceneGraph,
        action_sequence: list[Action],
        horizon: int,
    ) -> list[tuple[SceneGraph, SceneGraphUncertainty]]: ...

    def update_adapter(
        self,
        slot: Slot,
        observation: Observation,
    ) -> SlotDynamicsAdapter: ...   # online per-slot adaptation
```

#### 5.4.3 Two output channels

- **Latent rollout** for the planner: compact `z_t ∈ R^d` representation with fast multi-step predictions. Used inside MPC/MCTS rollouts.
- **Interpretable readouts** for the safety evaluator (SafetySurface) and planner: predicted poses, predicted forces, anatomy SDFs, no-go region distances — all with calibrated uncertainty.

#### 5.4.4 Performance contract

| Property | Requirement |
|---|---|
| Tool dynamics step latency | ≤1 ms |
| Tissue dynamics step latency | ≤20 ms (calibrated for fast loop horizon) |
| Latent rollout, 1 s horizon | ≤50 ms |
| Calibration target (interpretable readouts) | 90% conformal coverage at 95% nominal |

#### 5.4.5 Status

The unified embedding-conditioned dynamics model is **the primary component-level research bet.** It subsumes what would have been per-type research efforts:
- Embedding-conditioned hybrid (sim physics + learned residual) for soft tissue, with topology change handled via slot spawn/despawn.
- Tool dynamics emerging as the high-confidence-rigid region of the embedding space.
- Interaction term (pairwise embedding contact) is a research-active area in robot learning.

Component-level pieces that are off-the-shelf or engineering: physics prior (SOFA + standard rigid-body), simulator wrapping, online adapter machinery, latent rollout for planning (Dreamer/TD-MPC2 lineage).

### 5.5 Slow planner

#### 5.5.1 Purpose

Produce the conditioning bundle that drives the policy substrate, given the current scene, world model, surgeon goal, and risk state. The planner does not select a skill from a registry — it composes a language directive, sets modulation parameters, specifies the contract under which substrate behavior will be evaluated, and configures escalation triggers. Triggered on contract completion (success/failure), at periodic intervals, on escalation events, or on surgeon directive changes.

#### 5.5.2 Components

- **Recurrent reasoning module.** Loop transformer / Universal Transformer / HRN-class. Iterates K times over slot states with adaptive halting. K is bounded; halting is learned. Used for deliberation about goal decomposition, contract specification, modulation choice, and escalation.
- **Goal-to-directive translator.** Converts surgeon goals or higher-level plans into a language directive plus an associated contract. Example: surgeon goal "remove tumor at left margin" → directive "aspirate intact tumor tissue at slot tissue_007 with depth ≤ 2 mm" + contract with hard slots (force_envelope: max 0.3 N; abort_conditions: any slot with `SurgeonTag(constraint_type=NO_GO_REGION)` within 2 mm of tool tip; forbidden_constraint_types: any `DO_NOT_ACTION` of class "aspirate" on the target) and soft slots (success_criteria: "target tumor volume removed without unintended damage to adjacent parenchyma"; quality_measures: tumor margin cleanliness score).
- **Modulation controller.** Sets continuous modulation parameters (caution_level, time_pressure, attention_targets) based on risk, context, and surgeon state. These bias substrate output without re-issuing directives.
- **Search over candidate directives (when needed).** MCTS / MPPI / CEM over candidate language directives, evaluated by rolling out the substrate against the world model. Used when goal decomposition is ambiguous. Note: this is search over *directive sequences*, not over *skill calls* — there is no skill registry to enumerate.
- **Voluntary attention dispatcher.** "Look here" signal biasing the action-stream encoder's compute toward specific slots.
- **Escalation logic.** Conditions under which the planner stops autonomous operation and hands control to the surgeon.

#### 5.5.3 Interface

```python
class SlowPlanner:
    def step(
        self,
        scene: SceneGraph,
        world_model: WorldModel,
        risk: RiskState,
        goal: SurgeonGoal,
        substrate_state: SubstrateRuntimeState,   # for closed-loop reasoning
    ) -> PlannerOutput: ...

class PlannerOutput:
    # Primary output: conditioning bundle for the substrate
    directive: LanguageDirective       # natural-language description of intent
    contract: BehaviorContract         # mixed hard/soft slots; see below
    modulation: PlannerModulation      # continuous bias signals
    attention_targets: list[SlotId]
    escalation_triggers: list[Trigger] # conditions that raise an escalation event
    escalation: Optional[EscalationReason]   # immediate escalation, if any
```

**Directives are language, drawn from a continuous distribution.** A `LanguageDirective` is natural-language text describing intended behavior plus structured slot anchors (e.g., "aspirate intact tumor tissue at slot tissue_007 with depth ≤ 2 mm"). There is no enumerated directive vocabulary. The substrate is trained on a continuous distribution of natural-language directives across phases; new directives are tested for whether they fall inside the substrate's training distribution (via the OOD detector in §5.6.4) before autonomous execution. Phases (§10) measure this distribution's coverage and reliability, not the count of supported "skills."

**Behavior contract schema (v0.4) — mixed hard/soft.** The contract is the v0.3 successor to per-skill pre/post-conditions, with v0.4 extending it to express both formally verifiable safety properties and free-form success criteria the substrate can reason about:

```python
class BehaviorContract:
    # ── HARD slots (deterministic, formally verifiable, safety-critical) ──
    # Consumed by the sync command gate / SafetySurface for envelope
    # parameterization and by the substrate runtime for hard-constraint checks.
    # Verifiable.
    safety_invariants: list[FormalPredicate]      # over interpretable scene state
    abort_conditions: list[FormalPredicate]
    force_envelope: ForceEnvelope                  # max force, max velocity, contact rules
    forbidden_constraint_types: list[(ConstraintType, SlotId)]  # tags that must never be violated
    required_constraint_types: list[(ConstraintType, SlotId)]   # tags that must hold

    # ── SOFT slots (free-form, learned, success-quality) ──
    # Consumed by the substrate as conditioning and by the runtime via learned
    # heads. NOT verifiable formally; evaluated probabilistically.
    success_criteria: str                          # natural-language "what counts as success"
    quality_measures: list[QualityMeasure]         # learned heads predicting quality scores
    surgeon_intent_summary: str                    # the surgeon's verbal goal, preserved
    affordance_requirements: list[AffordanceRequirement]
        # can reference named affordances (`graspable_along_axis ≥ 0.7`)
        # or affordance-space neighborhoods (`distance(av, anchor) ≤ 0.3`)

    # ── Metadata ──
    directive_ref: LanguageDirective               # the directive this contract was issued for
    created_at: Timestamp
    contract_version: str                          # for audit / replay
```

The hard slots feed the safety evaluator's envelope selection (SafetySurface + sync gate) and the runtime's deterministic monitoring (§5.10). The soft slots feed the substrate as part of its conditioning bundle and the runtime's learned-head monitoring (§5.6.3). The two are evaluated separately: a contract violation in a hard slot is unconditional; a contract drift in a soft slot raises a soft-status flag that the planner can consider.

This mixed schema captures what surgeons actually want — "remove the tumor mostly, with this hard force limit and these absolute no-gos" — without forcing soft success criteria into formal predicates that they don't naturally fit, and without letting fuzzy success criteria contaminate the safety surface.

#### 5.5.4 Performance contract

| Property | Requirement |
|---|---|
| Step latency p99 | ≤2 s |
| Reasoning module compute budget | bounded; configurable per decision |
| Required scene staleness | ≤500 ms |
| Modulation update rate (without full re-step) | ≥5 Hz |

The modulation channel updates faster than the full planner step. This lets the planner continuously bias substrate behavior (e.g., raise caution after a bleed event) without paying the full reasoning cost.

#### 5.5.5 Status

Goal-to-directive translation: engineering + small research (LLM-style decomposition tuned on surgical goal/decomposition pairs). Search over directives: off-the-shelf (MCTS/MPPI/CEM). Recurrent reasoning module: research investment.

### 5.6 Policy substrate runtime

#### 5.6.1 Purpose

Run the compositional policy substrate against the current context bundle, monitor substrate output against the active behavioral contract, and forward target trajectories or commanded actions to the fast controller. Replaces what would conventionally be a "skill executor." There is no skill state machine; the substrate is itself a stateful network that integrates context over time. The runtime's job is to feed the substrate, watch it, and stop it when the contract is violated or the surgeon directs.

#### 5.6.2 Runtime architecture

```python
class PolicySubstrateRuntime:
    substrate: PolicySubstrate          # the learned compositional policy network

    # `risk` / RiskState in signatures below is the modulation-facing snapshot
    # produced by async safety assessment (former standalone risk system; §5.8, §5.10).

    def receive_conditioning(
        self,
        bundle: PlannerOutput,           # see §5.5
    ) -> None:
        """Update the substrate's conditioning. Does not reset substrate state;
        re-conditioning is continuous."""
        ...

    def step(
        self,
        scene: SceneGraph,
        world_model: WorldModel,
        risk: RiskState,
        recent_events: list[SceneEvent],
        surgeon_directives: list[Directive],
        modulation: PlannerModulation,
    ) -> RuntimeStepResult:
        """One mid-loop tick. Composes the full context bundle, runs the
        substrate forward, monitors the active contract, returns commanded
        actions and runtime status."""
        ...

    def reparameterize(
        self,
        new_directive: Optional[LanguageDirective] = None,
        new_contract: Optional[BehaviorContract] = None,
        new_modulation: Optional[PlannerModulation] = None,
    ) -> None:
        """Allows continuous mid-execution updates without resetting the
        substrate's hidden state."""
        ...

class RuntimeStepResult:
    commanded_action: Action               # to fast controller, then sync command gate (§5.10)
    contract_status: ContractStatus        # SATISFIED / IN_PROGRESS / DRIFTING / VIOLATED
    substrate_confidence: float            # the substrate's own confidence
    out_of_distribution_score: float       # from embedding-distance to training set
    requested_status: RuntimeStatus
```

`RuntimeStatus` ∈ {`RUNNING`, `CONTRACT_SATISFIED`, `CONTRACT_VIOLATED`, `LOW_CONFIDENCE`, `OUT_OF_DISTRIBUTION`, `REQUEST_ESCALATION`}.

The substrate itself is described in §6.5. The runtime is the thin layer between the planner and the substrate that handles I/O, monitoring, and runtime escalation.

#### 5.6.3 Behavioral contract monitoring

Because there is no per-skill code with explicit pre/post-conditions, contract monitoring is the runtime's responsibility. With v0.4's mixed contract schema, monitoring runs at two levels:

**Hard-slot monitoring (deterministic, on every tick):**
1. Evaluate **safety invariants** as formal predicates over interpretable scene state. If any is false, raise contract violation immediately.
2. Evaluate **abort conditions**. If any holds, raise abort.
3. Check that all **required constraint-type tags** still hold for their anchored slots. Violation = contract violation.
4. Check that no **forbidden constraint-type tags** are entered. Violation = contract violation.
5. Verify the active **force envelope** has not been narrowed by scene changes (e.g., approaching a flagged region triggers a tighter envelope).

Hard-slot violations are unconditional and route to the safety evaluator / planner immediately.

**Soft-slot monitoring (probabilistic, evaluated via learned heads):**
1. **Quality measures** are evaluated by their learned heads each tick; outputs are scalar quality scores tracked over time.
2. **Success criteria** are evaluated by a learned text-conditioned classifier head that consumes scene state, recent action history, and the criteria text; outputs a probability that the criteria are being met.
3. **Affordance requirements** are evaluated against the slot's affordance vector (for neighborhood-style requirements) or against its named-affordance readouts (for predicate-style requirements), with confidence.
4. Trajectory of soft-slot scores is tracked. Stalling or reversing over a configurable window raises a **drifting** status (not a violation; it's a signal to the planner).

Soft-slot drifting is a signal, not an emergency. The planner decides what to do with it (re-condition, modulate, escalate). This is what lets the system handle "the substrate is technically still aspirating but isn't making progress" without false-alarming on minor variation.

Either kind of contract event flows back to the planner, which decides whether to re-condition (issue a new directive), modulate (apply caution), or escalate.

#### 5.6.4 Distribution and confidence monitoring

Because the substrate composes behavior, novel compositions can drift far from training distribution. The runtime monitors:

- **Substrate confidence.** The substrate exposes a per-step confidence signal (e.g., entropy of action distribution, ensemble disagreement, or a learned uncertainty head). Low confidence raises `LOW_CONFIDENCE` status.
- **Out-of-distribution score.** Embedding-space distance between the current scene + context and the substrate's training distribution. High distance raises `OUT_OF_DISTRIBUTION` status.
- **Action plausibility.** Commanded actions outside the substrate's typical action distribution for the conditioning bundle are flagged.

Either status can trigger escalation, conservative modulation, or hand-off depending on the planner's policy.

#### 5.6.5 Continuous re-conditioning

The runtime does *not* terminate execution to receive new conditioning. The planner can update directive, contract, or modulation at any time, and the substrate's conditioning is updated on the next step without losing internal state. Worked example:

1. Active directive: "aspirate tissue at slot tissue_007." Substrate is mid-aspiration.
2. Async safety assessment detects a bleed event from a different slot (vessel_018) and updates SafetySurface / risk-modulation inputs. System attention updates: `caution_level: 0.3 → 0.8`, `attention_targets += [vessel_018]`.
3. Planner does *not* issue a new directive. It just calls `reparameterize(new_modulation=...)`.
4. On the next runtime step, the substrate's conditioning bundle includes the new modulation. Its output composes the still-active aspiration directive with the heightened caution + attention. The substrate may slow down, widen safety margins, and bias trajectory away from vessel_018, all without the runtime restarting any state machine.
5. If the surgeon then says "stop and address the bleed," the planner issues `reparameterize(new_directive="stop aspirating; cauterize vessel_018", new_contract=...)`. Substrate transitions to the new directive smoothly, leveraging its hidden state (it remembers what was happening) and the language-level transition.

This is the architectural payoff of substrate-as-policy: continuous, composable, attention-modulated execution without restart-and-replan boundaries.

#### 5.6.6 Performance contract

| Property | Requirement |
|---|---|
| Step latency p99 | ≤50 ms (substrate forward pass + monitoring) |
| Re-conditioning latency | ≤20 ms (modulation channel) |
| Contract evaluation completeness | 100% (every safety invariant evaluated every tick) |
| OOD detection coverage | configured per directive class; coverage measured on held-out evaluation set |

#### 5.6.7 Status

The runtime layer itself is engineering. The substrate is **the largest research investment in the project** (see §6.5). Contract monitoring and OOD detection are engineering + small research (conformal prediction over substrate uncertainty).

### 5.7 Fast controller

#### 5.7.1 Purpose

Convert the substrate runtime's output (target trajectories, force/velocity profiles, contact intents) into low-level robot commands, with closed-loop visual servoing and force-aware contact control. The substrate produces target trajectories at the runtime cadence (~5–20 Hz); the fast controller refines and executes them at ~100–500 Hz with tight sensor feedback.

This separation matters because:
- The substrate is too expensive to run at 200 Hz (it's a large network).
- Real-time servoing requires low latency that only a tight classical controller can provide.
- The substrate's output is a *plan over a short horizon*; the fast controller's job is to *track the plan robustly* given immediate sensor feedback and disturbances.

(An alternative design would have the substrate produce commanded actions directly at fast-loop rate. This is rejected for v0.3 because it ties substrate latency to control loop latency and makes safety verification harder. It can be reconsidered later if substrate inference improves enough.)

#### 5.7.2 Architecture

- **MPC** over the fast forward model. Short horizon (10–50 ms). Quadratic or cone-constrained programs solvable in real time.
- **Visual servoing** correction loop using the action-stream perception output. Drives end-effector pose toward the substrate's target trajectory.
- **Force-aware control** (impedance / admittance) for contact-sensitive operations, parameterized by the active contract's force envelopes.

#### 5.7.3 Interface

```python
class FastController:
    def step(
        self,
        substrate_command: SubstrateCommand,   # target trajectory or waypoint + force profile
        active_contract: BehaviorContract,     # for envelope parameterization
        scene: SceneGraph,
        world_model: WorldModel,
    ) -> RobotCommand: ...
```

#### 5.7.4 Performance contract

| Property | Requirement |
|---|---|
| Step rate | ≥100 Hz, target 200 Hz |
| Step latency p99 | ≤5 ms |

#### 5.7.5 Status

Off-the-shelf. MPC formulations and visual servoing are mature.

### 5.8 Risk system (absorbed; see §5.10)

#### 5.8.1 Purpose

**This section describes the former risk system's absorption into the safety evaluator's async assessment phase (§5.10).** A standalone risk-system component no longer exists in v0.8. Its capabilities — conformal prediction, per-entity risk scoring, escalation triggers, direct override — are now the foundation of the async safety assessment phase.

For detailed design of these capabilities' new home, see §5.10 (Safety evaluator).

#### 5.8.2 What was absorbed

- **Conformal prediction wrappers:** Now wrap entity-state-derived constraints in the async assessment phase. Calibrate uncertainty around deformation-compensated no-go geometry, context-aware force envelopes, and tissue-state risk scores.
- **Per-entity risk scoring:** Now part of EntityConstraints in the SafetySurface (§5.10.2). Combines learned entity state, interaction digest history, and predicted tissue state to compute cumulative risk per entity.
- **Escalation triggers:** Now handled by the safety evaluator's direct override path (§5.10.2). If async assessment detects critical threshold crossing, immediately update safety surface or veto. Target ≤50 ms latency.
- **Direct safety override:** Now the async assessment's escalation path. Updates the safety surface or triggers a blanket veto without waiting for the sync cycle.


| Property | Requirement |
|---|---|
| Update rate | ≥10 Hz |
| Override path latency | ≤50 ms (risk event → safety stop) |
| Calibration coverage | 90% at 95% nominal under in-distribution; degrade gracefully out-of-distribution with explicit warning |

#### 5.8.4 Status

Conformal prediction: off-the-shelf for static prediction; real-time/streaming variant is a small research investment.

### 5.9 System attention coordination

#### 5.9.1 Purpose

Coordinate four independent modulation signals (planner modulation, voluntary attention, **async-assessment** risk override, adaptive halting) into a single unified per-tick modulation state consumed by all control and perception components. *Risk* signals are produced by the **async safety assessment** phase of the safety evaluator (§5.8, §5.10), not by a separate risk-system component. This eliminates signal inconsistency and provides a single inspection surface.

#### 5.9.2 Components

```python
class SystemAttentionState:
    """Unified modulation state produced once per tick."""
    
    # Aggregated caution signals
    caution_level: float                    # [0.0, 1.0]; max(planner_caution, risk_caution)
    
    # Merged attention targets
    attention_targets: set[SlotId]          # union of planner targets + risk-flagged slots
    
    # Time pressure and deadline signals
    time_pressure: float                    # [0.0, 1.0]; set by planner, reduced if risk escalation imminent
    
    # Reasoning halting signal
    request_adaptive_halt: bool              # True if reasoning or confidence monitoring signals halt request
    
    # Escalation event (if triggered)
    escalation: Optional[EscalationEvent]    # aggregates risk escalation, contract violation, OOD detection
    escalation_source: Optional[str]         # "async_safety" | "contract_violation" | "ood_detection" | None
    
    # Metadata for debugging
    tick_id: int                             # for audit trail
    timestamp: float
    
    @staticmethod
    def aggregate(
        planner_modulation: PlannerModulation,
        risk_state: RiskState,
        contract_status: ContractStatus,
        substrate_confidence: float,
        ood_score: float,
        adaptive_halt_flags: dict[str, bool],
    ) -> "SystemAttentionState":
        """Deterministically aggregate modulation sources into unified state."""
        ...
```

#### 5.9.3 Aggregation logic (deterministic)

**Caution level:** `max(planner_modulation.caution_level, risk_state.caution_level)`. Fail-closed: if either source raises caution, the higher value is used. This ensures risk overrides cannot be undermined by planner optimism.

**Attention targets:** Union of `planner_modulation.attention_targets` and `risk_state.flagged_slots`. If both planner and async assessment flag a slot, it remains in the merged set.

**Time pressure:** Set by planner, but modulated down (clamped toward 0.0) if `risk_state.escalation_imminent == True`. This implements "slow down if risk is near threshold" without explicit planner intervention.

**Halting priority:** Set to `True` if any of the following hold:
- `reasoning_module.adaptive_halt_flag == True` (recurrent module decides to re-deliberate).
- `substrate_confidence < confidence_threshold` (substrate is uncertain).
- `ood_score > ood_threshold` (substrate is out of distribution).
- `contract_status == DRIFTING` and `drift_duration > drift_window` (soft-slot progress stalls).

Halting priority is checked by the slow planner; if True, the planner may trigger adaptive halting of the policy substrate runtime and re-step the reasoning module.

**Escalation aggregation:** If any of the following is true, populate `escalation` and route to the planner immediately:
- `risk_state.escalation != None` → escalation_source = `"async_safety"` (`RiskState` is published by async assessment; §5.10).
- `contract_status == VIOLATED` (hard-slot violation) → escalation_source = `"contract_violation"`.
- `ood_score > hard_ood_threshold` (OOD detection gates autonomy) → escalation_source = `"ood_detection"`.

#### 5.9.4 Consumer interface

All components that previously read modulation signals independently now consume `SystemAttentionState` from a module-level reference updated once per tick:

```python
class SystemAttention:
    current_state: SystemAttentionState
    
    def update(
        self,
        planner_modulation: PlannerModulation,
        risk_state: RiskState,
        contract_status: ContractStatus,
        substrate_confidence: float,
        ood_score: float,
        adaptive_halt_flags: dict[str, bool],
    ) -> None:
        """Called once per mid-loop tick by the slow planner after all monitoring is complete."""
        self.current_state = SystemAttentionState.aggregate(...)
    
    def get_state(self) -> SystemAttentionState:
        """Called by substrate runtime, fast controller, async safety assessment, perception."""
        return self.current_state
```

**Usage by substrate runtime (§5.6):** Consumes `system_attention.get_state().attention_targets` and `caution_level` for continuous re-conditioning.

**Usage by fast controller (§5.7):** Consumes `caution_level` and `time_pressure` to modulate servo gains and trajectory tracking aggressiveness.

**Usage by action-stream encoder (§5.2.2):** Consumes `attention_targets` to bias compute toward prioritized slots.

**Usage by async safety assessment (§5.8, §5.10):** Publishes `RiskState` into aggregation; may read back `current_state.caution_level` to verify planner-side honoring of caution (implementation choice).

**Usage by planner (§5.5):** Reads `request_adaptive_halt` and `escalation` to trigger re-deliberation or escalation; produces `planner_modulation` and `attention_targets` that feed into the next aggregation.

#### 5.9.5 Data flow diagram

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────┐
│ Slow Planner     │    │ Async safety     │    │ Substrate    │    │ Adaptive │
│ (PlannerModulation)   │ (RiskState)      │    │ Runtime      │    │ Halt     │
└────────┬─────────┘    └────────┬─────────┘    │ (Contract)   │    │ Flags    │
         │                      │            └────┬─────────┘    └────┬─────┘
         │                      │                 │                    │
         └──────────────────────┼─────────────────┼────────────────────┘
                                │
                                v
                  ┌──────────────────────┐
                  │ SystemAttention      │
                  │ .aggregate()         │
                  │ (deterministic)      │
                  └──────────┬───────────┘
                             │
                             v
                  ┌──────────────────────┐
                  │ SystemAttentionState │
                  │ (unified per-tick)   │
                  └──────────┬───────────┘
                             │
         ┌───────────────────┼────────────────────┬─────────────────┐
         │                   │                    │                 │
         v                   v                    v                 v
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │Substrate     │   │Fast          │   │Action-Stream │   │Async safety   │
   │Runtime       │   │Controller    │   │Encoder       │   │assessment     │
   │ (modulation) │   │ (gains)      │   │ (compute)    │   │(RiskState out)│
   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

#### 5.9.6 Performance contract

| Property | Requirement |
|---|---|
| Aggregation latency | ≤1 ms (deterministic, no learning) |
| Update frequency | Once per mid-loop tick (~5–20 Hz) |
| Consistency | All consumers see the same SystemAttentionState during a tick |
| Inspection surface | Single function call to get unified state; audit trail includes aggregation reason |

#### 5.9.7 Status

Deterministic aggregation logic: straightforward engineering. Integration point: planned for v0.6 implementation. No research content; the value is in coordination clarity and debugging surface.

### 5.10 Safety evaluator (two-phase: async assessment + sync command gate)

#### 5.10.1 Purpose

Final arbiter of every commanded action. Reads entity state to assess real anatomy. Publishes interpretable safety surface. Checks every command at controller rate. Fail-closed.

The safety evaluator honestly acknowledges its dependence on learned perception and world model while remaining dedicated to safety: entity embeddings and deformation predictions inform its assessment, but the assessments are trained separately from the policy, interpretable (a readable safety surface is published), conservative (uncertain → veto), and operator-authoritative (surgeon override is unconditional).

#### 5.10.2 Two-phase architecture

**Layer 1: Physical invariants (genuinely formal, no learned inputs)**

Small, trivially verifiable, model-independent:
- Joint limits, workspace bounds
- Absolute hardware force/velocity limits
- Kinematic singularity avoidance
- Communication watchdog (command timeout → stop)

Checked by Phase 2 (sync command gate) alongside the safety surface.

**Layer 2, Phase 1: Async safety assessment (~10-20 Hz)**

This is the "thinking" part. Absorbs former standalone risk-system capabilities (now baseline async assessment) and extends them with entity-informed reasoning:

Reads: entity embeddings, interaction digests, world model deformations, surgeon tags. 
Computes: deformation-compensated no-go geometry, context-aware force envelopes, per-entity cumulative risk scores, anomaly flags.
Publishes: SafetySurface (per-entity constraints, deformation-compensated SDFs, safety margins, confidence bounds).
Exposes interpretable signals to planner for reasoning.
Retains: direct override path (≤50 ms from detection to action).

**Layer 2, Phase 2: Sync command gate (100-500 Hz)**

This is the "checking" part. Fast and thin.

Reads: pre-computed safety surface + Layer 1 physical invariants.
Checks: each command against deformation-compensated no-go SDFs, force/velocity envelopes, risk thresholds.
Decides: pass / project / veto.
No heavy inference per tick — just geometric/envelope checks against pre-computed surfaces.

**Output:** Each command yields a **`SafetyDecision`** (logged as `safety_decision` in §8.1.2; see glossary **Sync command gate**). Fail-closed defaults, substrate-output-agnostic checks, and immediate surgeon override apply regardless of outcome shape.

```python
@dataclass
class SafetyDecision:
    """Outcome of the sync command gate for one command."""
    action: Literal["pass", "project", "veto"]
    modified_command: Optional[RobotCommand]  # if "project"
    veto_reason: Optional[str]  # if "veto"
```

#### 5.10.3 Constraint set and verification

**Layer 1: Physical invariants** (robot-level, not anatomy)
- Joint limits, workspace bounds
- Hardware force/velocity limits
- Singularity avoidance
- Communication watchdog

Formally verifiable. <500 LOC.

**Sync gate (Phase 2):** Deterministic logic.
- Geometric intersection tests against deformation-compensated no-go SDFs
- Envelope comparisons
- Anomaly flag logic
- Formally verified neural sub-checks (alpha-beta-CROWN) for geometric predicates only

<2000 LOC, fully unit-tested.

**Async assessment (Phase 1):** Calibrated learned evaluation.
- Deformation-compensated geometry: validated via world model calibration (phase testing).
- Per-entity risk scoring: calibrated via conformal prediction. Coverage: 90% at 95% nominal; degrade gracefully OOD.
- Safety head training: trained specifically for safety, separate evaluation suite, NOT optimized for policy convenience.
- System-level validation: "does safety surface veto rate match held-out test? does surface miss real hazards?"

#### 5.10.4 Verification surface

#### 5.10.5 Performance contract

Phase 1 (async assessment): ~10-20 Hz, ≤100 ms p99.
Phase 2 (sync gate): 100-500 Hz, ≤2 ms p99, 100% command coverage.
Direct override: event-driven, ≤50 ms.

#### 5.10.6 Status

Phase 2 (sync gate): Engineering. Phase 1 (async assessment): Research.

### 5.11 Robot interface

#### 5.11.1 Purpose

Bridge between the safety evaluator and the neuroArm hardware control layer.

#### 5.11.2 Modes

- **Teleop pass-through.** Surgeon commands flow through the **sync command gate** (safety evaluator) and reach the robot. Autonomous components are disabled or in observe-only mode.
- **Assistive.** Surgeon commands are augmented (virtual fixtures, force scaling, no-go warnings) but the surgeon still drives.
- **Autonomous narrow.** Single skill or skill sequence runs autonomously under surgeon supervision; abort-on-anything.
- **Autonomous procedural.** Multi-skill autonomous operation with surgeon as supervisor.

The mode is part of the scene state; the safety evaluator's active constraint set varies by mode.

---

## 6. Substrate / backbone

The shared neural substrate that perception, world model, and slot attention sit on.

### 6.1 Architecture

Hybrid SSM-Transformer following the Jamba/Hymba/Zamba pattern.

- **SSM layers (Mamba-2 lineage):** carry persistent state across the streaming surgical procedure. Linear-time scaling enables long-horizon dependencies (entire procedure-length context).
- **Attention layers:** cross-modal fusion (vision + kinematics + force + audio), long-range queries (refer back to a tissue region cottonoided ten minutes ago), slot-to-slot attention.
- **SE(3)-equivariant attention layers:** for tool slots and tool–tissue interactions. Burn geometric symmetry into the architecture.

### 6.2 Tokenization

| Modality | Tokenization |
|---|---|
| Stereo video | Patch tokens per frame, with stereo correspondence |
| Kinematics | One token per joint per timestep, downsampled |
| Force | One token per axis per timestep, downsampled |
| Audio | Mel-spectrogram patches |
| Slot state | One token per slot, refreshed each frame |

### 6.3 Status

Off-the-shelf. Multiple production-ready hybrid SSM-Transformer implementations exist. Choice of specific variant is a reversible decision; we will pick the best-supported open implementation at the time of build.

### 6.4 Joint embedding model

The substrate produces feature representations; the **joint embedding model** projects per-slot crops of those features (plus auxiliary modalities) into a shared embedding space that all consumers share.

#### 6.4.1 Architecture

A multi-modal encoder composed of:

- **Per-modality encoders.** Visual encoder (ViT-class on slot crops), geometric encoder (PointNet/Transformer over slot point clouds), kinematic encoder (small SSM/Transformer over recent kinematic windows), force encoder (small transformer over recent F/T traces), language encoder (off-the-shelf text encoder for surgeon utterances).
- **Projection heads.** Each modality projects to the shared `R^d` embedding space (initial `d ≈ 512`; revisable).
- **Fusion.** Per-slot embedding is the (uncertainty-weighted) mean of available modalities, plus a learned residual.

#### 6.4.2 Training objectives

Multi-modal contrastive (CLIP-style):

1. **Cross-modal alignment.** For a slot observed across modalities, embeddings of (vision, geometry, kinematics, force, language) should be mutually close. InfoNCE losses across modality pairs.
2. **Cross-instance similarity.** Slots labeled (in training data) as the same type should be closer than slots of different types. Slots that share affordances should be closer than slots that don't.
3. **Action-conditioned similarity.** Slots that respond similarly to similar actions (e.g., two cottonoids that absorb at similar rates) should be close in embedding space.
4. **Language alignment.** When the surgeon uses a name for a slot, the slot's embedding aligns with the language encoder's embedding of that name.

#### 6.4.3 Calibration and retraining

Embedding spaces drift when the model is retrained. This is a real concern because affordance predicates and dynamics adapters are conditioned on the embedding.

Protocol:

- **Embedding versioning.** Each embedding is tagged with the model version that produced it. Affordance predicates and dynamics adapters declare which embedding versions they're compatible with.
- **Compatibility-preserving fine-tuning.** When the embedding model is updated, fine-tune with a regularizer that penalizes large shifts in embeddings of canonical reference slots. Standard practice (anchor-based fine-tuning).
- **Recalibration runs.** After any retraining, re-run validation for affordance predicates and conformal calibration on a held-out reference set. If degraded, gate the rollout.
- **Rollback path.** Old model versions remain deployable.

#### 6.4.4 Status

**Component-level research investment.** A surgical multi-modal embedding model trained at scale on (vision, geometry, kinematics, force, language) does not currently exist. Closest analogs: surgical foundation models (SurgVLP, SurgMotion, etc.) which are vision+language only. Building the multi-modal joint embedding for this project is a primary research effort (estimated 12–24 month component-level program). Bootstrap path: start with a vision+language model + handcrafted projections from kinematics/force, replace with learned encoders as data accumulates.

### 6.5 Compositional policy substrate

The compositional policy substrate is the system's action source. It is the largest single research investment in the project. This section specifies what it is, how it's structured, how it's trained, how its output is shaped, and how it composes behavior across what would conventionally be considered separate skills.

#### 6.5.1 Architecture

The substrate is a unified neural policy network with the following structure:

- **Conditioning encoder.** Ingests the conditioning bundle from the planner (language directive, behavior contract, modulation parameters, attention targets). Produces a conditioning vector that biases the policy throughout execution.
- **Context encoder.** Ingests the runtime context bundle (scene state including all slot embeddings, world model rollouts, risk state, recent events, surgeon directives). Produces a context vector.
- **Recurrent policy core.** Hybrid SSM-Transformer (sharing the substrate from §6.1–6.3) with persistent hidden state across the procedure. Takes (conditioning, context, hidden state) and outputs the next action distribution.
- **Action head.** Produces structured output: target end-effector trajectory over a short horizon (typically 50–500 ms), force/velocity profile, gripper state, mode flags.
- **Confidence head.** Produces a per-step uncertainty estimate (entropy, ensemble disagreement, or learned uncertainty) used by the runtime for OOD/low-confidence detection.

**Tokenization for the substrate's input.** Slots are object-tokens with their embeddings, poses, and key derived state. Language directive is encoded by a language encoder. Modulation parameters are continuous-valued tokens. World-model predicted future state can be included as predicted-token sequences for short rollouts.

#### 6.5.2 Output shape

The substrate emits short-horizon target trajectories rather than instantaneous joint commands. Specifically:

- Target end-effector pose trajectory over the next ~100–500 ms in SE(3)
- Target force/velocity profile along that trajectory
- Mode flags: contact mode, gripper state, tool-tip engagement intent
- Termination flag: continue / contract-satisfied / abort

The fast controller (§5.7) consumes this and tracks it with closed-loop refinement at 100–500 Hz. The substrate's output cadence is ~5–20 Hz; planning happens at every output, but the fast controller continues servoing during the gap.

#### 6.5.3 Compositional behavior

The substrate composes behavior through three mechanisms (v0.5: episodic in-context examples removed):

1. **Language conditioning composition.** Multiple language directives or contract clauses can be combined in the conditioning encoder. The substrate learns to produce behavior consistent with all clauses simultaneously (e.g., "aspirate this" + "stay shallow" + "watch for bleeding" → behavior reflects all three). This is the same compositional generalization that LLMs exhibit.
2. **Modulation as continuous bias.** The modulation channel (caution_level, attention_targets, time_pressure, etc.) is a continuous input that can re-weight the substrate's output at any moment without changing the directive. This implements attention-driven motor modulation analogous to PFC-PMC modulation in the brain.
3. **Training-distribution coverage.** The substrate has been trained on a wide distribution of demonstrations covering many directive combinations and many situations. The patterns it composes from at inference time live in its weights, not in retrieved cases. Compositional generalization across directives emerges from the substrate's training; if a particular composition fails to generalize, the answer is more diverse training data, not inference-time retrieval.

These three mechanisms together let the substrate handle situations like "aspirate-while-bleed-from-different-vessel" without any pre-trained `AspirateWhileBleeding` skill and without retrieving past cases at inference: the directive is `Aspirate(slot_007)`, the contract includes the original bounds, the modulation reflects the bleed-elevated caution + new attention target, and the substrate's training distribution includes enough bleed-management examples for it to compose appropriately. The substrate's output composes all of this from its weights.

#### 6.5.4 Training

The substrate is trained with a mixture of objectives over a unified dataset:

- **Imitation from teleop.** Most data. Each frame: (sensor observation, scene snapshot, surgeon's commanded action, surgeon's verbal directive when available, contract reconstructed post-hoc from outcomes). Train the substrate to predict the surgeon's action conditioned on the rest.
- **Imitation from simulator.** Diverse simulated rollouts of skilled behavior (surgeons-in-loop in sim, scripted experts, DAgger). Used for coverage of states real teleop rarely reaches.
- **Reinforcement learning in simulation.** Reward shaping based on contract satisfaction. Used after imitation pretraining to refine and to discover behaviors imitation can't reach.
- **DAgger / human-in-the-loop correction.** Once the substrate is good enough to be assistive, surgeon corrections during real cases are recorded and used as additional training data. This is the long-term flywheel.
- **Contract-conditioned auxiliary tasks.** Auxiliary heads predict contract satisfaction from observations alone, providing additional learning signal and supporting the runtime's monitoring layer.

**The training data is the moat.** Every real-OR teleop session, every cadaver lab session, every phantom run, every simulator rollout is logged in the format the substrate consumes. The substrate's quality is approximately monotonic in dataset quality + diversity. This is why §8 (data, training, simulator) is so important.

#### 6.5.5 Behavioral evaluation (replaces per-skill testing)

Without a discrete skill library, evaluation is behavioral and contract-based:

- For each named directive type (e.g., `Aspirate`, `Cauterize`, `Suture`) define an evaluation suite of scenarios in the simulator.
- Run the substrate on each scenario, conditioned on the directive and a representative contract.
- Measure: contract post-condition satisfaction rate, safety invariant violation rate (caught by safety evaluator / sync gate, no invariant should ever be actually breached), substrate confidence calibration, time-to-completion, surgeon-rated quality on a held-out scenario subset.
- Aggregate into a per-directive performance profile. This is what replaces "is this skill working?"

Cross-directive composition is evaluated separately: scenarios that require the substrate to compose multiple directives' patterns (e.g., bleed during suture). These test the substrate's compositional generalization, which is the central architectural claim.

#### 6.5.6 Bootstrapping path

Honest about the cold-start problem. With v0.4's directive-distribution framing, phases are described as expansions of *distribution coverage*, not as additions of *directive types* to an enum:

- **Phase 0 (initialization).** Fine-tune an existing surgical or general VLA (Pi-zero / OpenVLA / surgical-VLP family) on collected teleop and simulator data. The substrate's directive distribution is narrow; OOD scores rise quickly outside it.
- **Phase 1 (peg transfer).** Distribution covers natural-language directives describing rigid-object manipulation: pick, transfer, place, move-to-pose. Substrate behavior on directives in this region is reliable; outside, OOD detector kicks in and autonomy is gated.
- **Phase 2–4.** Distribution expanded to cover suturing language, then retraction language, then resection language. Substrate is fine-tuned (not retrained from scratch) on the expanded data. Replay buffers preserve earlier coverage. Cross-directive composition starts to be tested.
- **Phase 5+.** Distribution covers full diverse surgical language. The substrate's compositional generalization is stress-tested on directives that fall *between* training clusters (e.g., bleed-during-suture-language) and on novel directive phrasings within the covered region.

The substrate's *capability frontier* is the set of directives where it behaves reliably under realistic context distributions, measured by the per-directive evaluation suites in §6.5.5 plus cross-directive composition tests. Behavior outside the frontier is unreliable — OOD detection (§5.6.4) gates autonomy at the frontier; surgeon supervision is required outside.

This framing also dictates the dataset shape: instead of curating per-skill demonstration sets, we curate broad, diverse, *language-annotated* demonstration sets where every clip has surgeon-spoken or surgeon-typed natural-language descriptions of intent, which become the substrate's training conditioning.

#### 6.5.7 Status

**The primary research investment of the entire project.** Estimated multi-year program. Critical risk: if compositional generalization doesn't emerge as expected, the substrate produces brittle behavior that doesn't transfer beyond its narrow training distribution, and the project effectively reverts to per-directive policies that are skills in everything but name. This is a real failure mode and the architecture's mitigations are: (a) the **safety evaluator** (sync gate + SafetySurface) catches unsafe commands regardless of substrate quality; (b) the contract-monitoring runtime escalates when behavior drifts; (c) the OOD detector keeps the substrate operating only within its frontier; (d) phase-by-phase scope expansion limits exposure to substrate failures.

This bet is the single biggest reason the spec's goals must be evaluated honestly against the substrate's maturity at each phase.

---

## 7. Memory architecture

Memory in this system is **distributed**, not a separate architectural layer. There are exactly two places memory lives, and neither requires its own component:

### 7.1 Semantic memory — substrate weights

The policy substrate's weights and the joint embedding model's weights. Slow, generalized, updated by training. This is where general knowledge of surgical procedures, anatomy, tool use, and learned skill patterns lives. Updated by:

- Periodic full fine-tunes on accumulated logged data (cadence: typically weekly to monthly).
- Fast online fine-tunes (LoRA-style adapters, hours to overnight) for surgeon-flagged teaching cases (§8.2).
- Per-phase substrate retraining when the directive distribution is expanded (§6.5.6).

There is no "rare case influence at inference time" mechanism. Rare or critical cases enter the weights through training; if a case must affect behavior immediately, the surgeon flags it and triggers a fast LoRA fine-tune.

### 7.2 Working memory — substrate recurrent state + slot-level scene state

Within-procedure context. Two sub-components, both already specified elsewhere in this document:

- **Substrate recurrent state.** The hybrid SSM-Transformer's hidden state (§6) carries information from the start of a procedure to its current moment. This is what lets the substrate "remember" actions taken minutes earlier. SSMs are particularly suited to this; this is one of their main advantages over pure attention.
- **Persistent slots (entity knowledge store).** The typed object slots in §5.3 — maintained inside the **entity knowledge store** (§5.2) — persist across frames and across occlusion, with explicit identities, embeddings, surgeon tags, and history. This is the system's structured short-term memory — the "where is everything, what's its state, what just happened" answer queryable at any time.

### 7.3 What's deliberately *not* here

Earlier drafts of this spec proposed an "episodic memory" architectural layer — a RAG database of past surgical cases with retrieval-into-substrate-context at inference time. That layer was removed in v0.5. The reasoning:

- Anything in the training set is already in the substrate's weights. Retrieving and re-presenting it is redundant.
- Patient-specific context flows in via the conditioning bundle (pre-op imaging, surgical plan, surgeon-tagged anatomy) at procedure start — that's data flow, not memory.
- Surgeon teaching is better served by fast online fine-tuning (§8.2) than by inference-time retrieval — better generalization, evaluable before deployment.
- Cross-procedure influence is the training pipeline's job, not the inference-time retrieval pipeline's job.

The case *log* still exists and is operationally important (surgeon UI, audit, training-data curation, failure analysis). It lives in §8.1 as part of the data pipeline, not as architectural memory. Surgeons can query it; auditors can trace through it; the training pipeline curates from it. None of that requires the substrate to retrieve cases at inference.

---

## 8. Data, training, simulator

### 8.1 Data pipeline

#### 8.1.1 Sources

- Real teleop sessions on neuroArm (primary moat).
- Cadaveric sessions (where available).
- Phantom sessions (developmental).
- Simulator rollouts.
- Public surgical video datasets (limited; mostly for foundation-model fine-tuning).

#### 8.1.2 Logging schema

Every session logs synchronously:

```python
class LoggedFrame:
    timestamp: Timestamp
    sensor_payload: SensorBundle      # all sensor streams
    scene_snapshot: SceneGraph
    commanded_action: Optional[RobotCommand]
    executed_action: Optional[RobotCommand]
    safety_decision: Optional[SafetyDecision]
    skill_state: Optional[SkillState]
    surgeon_input: Optional[TeleopInput]
    outcome_label: Optional[OutcomeLabel]   # post-hoc
```

Note: `skill_state` is retained as a logging field for backward compatibility / auditability of pre-substrate prototypes; for substrate-driven operation it is `None`. The substrate's runtime state (active directive, contract, modulation, runtime status) is logged in additional fields not shown above.

All frames written to immutable storage with strong schema versioning.

#### 8.1.3 Case log

The accumulated logged frames + per-case metadata constitute the **case log**. This is an operational artifact, not an architectural component:

- **Surgeon UI** queries the case log for review, comparison, and pre-op planning.
- **Audit** queries the case log to trace what scene state, conditioning, and substrate output occurred at each decision.
- **Training-data curation** pulls from the case log: cases get reviewed, outcome-labeled, and batched for the next substrate fine-tune. Surgeon-flagged teaching cases are batched separately for fast online fine-tunes (§8.2).
- **Failure analysis** queries the case log post-hoc using embedding similarity (or surgeon-tag content, or other queryable fields) to find cases similar to a given failure mode.

The case log is *not* queried by the substrate at inference time. There is no architectural retrieval-into-conditioning layer. Inference-time access to the log is restricted to the slow planner's reasoning module for surgeon-readable context where useful, but never as substrate input.

### 8.2 Training regimes

| Component | Training data | Method |
|---|---|---|
| Action stream encoder | Sim + real teleop | Self-supervised + supervised on tracking labels |
| Semantic stream encoder | Public surgical video + real teleop, fine-tune on labels | Foundation model fine-tune |
| **Joint embedding model (§6.4)** | Multi-modal pairs across all logged modalities | Multi-modal contrastive (CLIP-style) |
| Slot Attention | Real teleop, sim | Supervised on slot ground truth from sim, semi-supervised on real |
| Type / affordance heads | Labeled slot crops; affordance labels from sim and teleop | Supervised classification probes over the embedding |
| World model (unified, embedding-conditioned) | Sim + real teleop with logged scene + actions | Hybrid: sim physics prior + learned residual conditioned on slot embeddings |
| Per-slot dynamics adapter | Online interaction observations | Online few-shot adaptation; initialized from embedding-nearest-neighbor priors |
| **Compositional policy substrate (§6.5)** | All teleop + simulator rollouts + DAgger corrections, conditioned on language directives + contracts | Imitation pretraining → simulator RL with contract-based reward → DAgger fine-tuning. Bootstrap from existing VLA. |
| **Substrate online fine-tunes (LoRA-style)** | Surgeon-flagged teaching cases | Fast adapter training (hours-to-overnight cadence) gated by per-directive evaluation suite before deployment; merged into base substrate at next periodic full retrain |
| Substrate confidence head | Substrate's own predictions vs. observed outcomes | Self-supervised + held-out calibration |
| Slow planner — reasoning module | Teleop demonstrations of goal decomposition, paired with surgeon language | Imitation + supervised on decomposition labels |
| Slow planner — modulation policy | Teleop sessions with risk events, paired with surgeon's behavioral changes | Supervised on event-to-modulation mappings |
| Slow planner — value head (for directive search) | Sim rollouts + outcome labels | Supervised |
| Risk / conformal | Held-out calibration set across distribution-shifted scenarios | Calibration set construction; recalibrate after each substrate or embedding retrain |
| Out-of-distribution detector | Embedding distances on training-distribution data | Held-out OOD reference set; conformal thresholds |

### 8.3 Simulator

#### 8.3.1 Choice

SOFA-Framework (open source, mature soft-body, used in surgical sim research) wrapped to expose the project's `Environment` interface.

#### 8.3.2 Environment interface

```python
class Environment(Protocol):
    def reset(self, config: EnvConfig) -> SceneGraph: ...
    def step(self, action: RobotCommand) -> StepResult: ...
    def get_sensors(self) -> SensorBundle: ...
    def get_scene(self) -> SceneGraph: ...

class StepResult:
    next_scene: SceneGraph
    sensor_observation: SensorBundle
    info: dict
```

The same `Environment` is implemented by `SimEnvironment` (SOFA-backed) and `RealEnvironment` (neuroArm-backed).

#### 8.3.3 Domain randomization

Per-patient anatomy variants generated from preop-MRI distributions. Material parameters randomized within physiologically plausible ranges. Lighting, instrument appearance, camera calibration all randomized.

---

## 9. Implementation status (per-component)

| Component | Status | Notes |
|---|---|---|
| Entity knowledge store (embedding-first slots; object-centric) | Engineering | Build from scratch; <3K LOC, critical interface |
| Hybrid SSM-Transformer substrate | Off-the-shelf | Pick a variant; fine-tune |
| Action stream encoder (equivariant) | Off-the-shelf + integration | Streaming adaptation modest research |
| Semantic stream encoder (foundation) | Off-the-shelf | SAM 2 / MedSAM2 / surgical VLA fine-tune |
| **Joint multi-modal embedding model (§6.4)** | **Primary research investment** | 12–24 month component program; bootstrap from existing surgical VLPs |
| **Compositional policy substrate (§6.5)** | **Primary research investment — largest in the project** | Multi-year program; bootstrap from existing VLA |
| Substrate runtime (contract monitoring, OOD, re-conditioning) | Engineering + small research | Runtime layer: engineering. OOD detection under distribution shift: research. |
| Slot Attention (variable-cardinality, streaming) | **Research investment** | Streaming + variable-N is research-active |
| Type/affordance classifier heads | Engineering on top of embedding | Linear/MLP probes once embedding is available |
| **Unified embedding-conditioned world model** | **Primary research investment** | Subsumes per-type dynamics; 18–36 months for the soft-tissue + interaction term |
| Per-slot dynamics adapter | Engineering + research | Online few-shot adaptation machinery |
| Latent world model + interpretable readouts | Engineering + research | Latent: research; readouts: engineering |
| Case log (surgeon UI / audit / curation) | Engineering | Vector DB + similarity search over logged data; not architectural memory |
| Online fine-tuning pipeline (LoRA-style) | Engineering | Adapter training, eval-gated rollout, merge protocol |
| Recurrent reasoning module (planner) | **Research investment** | Loop transformer / HRN, smaller bet |
| Goal-to-directive translator | Engineering + small research | LLM-style decomposition fine-tuned on surgical goal/decomposition pairs |
| Fast controller (MPC + visual servo) | Off-the-shelf | Mature techniques |
| Search-based directive search (MCTS/MPPI/CEM) | Off-the-shelf | AlphaGo-template integration; over directives, not skills |
| Conformal prediction (calibrated uncertainty) | Off-the-shelf + research | Static: off-the-shelf; streaming/distribution-shift: research |
| Safety evaluator — sync command gate (CBF + verified NN sub-checks, SafetySurface + Layer 1) | Engineering | Mature techniques; entity-informed surface from async assessment; substrate-output-agnostic |
| Safety evaluator — async assessment (conformal + per-entity risk + escalation) | Engineering + small research | Absorbs former risk system; §5.8, §5.10 |
| Simulator (SOFA wrapper) | Engineering | Standard work, weeks not months |
| System attention coordination (§5.9) | Engineering | Deterministic aggregation logic; single inspection surface for modulation |
| Data pipeline | Engineering | Standard work, must be done early; the moat |
| Robot interface (neuroArm) | Engineering | Existing teleop layer + safety boundary |

Component-level research investments, in priority order:

1. **Compositional policy substrate (§6.5)** — the action source. Single largest research effort in the project. All autonomous behavior emerges from this. Multi-year, multi-person.
2. **Joint multi-modal embedding model (§6.4)** — substrate-of-the-substrate. Conditions scene representation, world model, and substrate input. 12–24 month program.
3. **Unified embedding-conditioned world model** — provides the substrate's training environment and the planner's rollout engine. 18–36 months for soft-tissue + interaction.
4. **Streaming variable-cardinality Slot Attention** — the gateway between perception and the embedding-first **slot set** in the entity knowledge store.
5. **Adaptive recurrent reasoning module at decision points** — the planner's deliberation engine.

Smaller research-active areas:

6. **Real-time conformal prediction under distribution shift** — for the substrate's confidence head, OOD detector, and safety-bridge calibration.
7. **Compatibility-preserving fine-tuning** of substrate and embedding model (so retraining doesn't break downstream behavior).

Everything else is engineering of off-the-shelf components.

**Critical risk:** the substrate is the load-bearing research bet. If compositional generalization across directives doesn't emerge reliably from training, the system effectively degrades to per-directive policies that are skills in everything but name — without the verification benefits a discrete skill library would have given. Mitigations: output-level **safety evaluator** (catches unsafe commands regardless), contract monitoring (escalates on drift), OOD detection (operates only within frontier), phase-by-phase scope (limits exposure). These mitigations make the bet survivable but not free; substrate underperformance directly impacts what phases are achievable when.

---

## 10. Phasing

A separate roadmap document will detail timelines and milestones. The phasing is framed around **the substrate's capability frontier in directive distribution and anatomical/object embedding-space coverage**, and around a third axis: **deformable / interaction world-model reliability** (how well the unified dynamics model predicts soft tissue, contact, and fluid under calibration, as phases add anatomy and disturbance complexity). Each phase expands these frontiers and the supporting data + simulator + evaluation infrastructure. New phases do not add new code modules or new entries in any registry.

| Phase | Capability | Directive distribution coverage | Anatomy / object regions in embedding space | Deformable / interaction world-model frontier |
|---|---|---|---|---|
| 0 | Architectural spec frozen, infrastructure built, data pipeline online, simulator wrapped, embedding model and substrate bootstrapped from existing VLPs/VLAs | — | — | — |
| 1 | FLS peg transfer, human-competitive | Rigid-object manipulation language: pick, place, transfer, move-to-pose | Rigid manipulation: rings, pegs, tools | Rigid kinematics + contact priors only |
| 2 | Suturing in simulation | + Suturing language: drive needle, throw knot, pull through, tie | + Needle, suture material, rigid-ish tissue analog | Light interaction residuals; needle–tissue contact |
| 3 | Suturing on phantom | (same language coverage, hardware-validated) | + Phantom tissue analog | Real wrench/stereo logs; adapter refinement |
| 4 | Resection in simulation | + Soft-tissue interaction language: aspirate, cauterize, suction, retract, manage bleeding | + Soft tissue, cottonoid, fluid, void | **Primary ramp:** sim-grounded hyperelastic + learned residual; evaluation on predicted deformation, forces, fluid–void coupling |
| 5 | Cadaveric narrow-task autonomy | (same language coverage, real cadaveric anatomy) | + Real anatomy, brain-shift-compensated registration | World model under registration error and real material spread; same metrics as Phase 4 where applicable |
| 6 | Suturing in realistic conditions (bleeding, obstacles, living tissue) | + Cross-directive language: bleed-during-suture, retract-while-aspirate, etc. | + Living tissue dynamics, bleeding sources | Composition + bleeding sources; stress calibration on forces and interaction outcomes |
| 7+ | Procedural autonomy | + Procedure-level language: "remove tumor at left margin," "expose corridor," "achieve hemostasis" — the substrate composes lower-level behavior internally | Full surgical scene | Long-horizon rollouts depend on prior phases' world-model maturity |

Capability is measured per phase by:
- **Distribution coverage:** OOD scores for held-out directives in the target language region remain low.
- **Per-directive reliability:** contract satisfaction rate ≥ threshold on each directive's evaluation suite.
- **Cross-directive composition reliability** (Phase 6+): contract satisfaction on scenarios requiring composition across directive regions.
- **World-model calibration on deformable and interaction predictions** (Phase 4+): conformal or held-out coverage on predicted forces, deformations, and key interaction readouts at agreed nominal levels; regression suites on sim-grounded scenarios before cadaver stress (Phase 5).

The architectural commitments in §3 are sufficient to support all phases without rewriting the stack. New phases require: more training data covering the new language region, additional named affordance readouts (with their labeled examples), additional `ConstraintType` entries for any new safety classes, expanded simulator coverage, expanded conformal calibration sets, expanded evaluation suites — and substrate fine-tuning. They do not require new architecture or new code paths.

**Honest framing of substrate maturity by phase.** Early phases (1–3) have a substrate that is narrow but capable; cross-directive composition is not yet stress-tested because the directive set is small. The middle phases (4–5) introduce soft-body interaction, where the substrate must compose with the world model in ways early phases didn't require. The late phases (6+) require *cross-directive composition under novel conditions* (bleed during suture, etc.) — the central architectural claim of substrate-as-policy. If substrate compositional generalization underperforms there, those phases stretch out.

The mitigations (safety evaluator, contract monitoring, OOD detection, surgeon override) operate identically across all phases. Failure modes degrade autonomy gracefully — they don't cause unsafe behavior.

---

## 11. Open questions

Decisions that need to be made but are not yet settled. Each blocks specific downstream work.

1. **Foundation model for the semantic stream.** Candidates: SAM 2, MedSAM2, custom surgical VLA, an open VLA family (RT-2-X, OpenVLA, π0/π0.5). Decision criterion: best fine-tunability on surgical data + best swap path. Reversible.

2. **SSM variant for the substrate.** Mamba-2 vs. successors (S6, RecurrentGemma, etc.). Decision criterion: open-source quality + community momentum. Reversible.

3. **Equivariant attention library.** SE(3)-Transformers, Equiformer, eSCN, or a custom implementation. Decision criterion: ease of integration with the substrate. Reversible.

4. **Slot Attention variant.** Original Slot Attention vs. DINOSAUR vs. OSRT vs. custom. Decision: pick one for the streaming variable-N adaptation research effort. Partially reversible.

5. **Search algorithm in the slow planner.** MCTS vs. MPPI vs. CEM vs. trajectory optimization. May start with CEM (simplest) and migrate. Reversible.

6. **Simulator choice.** SOFA-Framework vs. alternatives (Bullet, MuJoCo for rigid, Drake, custom). Likely SOFA for soft-body; may use multiple. Partially reversible at the `Environment` interface.

7. **Verified NN tooling.** alpha-beta-CROWN vs. Marabou vs. custom. Decision criterion: integration with chosen NN frameworks. Reversible.

8. **Online fine-tune cadence and protocol.** How often LoRA adapters are trained on accumulated teaching cases; what evaluation gate they must pass before deployment; when adapters are merged into the base substrate. Operationally important; needs definition before Phase 1 ends.

9. **Initial directive set.** Which named directives to support in Phase 1. Driven by peg transfer requirements (`MoveToPose`, `Grasp`, `Place`, `Transfer` minimum). Each directive needs: a contract template, an evaluation suite, sufficient training data.

10. **Sensor calibration and synchronization stack.** Implementation choice for hard-real-time multi-stream sync. Standard but project-specific.

11. **Embedding dimensionality.** Initial `d ≈ 512` is a reasonable default; the right value is data- and downstream-task-dependent. Decide after early experiments. Reversible.

12. **Bootstrap path for the joint embedding model.** Start from an existing surgical VLP plus handcrafted projections from kinematics/force, or train from scratch. Start with the former; migrate to the latter as data grows. Reversible.

13. **Affordance subspace dimensionality and seeding.** Initial `d_aff` (dimension of the affordance subspace) and what labeled affordance examples to seed it with. Driven by the directive contracts of Phase 1 + Phase 2 (which need to be expressible in this space). The space itself is learned; we choose the seed examples and the dimensionality.

14. **Initial named-affordance set.** Which named affordances (readout heads on the affordance subspace) to provide initially. Driven by what surgeons want to use in directives + what the substrate needs as soft conditioning. New named affordances can be added later without re-training the subspace, so this is not a one-way decision.

15. **Constraint-type enum (closed) for surgeon tags.** The closed set of `ConstraintType` values the **sync command gate** enforces (with SafetySurface; §5.10). Initial proposal: `NO_GO_REGION`, `FORCE_LIMIT`, `PROXIMITY_ALERT`, `DO_NOT_ACTION`, `REQUIRED_PROXIMITY`. Adding a new constraint type requires extending verified gate code; this is rare and deliberate.

16. **Tag-anchoring interface for surgeons.** How the surgeon associates natural-language content with a slot at runtime — voice + visual confirmation? Touch interface? Eye gaze? Needs surgeon-team input. Decision affects clinical workflow, not architecture.

17. **Embedding-model retraining cadence.** How often to retrain, what triggers a retrain, what the rollout/rollback protocol is. Operationally important; needs definition before Phase 1 ends.

18. **Bootstrap path for the policy substrate.** Which existing VLA or surgical foundation model to fine-tune from for Phase 1, or whether to train from scratch on a small directive set. The choice has multi-month cost implications. Pi-zero / OpenVLA / surgical-VLA candidates. Reversible at Phase boundaries.

19. **Substrate output shape.** Trajectory horizons (50 ms? 500 ms?), action representation (joint vs. Cartesian vs. mixed), inclusion of force profile. Affects fast-controller interface. Decide early; expensive to change after Phase 1.

20. **Behavior contract schema details.** The mixed hard/soft schema is committed (§5.5); the details aren't. Specifically: which formal predicate language for safety_invariants and abort_conditions; how soft success criteria are converted to learned monitor heads; how affordance_requirements compose multiple predicates with logical operators; what counts as a quality_measure. Needs an early decision (before substrate training begins in earnest); a separate `CONTRACT_SCHEMA.md` document is the right shape.

21. **Substrate retraining vs. fine-tuning vs. continual.** When new data arrives, is the substrate fine-tuned, retrained from scratch, or updated continually? Continual learning has known failure modes (catastrophic forgetting). Fine-tuning is safer but slower to incorporate new data. Decision per-phase, with rollback.

22. **Substrate evaluation framework.** Behavioral testing is harder than unit testing. The evaluation suite per directive type needs to be defined: scenarios, contract-satisfaction metrics, surgeon-rated quality, safety-invariant violations, OOD coverage. This is a major engineering effort that pays for itself.

23. **Failure-mode taxonomy for the substrate.** What does it mean for the substrate to "fail," operationally? Drift from contract? Low confidence? OOD? Surgeon override? Need a taxonomy so failures can be categorized, logged, and used for training. Defined collaboratively with surgeons.

24. **Surgeon trust-building protocol.** Surgeons will not trust a substrate-driven autonomous system on day one. The progression from observe-only → assistive → narrow-autonomous → broader-autonomous is a trust-building protocol that needs to be designed with surgeon input. Architectural support is mostly already in §5.11 (operational modes) but the protocol itself is open.

25. **System attention conflict resolution.** The system attention module (v0.6, §5.9) aggregates four modulation channels (planner, safety evaluator, attention, halting) using deterministic rules (max for caution, union for targets). The current rules are fail-closed and commutative. Should there be a priority ordering among sources, or should conflicts be explicitly routed to the planner for re-deliberation? Current assumption: no priority; the deterministic aggregation is sufficient. Revisit if conflicts cause unexpected behavior in Phase 1.

26. **Safety assessment calibration protocol (v0.8).** How to evaluate and maintain the calibration of the safety evaluator's async assessment phase per phase transition. Specifically: (1) what constitutes a hard-constraint violation (ground truth), (2) how to measure calibration on held-out test set (target: ≤5% violations), (3) what is the acceptable veto rate on in-distribution commands (<15%), (4) what is the rollback protocol if calibration drifts, (5) how often to re-calibrate using cadaver data, simulation, or logged surgical procedures.

27. **Safety head training separation from policy (v0.8).** How to train interpretable safety readout heads on entity embeddings in a way that is provably separate from (and not subserverted by) policy training. Specifically: (1) what are the safety-critical labels we need (tissue damage risk, bleed recurrence risk, cumulative stress)? (2) how do we collect ground-truth labels (cadaver annotations, simulator ground truth, surgeon review)? (3) how to prevent policy loss from influencing safety head training (separate networks, separate data batches, orthogonal gradient projection, or hard separation into two training phases)? (4) how to evaluate that safety heads are learning safety, not task completion.

28. **Safety surface update rate and staleness bounds (v0.8).** The async safety assessment runs at ~10-20 Hz, publishing a safety surface consumed by the sync command gate at 100-500 Hz. Acceptable staleness bounds: how long can the surface be out-of-date before the gate should conservatively veto? Target: surface staleness <100 ms. If surface is stale (>100 ms), gate should (a) veto all commands, (b) use last-known surface with expanded margins, or (c) trigger an immediate re-assessment? Design decision affects safety-liveness tradeoff.

29. **Async/sync boundary: which checks belong in which phase (v0.8).** Some safety checks (e.g., "is this pose near a no-go region?") could run in either phase. Currently specified: heavy entity-state reasoning in async (~10-20 Hz), lightweight geometric checks in sync (100-500 Hz). But the boundary is not sharp. Decision criteria: latency, computational cost, information freshness, false positive rate. Needs formal design document before Phase 1.

---

## 12. Glossary

- **Affordance space (v0.4):** A continuous learned subspace `R^d_aff` of the joint embedding capturing the functional properties of slots. Each slot has an `affordance_vector` (its position in the space). Functionally similar slots have nearby vectors regardless of category. Replaces the closed list of named affordance predicates as the primary representation.
- **Affordance (named):** A *labeled region* of the affordance subspace, exposed as a readout head on the embedding (e.g., `graspable_along_axis`, `has_cutting_edge`, `absorbs_fluid`). Yields a confidence in [0, 1]. New named affordances can be added by labeling examples and training a new readout, without retraining the affordance subspace itself.
- **Behavior contract (v0.4):** A specification of behavior for a given directive, with two slot classes. **Hard slots** (formally verifiable predicates, force envelopes, forbidden constraint types, abort conditions) are consumed by the **safety evaluator** (SafetySurface + sync command gate; §5.10). **Soft slots** (free-form success criteria, learned quality measures, surgeon-stated intent, affordance requirements) are consumed by the substrate as conditioning and by the runtime via learned monitor heads. Used by the runtime for hard + soft monitoring and by the safety evaluator for envelope parameterization.
- **CBF (Control Barrier Function):** A function over scene state whose forward-invariance under commanded action implies safety. Used in the **sync command gate** as a hard constraint.
- **Safety filter (legacy):** Pre-v0.8 name for the final command gate. In v0.8 use **sync command gate** (part of the **safety evaluator**); see document header terminology.
- **Compositional generalization:** A model's ability to produce coherent behavior on combinations of training inputs it has not directly seen. The central architectural claim of substrate-as-policy.
- **Conditioning bundle:** The package of inputs the planner provides to the substrate runtime each tick: language directive, behavior contract, modulation parameters, attention targets. Replaces "skill + parameters" in v0.3.
- **Case log:** The accumulated logged frames + per-case metadata from all teleop and autonomous sessions. An operational artifact in the data pipeline (§8.1.3), used for surgeon UI, audit, training-data curation, and failure analysis. *Not* an architectural memory layer; not queried by the substrate at inference time.
- **Conformal prediction:** A statistical wrapper that produces calibrated prediction sets / intervals from any underlying predictor. Used here for calibrated uncertainty.
- **Contrastive learning:** Training objective in which similar examples are pulled together in embedding space and dissimilar examples are pushed apart. Foundation of the joint embedding model.
- **DAgger (Dataset Aggregation):** Imitation learning algorithm where the learner queries the expert at states the learner reaches, addressing distributional shift.
- **Diffeomorphism:** A smooth, invertible map with smooth inverse. Considered as a candidate local prior for tissue dynamics; *not* used as a global scene representation.
- **Directive (language directive):** A natural-language description of the surgeon-or-planner-intended behavior (e.g., "aspirate intact tumor tissue at slot tissue_007 with depth ≤ 2 mm"). Drawn from a continuous distribution of language patterns the substrate has been trained on (v0.4); not from an enumerated vocabulary.
- **Directive distribution (v0.4):** The distribution over natural-language directives the substrate has been trained on. Replaces the v0.3 implicit "directive vocabulary" framing. Capability is measured by coverage of this distribution, not by counting supported directive types.
- **Embedding (slot embedding):** A learned multi-modal vector representation of a slot. The primary data of the slot; types and affordances are derived from it.
- **Constraint type (v0.4):** Closed enum of formal constraint kinds the **sync command gate** enforces (with SafetySurface; §5.10): `NO_GO_REGION`, `FORCE_LIMIT`, `PROXIMITY_ALERT`, `DO_NOT_ACTION`, `REQUIRED_PROXIMITY`, etc. Adding a new constraint type requires extending verified gate code.
- **Surgeon tag (v0.4):** Two-layer surgeon-authored tag on a slot. **Constraint type** (deterministic, from the closed enum) + **constraint params** + **natural-language content** (open, free-form description) + **slot anchor**. The constraint type and params drive the **sync gate**; the natural-language content drives surgeon communication, case-log retrieval (surgeon UI, audit, training-data curation), and audit. Surgeon visual confirmation gates the tag's activation as a hard constraint.
- **HRN (Hierarchical Reasoning Network):** A class of recurrent neural network that performs deliberative iterative reasoning.
- **InfoNCE:** A widely used contrastive loss function. The default training objective for the joint embedding model's cross-modal alignment.
- **Joint embedding:** A single embedding space shared across modalities (vision, geometry, kinematics, force, language). Allows queries from any modality to retrieve neighbors in any other.
- **MCTS (Monte Carlo Tree Search):** Search algorithm used in the slow planner, guided by learned policy and value networks.
- **Modulation:** Continuous bias signals from the planner to the substrate (caution_level, time_pressure, attention_targets). Updates more frequently than directives; lets the planner re-bias active execution without re-issuing intent. Brain-inspired (PFC-PMC modulation analog).
- **MPC (Model Predictive Control):** Optimization-based control that re-plans at every step over a receding horizon. Used in the fast controller.
- **OOD (Out-of-Distribution):** Refers to scenarios that fall outside the substrate's training distribution. Detected via embedding distance and substrate confidence; used to gate autonomy.
- **Policy substrate:** The compositional learned policy network that is the action source in v0.3. A unified network conditioned on language directives, contracts, and runtime context, producing target trajectories or commanded actions. Replaces the discrete skill library. (Inference-time case retrieval / RAG is not in the runtime bundle; §8 case log is operational and training-time only.)
- **RAG (Retrieval-Augmented Generation):** Architecture pattern where a learned model is augmented with retrieval over an indexed corpus. *Not used architecturally in this system as of v0.5* — the case log is queried for surgeon UI / audit / curation but not as substrate input. Considered and rejected as inference-time retrieval; cross-procedure influence happens through training-data curation and online fine-tuning instead.
- **LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning method that trains small adapter matrices added to a frozen base model. Used here for fast surgeon-teaching fine-tunes (§8.2) on hours-to-overnight cadence.
- **Online fine-tuning:** The mechanism for fast assimilation of surgeon-flagged teaching cases without waiting for the next full retrain. LoRA-style adapters trained on flagged cases, gated by per-directive evaluation suites, deployed if passing, and merged into the base substrate at the next periodic full retrain.
- **SDF (Signed Distance Field):** Geometric representation of a region as a function whose value is the signed distance to the region's boundary. Used for no-go regions and anatomy.
- **SE(3):** The Special Euclidean group of rigid-body transformations in 3D (rotation + translation). The state space of tool poses.
- **Slot:** A persistent object instance in **scene state** (§5.3), held inside the **entity knowledge store**. Embedding-first: its primary data is a learned embedding; types, affordances, and dynamics adapters are derived; surgeon tags are explicit.
- **Slot Attention:** A neural mechanism that maintains object-centric representations as a fixed or variable number of "slots."
- **SSM (State Space Model):** Class of sequence models with linear-time scaling and explicit recurrent state. Mamba is the prominent example.
- **Substrate runtime:** The thin layer between the planner and the policy substrate that handles I/O, contract monitoring, OOD detection, re-conditioning, and escalation. Replaces "skill executor" in v0.3.
- **SystemAttentionState (v0.6):** Unified modulation state produced once per mid-loop tick by aggregating planner modulation, risk signals (from async safety assessment post-v0.8), attention targets, and halting flags. All control and perception components consume this single state instead of reading modulation signals independently. Reduces signal inconsistency and provides a single inspection surface for understanding system behavior.
- **System attention module (v0.6):** Deterministic coordination layer that aggregates four independent modulation channels (planner modulation, voluntary attention, safety evaluator async-assessment signals, adaptive halting) into a single unified `SystemAttentionState` consumed by all downstream components. Fail-closed: higher caution wins, slot unions are merged. Not learned; pure coordination.
- **Safety evaluator (v0.8):** Two-phase entity-informed safety system that reads entity state and publishes an interpretable safety surface. **Async assessment phase (~10-20 Hz):** reads entity embeddings, interaction digests, world model deformations; computes per-entity constraints (deformation-compensated no-go geometry, context-aware force envelopes, cumulative risk scores); exposes interpretable signals to planner; retains direct override path (≤50 ms). **Sync command gate phase (100-500 Hz):** fast deterministic checks of commanded actions against the safety surface and Layer 1 physical invariants; fail-closed: any error vetoes. Absorbs former risk system capabilities (conformal prediction, per-entity risk scoring, escalation triggers). Replaces former safety filter with honest model-dependence and dedicated training separation from policy.
- **Safety surface (v0.8):** Pre-computed, entity-informed constraint landscape published by the safety evaluator's async assessment phase (~10-20 Hz). Contains per-entity constraints (deformation-compensated no-go SDFs, context-aware force envelopes, cumulative risk scores, safety margins, confidence bounds), Layer 1 physical invariants, and calibrated uncertainty. The inspectable intermediate artifact: logged for audit, consumed by planner for reasoning, visualized in surgeon UI, checked by the sync command gate (100-500 Hz). Replaces former "decoupled safety constraint surface" (which was claimed but not actually decoupled).
- **Physical invariant layer (v0.8):** Layer 1 of the safety evaluator. Hard constraints about the robot hardware, not the patient anatomy: joint limits, workspace bounds, hardware force/velocity limits, singularity avoidance, communication watchdog. Formally verifiable, model-independent. Checked by both async assessment and sync command gate.
- **Async safety assessment (v0.8):** Phase 1 of the safety evaluator (~10-20 Hz). Reads entity state (embeddings, interaction digests, world model deformations), interprets learned entity context, computes per-entity constraints, publishes safety surface, exposes interpretable signals to planner. Absorbs conformal prediction, per-entity risk scoring, escalation triggers from former risk system. Produces SafetySurface data structure with per-entity EntityConstraints including deformation-compensated geometry, context-aware envelopes, tissue risk scores, and calibrated confidence.
- **Sync command gate (v0.8):** Phase 2 of the safety evaluator (100-500 Hz). Fast and thin. Checks each commanded action against the pre-computed safety surface + Layer 1 physical invariants. No heavy inference per command; just geometric and envelope checks. Produces SafetyDecision (pass/project/veto). Fail-closed: any error vetoes. Retains all properties of former safety filter (100% command coverage, no bypass paths, immediate surgeon override).
- **VLA (Vision-Language-Action model):** A foundation model class that maps observations and instructions to actions. RT-2, OpenVLA, π0 are examples. Bootstrap candidates for the policy substrate.
- **VLP (Vision-Language Pretraining):** Pretraining paradigm for vision-language models. Surgical VLPs (SurgVLP, etc.) are the bootstrap candidates for the joint embedding model.

---

## 13. Document maintenance

This is a living document. Architectural commitments (§3) and the layer specifications (§5) change rarely; status (§9) and open questions (§11) change often. Update on every architectural decision; review formally on every phase transition.

Owner: TBD
Review cadence: monthly during build phase
Versioning: semver-like (major bumps for §3 changes; minor for §5; patch for §9, §11)
