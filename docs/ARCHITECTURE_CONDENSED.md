# Condensed Autonomous Surgery Architecture

**Source:** `ARCHITECTURE.md` draft v0.8
**Purpose:** Condensed reference for the proposed autonomous surgery architecture — what the system is, how the components interact, and which technologies underpin each layer.

---

## 1. Executive Summary

The proposed architecture is a learning-centric autonomous surgical robotics stack for neuroArm. It is designed to progress from controlled dexterity benchmarks, through narrow cadaver-validated autonomy, toward procedural autonomy without replacing the stack at each phase.

The core design is a compositional learned policy substrate: a neural policy that consumes a structured entity representation, language directives, behavior contracts, risk state, surgeon input, and recent context, then emits short-horizon target trajectories for a fast controller to execute. The system's surgical knowledge lives in continuous multimodal embeddings, not in a fixed library of hand-coded skill modules.

Rigid spatial structure is handled with geometry-first inductive bias (SE(3)-equivariant layers in the action stream). Non-rigid tissue behavior — compliance, deformation, cutting, bleeding, and fluid — is modeled by the embedding-conditioned world model, which fuses stereo, kinematics, force, contact history, and entity embeddings over a physics prior. Phases (§9) expand the capability frontier with data, simulation coverage, and evaluation, not by replacing the stack.

The architecture separates:

- **Learned soft reasoning:** perception embeddings, affordances, world-model residuals, policy behavior, confidence, OOD scoring, and success-quality estimates.
- **Hard safety guarantees:** surgeon-authored constraints, formal predicates, control barrier functions, force envelopes, no-go regions, and fail-closed command filtering.

Learning is used where flexibility and generalization matter. Engineered verification is used where clinical safety requires deterministic guarantees. These two concerns never collapse into the same module.

---

## 2. Architectural Thesis

Most practical surgical robotics systems are built as brittle pipelines of task-specific modules, explicit state machines, and fixed action libraries. Those systems can be made reliable for narrow demonstrations, but they struggle when the surgical scene changes, when a tool or tissue behaves differently, or when two behaviors need to compose in a way that was not explicitly programmed.

This architecture treats autonomy as a learned control problem over structured surgical state. It keeps the parts that traditional robotics does well — explicit geometry, fast feedback control, safety evaluation with entity-informed assessment, simulator parity, logging, and verification — and replaces the brittle center of the conventional stack with a learned policy substrate trained from teleoperation, simulation, reinforcement learning, surgeon corrections, and contract-conditioned evaluation.

The goal is not to remove structure. The goal is to put structure in the right places:

- A persistent entity representation gives the learned system stable handles for tools, tissue, fluid, cottonoids, voids, and surgeon-tagged anatomy.
- Multi-loop control gives each decision the right time scale.
- Behavior contracts tell the substrate what success means while giving the safety evaluator formal hard constraints.
- A safety evaluator reads entity state and publishes an interpretable safety surface, checking every command before it reaches the robot.

---

## 3. System Overview

At runtime, the system operates as a hierarchy of loops around a shared entity knowledge store:

1. **Surgeon interface:** supplies goals, permissions, tags, corrections, and overrides.
2. **Slow planner (~0.5–2 Hz):** receives signals from world model (prediction errors) and risk system (assessments); decides what to consolidate into entity state; translates surgeon goals into language directives, behavior contracts, modulation signals, attention targets, and escalation triggers.
3. **Policy substrate runtime (~5–20 Hz):** runs the learned compositional policy, monitors behavior contracts, handles OOD and confidence checks, and emits short-horizon target trajectories.
4. **Fast controller (~100–500 Hz):** performs MPC, visual servoing, and force-aware tracking of the substrate's target trajectory.
5. **Safety evaluator (asynchronous assessment ~10-20 Hz + synchronous command gate 100-500 Hz):** async assessment reads entity state and publishes a safety surface; sync gate checks commands against the safety surface + Layer 1 physical invariants. Absorbs former risk system capabilities (conformal prediction, per-entity risk scoring, escalation triggers). Replaces former safety filter with honest model-dependence.

### Data flow between components

The entity knowledge store is the shared workspace. All learned components read entity embeddings, which carry both current observation (from perception) and distilled interaction history (from planner-gated consolidation). The safety evaluator reads entity state (embeddings, interaction digests, world model deformations) to assess real anatomy and publishes an interpretable safety surface that the planner reads for reasoning. The safety evaluator operates in two phases: async assessment (~10-20 Hz) reads entity state and computes constraints; sync command gate (100-500 Hz) checks commands against the pre-computed safety surface + Layer 1 physical invariants.

Information flows through two paths into the entity knowledge store:

- **Direct path (ungated):** Perception refreshes current observations each cycle (~10–20 Hz). Surgeon corrections write authoritative overrides unconditionally.
- **Consolidated path (planner-gated):** World model prediction errors, risk assessments, and event records flow to the slow planner. The planner decides what gets consolidated into each entity's interaction digest, at what priority, and with what framing. A digest update network executes planner-selected consolidations at 0.5–2 Hz.

The safety filter sits between every commanded action and the robot, regardless of whether the command came from the learned substrate, the controller, or surgeon teleoperation.

---

## 4. Design Decisions

### 4.1 Persistent Entity Representation with Continuous Embeddings

**Decision.** Represent the surgical world as a variable-cardinality set of persistent entities. Each entity has a stable identity, pose, geometry, a learned multimodal embedding, embedding uncertainty, derived type estimates, affordance vector, dynamics adapter, and explicit surgeon tags.

The entity's embedding is the primary representation. Type labels and named affordances are derived readouts — small classifier heads on the embedding — not the source of truth. Surgeon-authored tags remain separate and authoritative for hard safety constraints.

Entities are created and destroyed dynamically as the surgical field changes. The perception system's binding mechanism (a variable-cardinality slot attention head) assigns persistent identities to observed objects and maintains identity across frames and occlusion. The term "slot attention" refers to this perception binding mechanism; the entities themselves are not constrained to a fixed number or a fixed-size data structure.

**Why this design.** Surgery is object-rich and interaction-heavy. The system needs persistent handles for tissue regions, vessels, tools, cottonoids, blood pools, resection voids, and surgeon-tagged anatomy. A global image feature or monolithic latent state does not provide stable references for planning, control, audit, or surgeon communication.

The embedding-first choice preserves continuous structure. A novel forceps may be functionally similar to a known forceps before the system has a confident label. Two cottonoids may share a type but differ in saturation dynamics. A continuous embedding represents those similarities and differences without exploding the type system.

**Tumor resection illustrates why continuous state matters.** The tumor entity is not a static class; it is a changing volume, margin geometry, and risk surface relative to tagged vessels and eloquent tissue. Clinical success is often not equivalent to "zero residual tumor." Intentional subtotal resection, maximal safe resection with planned small remnant, or debulking under a specific contract can all be successful outcomes. A binary or coarse label (`present`/`absent`/`partial`) conflates situations that require opposite cautions and gives the model almost no usable gradient signal for learning safe continuation. The embedding-first representation keeps the primary state continuous so similarity and difference along the resection trajectory can be learned rather than erased at the first discrete threshold.

**Physically overlapping entities illustrate why per-entity decomposition is correct.** Consider a vein draped over a tumor. They occupy overlapping space, are mechanically coupled, and appear visually intertwined — but they must be separate entities. They have different dynamics (pressurized vessel vs. bulk tissue mass), different affordances (preserve vs. resect), different risk profiles (catastrophic hemorrhage vs. margin management), and different surgeon constraints (hard no-go vs. active removal target). If the system merged them into a single entity, it could not express "remove this, preserve that" — which is the core surgical problem. The entity decomposition gives the planner, safety filter, and surgeon UI separate handles for each, while the world model's interaction terms (§4.9) and spatial proximity in embedding space capture exactly how they are physically coupled. A directive like "resect tumor at inferior margin, maintaining 2mm clearance from tagged vessel" only makes sense with two distinct entities carrying separate constraints. The relationship between them is not lost — it is represented in the right place: interaction coupling in the world model, spatial structure in the SE(3)-equivariant features, and strategic reasoning in the planner.

**Entity splitting is a known failure mode of slot attention.** The competitive softmax in the slot attention loop encourages each perceptual feature to be predominantly explained by one slot, but does not strictly enforce one-to-one entity-to-slot mapping. A large or heterogeneous entity (e.g., a multi-lobed tumor, a long vessel) could have different spatial regions claimed by different slots if those regions look sufficiently different in feature space. This "over-segmentation" is mitigated by several architectural properties: persistent identity tracking across frames makes sporadic splits unstable; entity-level training supervision penalizes gratuitous splitting; and the downstream planner would observe two entities with near-identical embeddings and overlapping spatial extent. However, genuinely ambiguous boundaries (partially resected tissue), under-training, and distribution shift remain risk factors. Custom streaming adaptations should include explicit merge/split detection — slot-merging heuristics or a learned merge/split head that runs after the binding pass — as part of the research investment in streaming variable-cardinality binding.

**Technologies.** Variable-cardinality slot attention (DINOSAUR / OSRT / custom streaming adaptation) for perception binding. Joint multimodal embedding model (CLIP-style contrastive training across vision, geometry, kinematics, force, language) for the primary entity representation. SE(3)-equivariant networks (Equiformer / SE(3)-Transformer) for geometric features fed into the embedding.

**Alternative considered:** A typed scene graph with explicit object classes and per-class properties. This forces discrete categorization into a closed schema, generalizes poorly to novel instruments and tissue states, and makes cross-type similarity hard to exploit.

**Alternative considered:** A global latent scene representation (pure VLA-style). Easier to train, but hard to inspect, constrain, audit, and poorly suited to entity-anchored surgeon tags or formal safety predicates.

### 4.2 Entity Knowledge Store with Planner-Gated Consolidation

**Decision.** Each persistent entity carries layered state that evolves during a procedure:

- **Current observation** (refreshed ~10–20 Hz): pose, visual features, contact state from perception. What we see right now.
- **Interaction digest** (evolves at planner rate, 0.5–2 Hz): distilled understanding of past interactions — tissue compliance, deformation patterns, bleeding tendencies, tool response history. Patient-specific calibration learned during the procedure.
- **Entity embedding** (fuses observation + digest): the primary representation consumed by all learned components. Carries both "what this entity looks like now" and "what we have learned about it."
- **Pre-computed priors** (optional): consumer-specific derivations from the digest (e.g., linearized compliance for the world model, risk adjustment for the risk system).

The two-path input model determines how entity state evolves:

- **Direct path (ungated):** Perception refreshes current observations every cycle. Surgeon corrections write authoritative overrides unconditionally.
- **Consolidated path (planner-gated):** World model prediction errors, risk assessments, and event records flow to the planner as input signals. The planner — which has strategic context, goals, and full situational awareness — decides what gets consolidated into each entity's interaction digest and at what priority. A digest update network executes the consolidation.

All learned components read the entity embedding, which already carries distilled interaction history, dynamics calibration, and strategic context. No component re-attends to raw interaction history; the digest absorbs the significant information once.

**Why this design.** Surgery requires both high-resolution short-term memory (controlling an active bleed) and low-resolution long-horizon memory (tumor state from 10 minutes ago). A fixed history window forces a false tradeoff. Patient-specific calibration should be explicit and shared, not scattered across adapter weights, SSM recurrent state, and learned confidence heads. When the world model learns "this tissue is unusually compliant," that learning persists in the entity's interaction digest where all consumers can benefit. When the planner decides "this region matters because it is adjacent to tagged critical anatomy," that strategic context persists where the policy substrate can use it.

The planner's role as memory gate is principled: it has strategic context so it can decide significance far better than threshold-based event detection alone. This also gives the planner a natural mechanism to modulate the system's attention budget — denser consolidation during uncertain or novel situations, sparser during routine execution.

**Neuroscience parallel.** The design maps onto hippocampal-cortical complementary learning: an event archive serves as fast, high-fidelity episode recording (hippocampal); the interaction digest is slow, consolidated understanding (neocortical); the planner provides strategic modulation of what gets remembered (prefrontal). World model prediction errors to the planner parallel dopaminergic prediction-error signaling; risk signals parallel amygdala arousal.

**Technologies.** Digest update network: gated recurrent model (GRU or transformer-based write mechanism). Planner consolidation controller: deterministic logic layer in the planner. Event archive: append-only store indexed by entity ID + event type + timestamp.

**Alternative considered:** Treat scene state as a static snapshot, updated by perception each cycle, with temporal memory implicit in adapter weights and recurrent state. This scatters patient-specific calibration across components, creates no explicit mechanism for deciding what is worth remembering, and fragments multi-scale temporal requirements.

**Alternative considered:** Direct world-model and risk-system writes to entity state without planner gating. This creates write conflicts between multiple sources and loses the strategic memory control the planner provides.

### 4.3 Separation of Safety Dedication from Task Commitment

**Decision.** The safety evaluator is dedicated to safety—trained, evaluated, and architected separately from policy training. Entity state informs safety assessment, but the safety evaluator cannot be influenced by policy optimization, and learned signals cannot relax safety margins. The discipline is dedication, interpretability, conservatism, and training separation—not the false claim of formal independence from all learned representations.

**Why this design.** Clinical autonomy cannot rely on the policy to police itself. If the policy could relax safety margins or if safety training were intertwined with task completion, the system would be unsafe-by-design. The separation is not about preventing learned signals from reaching the safety system—entity embeddings must inform it to assess real anatomy (e.g., brain shift, bleed history). The separation is about never allowing the policy's task optimization to influence what counts as safe.

**Technologies.** Safety evaluator: async assessment reads entity state and publishes safety surface; sync gate applies formal checks + CBFs + verified neural sub-checks. Safety head training: separate from policy, labeled for safety outcomes (tissue damage, bleed recurrence), evaluated per-phase. Conformal prediction: calibrate uncertainty in safety assessments.

**Alternative considered:** Learned safety inside the policy or as a learned reward. This entangles safety behavior with task completion, makes formal reasoning impossible, and creates perverse incentives (the policy learns to hide or downweight safety concerns).

### 4.4 Multi-Loop Control Hierarchy

**Decision.** Separate loops for deliberation, learned policy execution, fast control, and risk monitoring:

| Loop | Rate | Role |
|---|---|---|
| Slow planner | ~0.5–2 Hz | Directive composition, deliberative reasoning, memory consolidation, escalation |
| Policy substrate runtime | ~5–20 Hz | Learned behavior generation, contract monitoring, OOD/confidence checks |
| Fast controller | ~100–500 Hz | MPC, visual servoing, force-aware control |
| Risk system | Asynchronous | Uncertainty and risk monitoring with direct override |

**Why this design.** Surgical autonomy has conflicting timing requirements. Deliberative reasoning about goals, anatomy, and escalation can take hundreds of milliseconds. Contact control and servoing need millisecond-scale response. A single loop cannot satisfy both.

**Technologies.** Slow planner: recurrent reasoning module (Loop Transformer / Universal Transformer / HRN-class with adaptive halting) + MCTS / MPPI / CEM for directive search. Fast controller: MPC, visual servoing, impedance/admittance control. System attention coordination: deterministic aggregation layer that merges planner modulation, risk signals, attention targets, and halting flags into a unified per-tick state consumed by all components.

**Alternative considered:** A single behavior loop or state machine at one dominant rate. Slow reasoning blocks fast control, and fast control pressures reasoning into shallow decisions.

**Alternative considered:** Substrate directly drives the robot at fast-loop rate. This ties neural inference latency to servo latency and makes safety verification harder. The substrate instead emits short-horizon target trajectories, and the fast controller tracks them.

### 4.5 Dual-Stream Perception

**Decision.** Two perception streams feeding a unified entity-binding head:

- **Action stream:** low-latency geometric perception for control, using stereo, kinematics, and force. Target latency tight enough for servoing (≤15 ms p99).
- **Semantic stream:** slower, higher-resolution semantic perception for planning and risk, using foundation-model segmentation and surgical fine-tuning (≤150 ms p99).

The streams are not required to share the same backbone or architecture. Each may use a different model family, inductive bias, and training recipe as long as outputs align in the entity-binding fusion stage.

**Why this design.** Control and semantic understanding need different compromises. The fast loop needs geometry, pose, contact, and motion fields quickly. The planner and risk system need richer semantic labels, anatomy, segmentation, and context but can tolerate higher latency. Splitting streams avoids forcing one network to be both a millisecond-scale geometric estimator and a heavy semantic reasoner.

**Technologies.** Action stream: SE(3)-equivariant attention layers (SE(3)-Transformer / Equiformer). Semantic stream: foundation model (SAM 2 / MedSAM2 / surgical VLA backbone), fine-tuned on surgical data. Entity binding: variable-cardinality slot attention head (DINOSAUR / OSRT / custom streaming adaptation) that integrates both streams and maintains persistent entity identities across frames.

**Alternative considered:** A single perception model serving both fast control and semantic understanding. A model optimized for semantic richness is often too slow for control; a model optimized for low latency is too shallow for surgical reasoning.

### 4.6 SE(3)-Equivariant Transformers for Action Geometry

**Decision.** Use SE(3)-equivariant attention layers in the action stream and tool/tissue interaction modeling. These layers bake the symmetries of 3D rigid motion into the architecture: if the scene rotates or translates, the representation transforms consistently.

**Why this design.** Surgical manipulation is fundamentally geometric. Tool poses, contact frames, rotations, translations, and relative spatial relationships matter. An equivariant model encodes these symmetries structurally instead of forcing the model to learn them from data.

**Scope and hand-off.** SE(3) equivariance is a rigid-frame prior. It stabilizes representations of tool pose, contact frames, and relative spatial layout. It does not encode constitutive soft-tissue physics, plastic deformation, topology change, or bleeding. Those behaviors are modeled in the embedding-conditioned world model (§4.9), which consumes action-stream geometry together with stereo, force/torque, and contact history.

**Technologies.** SE(3)-Transformer, Equiformer, or related equivariant attention libraries.

**Alternative considered:** Standard CNN/ViT/Transformer that learns geometric invariances from data. This wastes data on symmetries that are already known and behaves inconsistently under pose changes.

### 4.7 Compositional Policy Substrate

**Decision.** The action source is a single learned compositional policy network. It consumes the full context bundle — entity embeddings, language directive, behavior contract, risk state, planner modulation, surgeon directives, recent events — and produces short-horizon target trajectories for the fast controller to execute.

The substrate composes behavior in real time across patterns learned during training. It can recombine primitives across what would conventionally be separate "skills" (e.g., aspirate-while-watching-for-bleed) without those compositions having been pre-programmed. Named surgical behaviors (`Aspirate`, `Cauterize`, `Suture`, `Retract`) persist as language conditioning, audit labels, retrieval keys for case-log queries, and anchors for behavior contracts. They are not code modules or state machines.

**How composition works.** Three mechanisms:

1. **Language conditioning composition.** Multiple directives or contract clauses combine in the conditioning encoder. The substrate learns to produce behavior consistent with all clauses simultaneously.
2. **Modulation as continuous bias.** The modulation channel (caution_level, attention_targets, time_pressure) is a continuous input that re-weights the substrate's output at any moment without changing the directive.
3. **Training-distribution coverage.** The substrate is trained on a wide distribution of demonstrations covering many directive combinations and situations. Compositional generalization across directives emerges from training; if a particular composition fails to generalize, the answer is more diverse training data.

**Why this design.** Real surgical behavior is compositional. A surgeon does not run a separate hard-coded program for every combination of simultaneous objectives. Humans generalize by composing learned motor patterns under context and attention. A learned substrate can share representations across behaviors, use language and contracts as conditioning, and generalize within its training distribution. The tradeoff is real: debugging is harder, data requirements are higher, and behavior must be evaluated contractually. The architecture accepts those costs because compositionality is central to the long-term autonomy goal.

**Technologies.** Hybrid SSM-Transformer (Mamba-2 / Jamba / Hymba / Zamba lineage) as the recurrent policy core. SSM layers carry persistent state across the streaming procedure with linear-time scaling; attention layers handle cross-modal fusion and long-range queries. Bootstrap from existing VLA (Pi-zero / OpenVLA / surgical-VLA family). LoRA-style fine-tuning for fast surgeon-teaching updates.

**Alternative considered:** A library of fixed skills (`Grasp`, `Aspirate`, `Cauterize`, etc.), each with its own state machine, preconditions, postconditions, and implementation. Novel compositions require new skills. Skill interactions become combinatorial. The system becomes brittle when the surgical context falls between predefined skills. Validation shifts to module boundaries and misses emergent failures from composition.

### 4.8 Language Directives and Mixed Behavior Contracts

**Decision.** The planner emits natural-language directives plus behavior contracts. Directives are drawn from a continuous language distribution, not a closed vocabulary. Contracts contain both hard and soft components.

- **Hard:** formal safety invariants, abort conditions, force envelopes, forbidden constraint types, required constraint types. Consumed by the safety filter and deterministic contract monitoring.
- **Soft:** success criteria, quality measures, surgeon intent summaries, affordance requirements. Consumed by the substrate as conditioning and evaluated via learned monitor heads.

**Why this design.** Surgeons naturally express intent in language, and many surgical success criteria are not clean formal predicates. "Remove this tissue cleanly without damaging adjacent parenchyma" has hard safety constraints and soft quality goals. The mixed schema captures what surgeons actually want without forcing soft success criteria into formal predicates and without letting fuzzy criteria contaminate the safety surface.

**Technologies.** Goal-to-directive translator: LLM-style decomposition fine-tuned on surgical goal/decomposition pairs. Contract monitoring: deterministic evaluation for hard slots (every tick); learned text-conditioned classifier heads for soft slots. OOD detection: embedding-space distance from training distribution, conformal prediction thresholds.

**Alternative considered:** A fixed command vocabulary or API with structured parameters only. This cannot capture nuanced surgical intent and forces soft quality goals into rigid predicates.

### 4.9 Embedding-Conditioned World Model

**Decision.** A unified world model conditioned on entity embeddings and per-entity dynamics adapters predicts future entity states and interactions. The model combines physics priors, learned residuals, and contact/interaction terms. It exposes latent rollouts for planning and interpretable readouts for safety and risk.

**How entity embeddings drive prediction.** Because dynamics are conditioned on entity embeddings within the full entity knowledge store (poses, contact edges, events, interaction digests), the world model reasons over the whole surgical field in one forward pass. Anatomically or geometrically neighboring regions tend to land in nearby regions of embedding space, so contact or deformation observed on one entity can sharpen predictions for adjacent entities before the tool touches them. Bleed risk and fluid propagation can spread through coupled entity updates and interaction terms — a learned prior validated by calibration and phase-gated evaluation, not assumed a priori.

**Multimodal fusion for deformable tissue.** Compliance, deformation, and interaction coupling are inferred by relating SE(3)-structured action geometry to stereo-evolved surface state, applied wrenches, and temporal history. The result is a predictive model (residual over the physics prior) with calibrated uncertainty — not a formal guarantee that the patient matches a particular material law. Hard safety remains with surgeon tags and the safety filter.

**Prediction errors as calibration signal.** When the world model's prediction diverges from observation, these prediction errors flow to the planner. The planner decides whether to consolidate the lesson into the entity's interaction digest, where it becomes patient-specific calibration available to all downstream components.

**Technologies.** Physics prior: SOFA-Framework (open source, mature soft-body) + standard rigid-body kinematics. Learned residual: neural network conditioned on entity embedding + dynamics adapter. Per-entity dynamics adapter: small parameter set (or hypernetwork-conditioned weights) initialized from embedding nearest neighbors, refined online from observed interactions. Latent rollouts: Dreamer / TD-MPC2 lineage. Interpretable readouts: predicted poses, predicted forces, anatomy SDFs, no-go region distances with calibrated uncertainty (conformal prediction).

**Alternative considered:** Separate dynamics models for each object class and interaction pair. This requires many pairwise models, does not transfer across similar entities, encourages duplicated code, and handles novel objects poorly.

**Alternative considered:** Pure simulator dynamics without learned residuals. Soft tissue, bleeding, and patient-specific anatomy are too variable for an analytic simulator alone.

### 4.10 Safety Evaluator as a Dedicated System Path

**Decision.** Safety is a dedicated two-phase system separate from the policy substrate. Async assessment (~10-20 Hz) reads entity state and publishes an interpretable safety surface. Sync command gate (100-500 Hz) checks every command against the safety surface + Layer 1 physical invariants. The evaluator is substrate-output-agnostic: it does not care whether a command came from policy, planner, or teleoperation.

**Layer 1 (Physical invariants):** Formally verifiable, model-independent constraints about the robot: joint limits, workspace bounds, hardware force/velocity limits, singularity avoidance, communication watchdog.

**Layer 2, Phase 1 (Async safety assessment):** Reads entity embeddings, interaction digests, world model deformations, surgeon tags. Computes per-entity constraints: deformation-compensated no-go geometry (brain-shift-aware), context-aware force envelopes (entity that bled → tighter threshold), cumulative risk scores, anomaly flags. Publishes safety surface consumed by planner for reasoning and by sync gate for command gating.

**Layer 2, Phase 2 (Sync command gate):** Fast and thin. Checks commands against pre-computed safety surface + Layer 1 invariants. Geometric intersection tests, envelope comparisons, anomaly checks. No heavy inference per command.

**What is verified.** Layer 1 logic and sync gate logic are formally verifiable. Layer 2 async assessment is calibrated via conformal prediction and phase-validation; it acknowledges model-dependence (entity embeddings, world model deformations) while maintaining conservatism (uncertain → veto) and training separation from policy.

**Technologies.** Async assessment: conformal prediction wrappers, per-entity risk scoring (absorbed from risk system), safety head training (separate from policy). Sync gate: CBFs, verified neural sub-checks (alpha-beta-CROWN), <2K LOC target.

**Alternative considered:** Rely on controller limits, policy training, or an ad hoc collection of runtime checks. This does not produce a clean verification surface.

### 4.11 Learning-Centric Data and Training Flywheel

**Decision.** Treat data, logging, simulation, evaluation, and fine-tuning as first-class architecture, not downstream tooling. Every teleop session, phantom run, cadaver lab, simulator rollout, correction, safety decision, and outcome label feeds the training pipeline.

**Why this design.** The learned substrate, embedding model, world model, confidence heads, and planner all improve with logged multimodal experience. The architecture's long-term advantage is the accumulation of surgical data in the exact format the system consumes.

**Technologies.** Data pipeline: synchronized multimodal logging with strong schema versioning (all sensor streams, scene state, commands, safety decisions, surgeon input, outcomes). Case log: operational store for surgeon UI, audit, curation, and failure analysis (vector DB + similarity search over logged embeddings). Training modes: imitation from teleop, imitation from simulator, RL in simulation (contract-based reward), DAgger / human-in-the-loop correction, LoRA-style fast fine-tuning for surgeon teaching cases. Domain randomization: per-patient anatomy variants from preop-MRI distributions, randomized materials, lighting, and camera calibration.

**Alternative considered:** Build autonomy features first and add data infrastructure later. Early demonstrations then fail to create a reusable training asset, failure analysis becomes anecdotal, and evaluation cannot scale with capability.

### 4.12 Sim/Real Parity Through a Shared Environment Interface

**Decision.** Simulator and real robot implement the same `Environment` interface. Code above that boundary runs unchanged in simulation and on neuroArm.

**Why this design.** The system needs simulator rollouts for training, evaluation, stress testing, and risk calibration. If sim and real stacks diverge, simulator results stop being useful.

**Technologies.** Simulator: SOFA-Framework wrapped to expose the project's `Environment` interface. The same interface is implemented by `SimEnvironment` (SOFA-backed) and `RealEnvironment` (neuroArm-backed). Domain randomization covers anatomy, materials, lighting, and sensor calibration.

**Alternative considered:** Separate simulator code and robot code bridged by custom adapters. This duplicates logic that drifts, leaks sim-only hacks into training, and makes evaluation results less trustworthy.

---

## 5. Component Summary

| Component | Purpose | Key Technologies | Status |
|---|---|---|---|
| **Sensors** | Stereo microscope, kinematics, force/torque, audio, intraop ultrasound, preop MRI/CT. All synchronized to a common clock, continuously logged. | Hardware integration, ≤1 ms sync | Engineering |
| **Action stream encoder** | Low-latency geometric perception for fast control. Outputs tool poses, contact state, tissue surface motion field, force estimates. ≤15 ms p99. | SE(3)-equivariant attention (Equiformer / SE(3)-Transformer) | Off-the-shelf + integration |
| **Semantic stream encoder** | Rich semantic perception for planning and risk. Outputs segmentation, classification, anatomy labels. ≤150 ms p99. | Foundation model (SAM 2 / MedSAM2 / surgical VLA backbone) | Off-the-shelf, fine-tuned |
| **Entity binding (slot attention)** | Integrates dual-stream output into persistent entity identities. Variable cardinality; handles spawn, despawn, occlusion. ≤30 ms p99. | Variable-cardinality slot attention (DINOSAUR / OSRT / custom) | **Research investment** (streaming variable-N) |
| **Joint embedding head** | Produces the primary multimodal embedding for each entity. Types, affordances, dynamics adapters are derived from this. ≤30 ms p99. | Multi-modal contrastive encoder (CLIP-style across vision, geometry, kinematics, force, language) | **Primary research investment** (12–24 months) |
| **Entity knowledge store** | Shared workspace accumulating patient-specific understanding. Each entity: current observation + interaction digest + entity embedding + pre-computed priors. | Digest update network (GRU / transformer write mechanism), deterministic consolidation controller, append-only event archive | Engineering + research |
| **World model** | Predicts future entity states and interactions. Embedding-conditioned dynamics with physics priors, learned residuals, per-entity adapters. Reports prediction errors to planner for consolidation. | SOFA physics prior + learned residual, per-entity dynamics adapter (hypernetwork), latent rollouts (Dreamer / TD-MPC2 lineage), conformal prediction for calibrated readouts | **Primary research investment** (18–36 months for soft-tissue + interaction) |
| **Slow planner** | Directive composition, deliberative reasoning, memory consolidation gating, escalation. The single consolidation authority for entity knowledge. ≤2 s p99 per step; modulation updates ≥5 Hz. | Recurrent reasoning module (Loop Transformer / Universal Transformer / HRN), MCTS / MPPI / CEM for directive search, LLM-style goal-to-directive translation | Research (reasoning module) + engineering |
| **Policy substrate runtime** | Feeds context to the substrate, receives target trajectories, monitors contracts (hard + soft), checks OOD/confidence, escalates when needed. ≤50 ms p99 per step. | — (thin runtime layer around the substrate) | Engineering |
| **Compositional policy substrate** | The action source. Composes behavior from language + contracts + context + learned patterns. Emits short-horizon target trajectories at ~5–20 Hz. | Hybrid SSM-Transformer (Mamba-2 / Jamba lineage), bootstrap from VLA (Pi-zero / OpenVLA), LoRA fine-tuning | **Primary research investment — largest in the project** (multi-year) |
| **System attention coordination** | Aggregates planner modulation, risk signals, attention targets, and halting flags into a unified per-tick modulation state. All components consume this single state. | Deterministic aggregation logic (max for caution, union for targets, fail-closed) | Engineering |
| **Fast controller** | Tracks substrate target trajectories at ≥100 Hz with tight sensor feedback. | MPC, visual servoing, impedance/admittance control | Off-the-shelf |
| **Risk system** | Calibrated uncertainty, per-entity risk scores, escalation triggers, direct safety override. ≥10 Hz updates; ≤50 ms override path. | Conformal prediction wrappers (streaming variant is small research), per-entity risk scoring | Engineering + small research |
|| **Safety evaluator (v0.8)** | Two-phase entity-informed safety. Async assessment (~10-20 Hz) reads entity state, publishes interpretable safety surface. Sync command gate (100-500 Hz) checks commands against surface + Layer 1 physical invariants. Absorbs risk system capabilities. Final arbiter, fail-closed. | Layer 1: CBFs, verified sub-checks. Layer 2 async: conformal prediction, safety head training (separate from policy). ≤2 ms p99 sync gate. | Engineering (sync gate) + research (async assessment) |
| **Data pipeline and case log** | Synchronized multimodal logging, case storage, curation, failure analysis. The training flywheel. | Vector DB + similarity search, schema-versioned immutable storage | Engineering |

---

## 6. Rejected Architectural Patterns

### 6.1 Fixed Skill Library as the Autonomy Center

A conventional safety-conscious robotics stack selects from a fixed skill library (`Grasp`, `Aspirate`, `Cauterize`, etc.) where each skill runs a hand-designed state machine. This is a reasonable baseline for narrow autonomy — easier to debug, unit test, and certify for a small number of scripted behaviors.

The architecture rejects it as the end-state because it does not scale to surgical variability:

- Surgery contains too many intermediate states for closed skill definitions.
- Skill composition is combinatorial — novel combinations require new skills.
- Hand-coded state machines are brittle under unexpected tissue behavior.
- Per-skill validation does not prove behavior under novel combinations.
- A fixed library makes autonomy expansion a software-engineering bottleneck instead of a data-and-evaluation problem.

The proposed architecture keeps the conventional stack's strongest parts: explicit entity state, classical fast control, simulator parity, logging, and a verified safety filter. It replaces the brittle center — the fixed skill library — with a learned substrate whose capability frontier expands through data, training, evaluation, and phased deployment.

### 6.2 Inference-Time Episodic Memory / RAG

The system does not retrieve past cases into the substrate at inference time. Memory lives in substrate weights, recurrent state, and the persistent entity knowledge store (interaction digests + current observations + strategic context). The case log remains operational for surgeon UI, audit, training-data curation, and post-hoc failure analysis.

**Why rejected.** Inference-time retrieval is often a substitute for training. If a case is important for behavior, it should enter the substrate through fine-tuning and evaluation, not as an unverified in-context example during surgery. Retrieved cases may be superficially similar but clinically different. In-context influence is harder to evaluate before deployment. Fast LoRA-style fine-tuning is safer because the adapted behavior can be evaluated before deployment.

Case data still matters — it affects behavior through the training pipeline, where it can be validated, calibrated, rolled back, and audited.

### 6.3 Pure End-to-End VLA

A pure vision-language-action model could simplify the stack but lacks the structural commitments this architecture requires: multi-loop control, formal safety boundaries, entity-centric state, auditability, and surgeon-confirmed constraints. Clinical autonomy needs object state and deterministic safety that a monolithic VLA cannot provide.

---

## 7. Off-the-Shelf Technologies

The architecture does not try to invent every layer. It uses existing or mature components where they are adequate:

| Technology | Component | Role |
|---|---|---|
| Hybrid SSM-Transformer (Mamba-2 / Jamba / Hymba / Zamba) | Substrate backbone | Long-context recurrent policy core with linear-time scaling |
| SAM 2 / MedSAM2 / surgical foundation backbones | Semantic stream | Semantic perception, swappable as the field improves |
| SE(3)-equivariant libraries (Equiformer, SE(3)-Transformer) | Action stream | Geometry-first inductive bias for tool/tissue spatial relations |
| MPC, visual servoing, impedance/admittance control | Fast controller | Mature real-time control techniques |
| SOFA-Framework | Simulator | Open-source surgical soft-body simulation |
| MCTS / MPPI / CEM / trajectory optimization | Slow planner search | Directive search over world-model rollouts |
| Conformal prediction | Risk system, OOD detection | Calibrated uncertainty for learned predictions |
| alpha-beta-CROWN / Marabou | Safety filter sub-checks | Formal verification of small neural geometric predicates |
| LoRA | Substrate fine-tuning | Fast surgeon-teaching updates |
| CLIP-style contrastive training | Joint embedding model | Multi-modal alignment across (vision, geometry, kinematics, force, language) |
| Dreamer / TD-MPC2 lineage | World model rollouts | Latent planning over predicted futures |

The rule: use fixed software where the interface is stable and the behavior is mature; use learned representations where the domain is continuous, variable, and data-rich.

---

## 8. Research Bets and Risks

### Primary Research Bets

1. **Compositional policy substrate.** The largest bet. If it works, autonomous behavior generalizes across directive combinations. If it fails, the system degrades toward per-directive policies without the clarity of a skill library.
2. **Joint multimodal embedding model.** The substrate-of-the-substrate. It must align vision, geometry, kinematics, force, and language into one useful entity embedding space.
3. **Unified embedding-conditioned world model.** Needed for planning, rollouts, simulator learning, and per-entity dynamics adaptation. Success in middle phases depends on the deformable-dynamics frontier — predictive accuracy and calibration on soft tissue, contact, and fluid.
4. **Streaming variable-cardinality entity binding.** Needed to maintain stable entity identities in a changing surgical field with dynamic object count.
5. **Adaptive recurrent reasoning module.** Needed for slow planner deliberation and goal decomposition.

### Mitigations

- The safety filter catches unsafe commands regardless of substrate quality.
- Contract monitoring catches behavioral drift.
- OOD detection gates autonomy at the capability frontier.
- Phased deployment limits exposure to the distribution where evaluation shows reliability.
- Surgeon override remains immediate and authoritative.
- Evaluation is behavior-contract-based rather than module-structure-based.

### Residual Risk

The architecture depends on compositional generalization emerging from the substrate's training distribution. That is not guaranteed. The design makes this bet explicit rather than hiding it behind a skill-library interface. The program should continuously measure whether the substrate is expanding its reliability frontier or merely memorizing narrow directives.

---

## 9. Phased Capability Expansion

The architecture expands by growing the substrate's directive distribution and the embedding space coverage, not by adding new code paths. A third axis is explicit: deformable and interaction world-model reliability — how well predicted tissue/fluid/contact evolution matches logged and sim-grounded behavior under calibration.

| Phase | Capability | Key Expansion |
|---|---|---|
| 0 | Infrastructure, data pipeline, simulator wrapper, bootstrap models | Freeze architecture; build everything below the substrate |
| 1 | FLS peg transfer (human-competitive) | Rigid-object manipulation language (pick, place, transfer, move-to-pose) |
| 2 | Suturing in simulation | + Suturing language (drive needle, throw knot, pull through, tie) |
| 3 | Suturing on phantom | Same language coverage, hardware-validated; adapter refinement from real wrench/stereo logs |
| 4 | Resection in simulation | + Soft-tissue interaction language (aspirate, cauterize, suction, retract, manage bleeding); primary ramp on sim-grounded deformable prediction |
| 5 | Cadaveric narrow-task autonomy | World model under real anatomy and registration error |
| 6 | Realistic suturing with bleeding, obstacles, living tissue | Cross-directive composition + material variation + bleeding sources; stress world-model calibration |
| 7+ | Procedural autonomy | Procedure-level language ("remove tumor at left margin," "achieve hemostasis"); substrate composes lower-level behavior internally |

Each phase adds data, simulator coverage, evaluation suites, calibration sets, and fine-tuning. It should not require architectural rewrites.

---

## 10. Final Architectural Position

This architecture is intentionally hybrid:

- It is not pure classical robotics, because fixed symbolic skills and typed scene representations are too brittle for long-horizon surgical autonomy.
- It is not pure end-to-end VLA, because clinical autonomy needs entity state, multi-loop control, formal safety boundaries, auditability, and surgeon-confirmed constraints.
- It is not RAG-driven, because behavior-changing knowledge should enter through evaluated training and fine-tuning rather than unverified inference-time examples.

The proposed system is a structured learned autonomy stack: persistent entity representations with continuous multimodal embeddings, an evolving entity knowledge store with planner-gated consolidation, a compositional policy substrate conditioned on language and contracts, multi-loop control with classical fast feedback, calibrated risk assessment, and a verified safety filter that sits between every command and the robot.

The central claim is that surgical autonomy will scale only if the system can learn continuous structure and compose behavior across contexts, while preserving hard deterministic safety at the robot boundary. This architecture is designed around that claim.
