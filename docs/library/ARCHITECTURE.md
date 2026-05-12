# Autonomous Surgery Architecture

**Status:** v2.5 — engineering spec
**Preserves principles from:** `ARCHITECTURE_THESIS.md` (embedding-first state, multi-loop control, engineered safety boundary)
**Risk and validation plan:** `RISKS_AND_VALIDATION.md`
**Scope:** End-to-end architecture for autonomous surgical robotics on KUKA KR 6-2 (development) and neuroArm (target)

---

## 1. Overview

A learning-centric autonomous surgery stack built around **self-supervised foundation models**, **imitation learning from mono surgical video**, and an **engineered safety boundary**. The system progresses from controlled dexterity benchmarks toward procedural autonomy without architectural rewrites — capability expands by training on more data, not by adding code paths.

The architecture inverts the classical robotics dependency on annotated multimodal sensor data:

- Uses pretrained self-supervised models (DINOv2, V-JEPA 2) as the perception substrate
- Discovers entities via slot attention without labels
- Learns world dynamics from foundation-model world models (DINO-WM) on raw video
- Imitates surgeons via inverse-dynamics pseudo-actions extracted from sim
- Uses SOFA simulation only for inverse-dynamics and forward-model training, not full physics replication
- Stores memory at three timescales — activations, Hopfield fast weights, model slow weights — with no buffer or retrieval database

The architectural separation of **soft learned reasoning** from **hard safety guarantees** is preserved from v0.8.

---

## 2. Operating Constraints

| Constraint | Implication |
|---|---|
| **Mono surgical video only** for the foreseeable future | No multimodal sensor ground truth at training time |
| **No annotation budget** | All training is self-supervised, imitation, or sim-supervised |
| **No robot data yet** | Cannot pretrain on robot-collected multimodal data |
| **Real surgeons, not robots, in source data** | Action labels do not exist; must be inferred via IDM |
| **KUKA KR 6-2 for development** | Different from the final neuroArm target |
| **Clinical safety required** | Hard guarantees cannot depend on learned models alone |

**Consequence:** The architecture must be trainable from raw mono surgical video and pretrained foundation models, with simulation used as a narrow data-generation tool rather than a full physics replica.

---

## 3. Design Philosophy

Six principles, in priority order:

1. **Self-supervised first.** No component requires labeled data as a first-class dependency. Annotation is a refinement, not a prerequisite.
2. **Embedding-first state.** Per-entity state is continuous and addressable, never reduced to discrete labels. Carried forward verbatim from v0.8 §4.
3. **Memory = weights and activations.** No retrieval buffers, no entity stores with fields. Memory lives at three timescales: activations (working memory), fast weights (Hopfield), slow weights (cortical models).
4. **Foundation models as substrate.** Use pretrained checkpoints (DINOv2, V-JEPA 2, DINO-WM) where they exist. Custom training is for surgical adaptation, not from-scratch builds.
5. **Imitation from observation.** Surgical video is the demonstration set. Inverse dynamics extracts pseudo-actions; behavioral cloning learns from them.
6. **Hard safety stays engineered.** Learned components inform but cannot override formally verified safety predicates.
7. **One environment interface.** Sim and real implement the same boundary so code above it runs unchanged on KUKA, neuroArm, or simulation.

---

## 4. System Loops and Data Flow

Five concurrent loops:

| Loop | Rate | Role |
|---|---|---|
| Surgeon interface | event-driven | Goals, permissions, tags, corrections, overrides |
| Slow planner | 0.5–2 Hz | Directives, behavior contracts, memory consolidation, escalation |
| Policy substrate | 5–20 Hz | Compositional behavior generation, OOD/confidence checks |
| Fast controller | 100–500 Hz | MPC, visual servoing, force-aware tracking |
| Safety evaluator | 10–20 Hz async + 100–500 Hz sync gate | Two-phase: assessment publishes safety surface; sync gate vetoes commands |

### Data flow

```
Mono surgical video / robot stereo
        │
        ▼
[DINOv2 + V-JEPA 2 backbones]      ← perception (frozen / fine-tuned)
        │
        ▼
[Slot attention with temporal recurrence]      ← entity binding (self-supervised)
        │
        ▼
[Active entity state]              ← variable-K embedding bundles in working memory
        │
        ├──→ [Hopfield fast-weight memory]     ← case-long associative recall
        │
        ▼
[Slow planner]   ⇄   [World model: DINO-WM]    ← search over predicted futures
        │
        ▼
[Policy substrate (V-JEPA 2 bootstrap)]
        │
        ▼
[Fast controller] ─────→ [Sync gate] ────→ Robot
        ▲                       ▲
        │                       │
[Fast forward model]            │
(physical implausibility) ──────┘
```

The **sync command gate** sits between every command and the robot, regardless of source.

---

## 5. Foundation Model Stack

Each component below is a pretrained checkpoint that can be fine-tuned on surgical video without labels.

### 5.1 Critical path

| Foundation Model | Provenance | Role | Training Cost |
|---|---|---|---|
| **DINOv2** | Meta, 2023 | Per-patch visual features (identity-coded) | Fine-tune on surgical video |
| **V-JEPA 2** | Meta, 2025 | Self-supervised video features, robot transfer head bootstraps the policy | Fine-tune on surgical video |
| **DINO-WM** | NYU + Meta, 2025 | World model on DINOv2 features for planner search | Fine-tune on surgical transitions |
| **Modern Hopfield** | Ramsauer et al., 2020 | Associative memory for case-long episodes | No gradient training; inference-time updates |
| **Slot attention (DINOSAUR-style)** | Pattern, not single checkpoint | Self-supervised entity decomposition over DINOv2 features with SAVi-style temporal recurrence | Train from scratch on surgical video |

### 5.2 Deferred / optional

These are **not on the critical path** for v0. They become useful once specific input requirements emerge.

| Foundation Model | Becomes useful when |
|---|---|
| **VL-JEPA** (Meta + collaborators, 2026) | Surgeon language directives become a primary input; transcripts or operative reports are aligned to the embedding space |
| **SAM 2** (Meta, 2024) | Explicit masks are needed for surgeon UI, mask-conditioned tasks, or auxiliary fine-tuning of the slot decoder |

These are excluded from the critical path because they add training and integration surface without serving a v0 requirement. They can be added later without architectural change.

---

## 6. Component Architecture

### 6.1 Perception

**Two streams sharing foundation backbones:**

- **Action stream.** DINOv2 features projected through SE(3)-equivariant layers (Equiformer or SE(3)-Transformer) for geometric grounding. Pretrain SE(3) projection on open robotics data (BRIDGE, RoboNet, OpenVLA), then fine-tune via V-JEPA 2 self-supervised prediction on surgical video.
- **Semantic stream.** V-JEPA 2 video features fine-tuned with masked feature prediction on surgical video for surgical-anatomy specificity.

Both streams output features consumed by the entity binding stage. No labels required for either stream.

### 6.2 Entity Binding and Memory

This section replaces v2.0's "entity knowledge store" with a three-timescale memory architecture. The v0.8 §4 embedding-first principle is preserved end-to-end.

#### 6.2.1 Active entity state (working memory)

Variable-cardinality slot attention runs on top of frozen-or-finetuned backbone features with SAVi-style temporal recurrence. Trained with a self-supervised reconstruction objective + temporal consistency:

```
backbone features → K slots → reconstruct backbone features
                              + temporal_consistency(slot_t, slot_{t+1})
```

Slots emerge as object-like representations (tools, tissue regions, blood pools, anatomy) without labels. A tracker matches slots to persistent active-entity records by embedding similarity and spatial prior.

**Per-entity record — embeddings + bookkeeping only, no labels.** This restores v0.8 §4: identity is *position in continuous embedding space*, not a stored class.

| Field | Type | Notes |
|---|---|---|
| `observation_emb` | continuous vector | Latest projected slot vector |
| `state_emb` | continuous vector | Running fusion via an update network |
| `pose` | continuous SE(3) | Geometry, not a label |
| `observation_confidence` | continuous scalar | Replaces any "observed: bool" flag |
| `id` | opaque identifier | Tracker bookkeeping, not semantic |
| `last_seen_t` | timestamp | Bookkeeping |

There is no `semantic_class`, no `is_vessel`, no `tool_type`. Class membership is a *similarity query* against a continuous embedding manifold, computed when needed.

#### 6.2.2 Hopfield-based fast-weight memory (case-long)

A modern Hopfield network (Ramsauer et al., 2020 line of work) plays the role of a hippocampal-style episodic memory. Storage is by outer-product updates at inference time; recall is by attention-style energy minimization with continuous queries. Capacity is exponential in dimension; pattern completion is natural.

The Hopfield network sits behind a small encoder/decoder bridge that maps active-entity activations into Hopfield keys/values and reads back via pattern completion to feed the planner.

This replaces any notion of an "episodic buffer" or retrieval database. Querying is by similarity / pattern completion, not indexed lookup. Within a case, the Hopfield accumulates episode-level associations. Across cases, those associations consolidate into the slow-weight components via offline replay (see §7.3).

#### 6.2.3 Slow-weight memory (cross-case)

The world model and planner backbones accumulate long-term knowledge in their weights via supervised fine-tuning passes between cases (see §7.3). There is no separate "long-term memory store" — the slow weights of these networks *are* the long-term memory.

#### 6.2.4 Three timescales summary

| Timescale | Substrate | Mechanism |
|---|---|---|
| Activations (frame–seconds) | Active entity records, recurrent state | Forward computation, no weight updates |
| Fast weights (case-long) | Modern Hopfield network | Online outer-product updates per frame; pattern completion |
| Slow weights (cross-case) | World model + planner network weights | Self-supervised + offline consolidation from fast weights |

This gives the system **object permanence** (working-memory tracker + Hopfield pattern completion under occlusion) and **long-term time** (cortical slow weights as accumulated dynamics priors), without any retrieval buffer or entity-fields database.

### 6.3 World Model (DINO-WM)

DINO-WM predicts future patch features given action sequences, trained offline on surgical video transitions. Used by the planner for goal-conditioned action search (MCTS / MPPI / CEM over predicted futures).

The original v0.8 plan called for SOFA physics + learned residual + per-entity adapters. v2.5 keeps DINO-WM as the world model because foundation-model world models learn physics implicitly from video. SOFA's role narrows correspondingly (see §9).

### 6.4 Fast Forward Model

A small fast network that predicts the immediate sensory consequence of motor commands:

```
forward_model(sensory_t, motor_command_t) → predicted_sensory_{t+1}
```

Trained in SOFA via supervised next-state prediction. Loss: MSE on actual `sensory_{t+1}`. This is the only place the architecture uses a "supervised by reality" signal — and the supervision is free.

**Two consumers:**

- **Fast controller** uses forward predictions for short-horizon action refinement.
- **Safety gate** uses prediction-error magnitude as a physical implausibility flag (high error → command produced an unexpected outcome → veto).

This replaces v2.0's LeWM with the same role and a clearer training story (supervised in sim with prediction error, rather than self-supervised pixel JEPA). The component is structurally what v2.0 §6.4 described as the "fast pixel WM"; the rename and retrain reflect a more honest specification of what it actually does.

### 6.5 Policy Substrate

Hybrid SSM-Transformer (Mamba-2 / Jamba lineage) bootstrapped from V-JEPA 2's robot transfer head. Trained via behavioral cloning on `(active entity state, language directive, behavior contract) → pseudo-action` pairs.

**Where pseudo-actions come from:** Inverse Dynamics Model (IDM) trained in SOFA simulation, applied to surgical video to extract per-frame pseudo-actions. This is the VPT-style trick that bridges the no-action-label gap (see §7.2).

When real robot data eventually exists, pseudo-actions are replaced by real actions and the substrate is fine-tuned. The architecture does not change.

### 6.6 Slow Planner

Recurrent reasoning module (Loop Transformer / Universal Transformer / HRN-class with adaptive halting) plus search over world-model rollouts (MCTS / MPPI / CEM). Inputs:

- Surgeon goals + tags
- Active entity state
- Hopfield-recalled case episodes (via the entorhinal-style bridge)
- Safety surface from the safety evaluator
- World model predictions

Outputs:

- Natural-language directives + behavior contracts (hard + soft)
- Modulation signals (caution, attention targets, time pressure)
- **Memory consolidation routing decisions:** which experiences become Hopfield fast-weight updates, which queue for offline slow-weight consolidation, which trigger surgeon escalation
- Escalation triggers

The planner is the *router* across memory substrates, not a writer to a single store.

### 6.7 Fast Controller

Classical, not learned. MPC + visual servoing + impedance/admittance control. Tracks short-horizon target trajectories from the policy substrate at 100–500 Hz. Takes geometric features directly from the action stream for low-latency servoing. Reads forward-model predictions from §6.4 to refine action choice.

### 6.8 Safety Evaluator (Two-Phase)

Engineered, not learned. Inherited structurally from v0.8 §4.10:

- **Layer 1 (sync, 100–500 Hz):** Formally verified physical invariants — joint limits, workspace bounds, force/velocity limits, singularity avoidance, communication watchdog. Plus geometric intersection tests against the safety surface. Plus **forward-model physical implausibility flags** (§6.4) as a free veto signal. <2K LOC, formally verifiable.
- **Layer 2, async assessment (10–20 Hz):** Reads active entity state, world model deformations, surgeon tags. Computes deformation-aware no-go geometry, context-aware force envelopes, per-entity risk scores via conformal prediction. Publishes the safety surface consumed by the sync gate.

Safety training data and pipeline remain separate from policy training. The policy cannot influence what counts as safe.

---

## 7. Training Strategy

### 7.1 Tier structure

| Tier | Components | Method | Data |
|---|---|---|---|
| **0: Foundation pretraining** | DINOv2, V-JEPA 2 | Use existing checkpoints | None (already trained) |
| **1: Surgical adaptation** | Slot attention, V-JEPA 2 fine-tune | Self-supervised | Mono surgical video |
| **2: World model + IDM + forward model** | DINO-WM, IDM, fast forward model | Self-supervised (DINO-WM) + supervised in sim (IDM, forward model) | Surgical video + SOFA sim |
| **3: Policy** | Policy substrate | Imitation learning on pseudo-actions | Surgical video × IDM pseudo-actions |
| **4: Memory** | Hopfield + bridge encoders, consolidation | Self-supervised + offline supervised consolidation | Surgical video, then real cases as collected |

**No supervised perception annotation. No multimodal sensor ground truth required.**

### 7.2 The Inverse Dynamics Bridge

The key technique that breaks the chicken-egg problem of having no robot action labels:

```
Step 1: Train IDM in SOFA sim
        IDM(frame_t, frame_t+1) → action_t   (small dataset, supervised)

Step 2: Apply IDM to surgical video
        Per-frame transition → pseudo-action

Step 3: Train policy via behavioral cloning
        (entity state + directive) → pseudo-action

Step 4: When real robot data exists, replace pseudo-actions with real actions
```

This is the same approach OpenAI used for VPT (YouTube Minecraft → SOTA Minecraft agent). It works for surgery because the **action representation** (delta tool pose, gripper state) is at the right level of abstraction to transfer from sim to real video.

**This is the load-bearing bet of the architecture.** See `RISKS_AND_VALIDATION.md` for the validation plan that should run before downstream work commits to it.

### 7.3 Memory Consolidation Roadmap

Memory consolidation from fast weights (Hopfield) into slow weights (model parameters) is added incrementally:

| Phase | Consolidation mechanism |
|---|---|
| **v0** | None. Pure imitation + self-supervised pretraining. Each case is independent. |
| **v1** | Hopfield active at inference, no cross-case consolidation. Episodes lost between cases. |
| **v2** | Offline supervised consolidation: replay Hopfield contents into slow-weight networks via standard fine-tuning between cases. |
| **v3** | Generative replay (Dreamer-style): planner's world-model rollouts retrained against by the policy. RL with dreaming. |

The architecture supports the full progression. **Only v0 is required for initial deployment.**

### 7.4 Refinement Path

1. **Initial deployment:** Self-supervised + imitation models from above
2. **Robot data collection:** Real KUKA teleop logs multimodal data
3. **Fine-tune all components** with real measurements replacing pseudo-labels
4. **RL refinement** in SOFA for behaviors imitation cannot reach
5. **DAgger** for surgeon corrections during real cases

The pipeline is monotone — never less data, always more grounded.

---

## 8. Robot Interface

### 8.1 Environment Protocol

The architecture's sim/real parity boundary:

```python
class Environment(Protocol):
    def reset(self, config: EnvConfig) -> SceneGraph: ...
    def step(self, action: RobotCommand) -> StepResult: ...
    def get_sensors(self) -> SensorBundle: ...
    def get_scene(self) -> SceneGraph: ...
```

Implemented by:

- `SimEnvironment` (SOFA-backed, primarily for IDM + forward-model training and evaluation)
- `KukaRealEnvironment` (KUKA KR 6-2 hardware, dev platform)
- `NeuroarmRealEnvironment` (neuroArm hardware, target platform)

Code above this boundary runs unchanged across all three.

### 8.2 Schema Design

Drawing on the reference Rust SDK (`reference_kuka/`) but standardizing inconsistencies:

**Standardized shapes:**

- `Vec3 { x: f64, y: f64, z: f64 }` — always for 3D positions, forces, velocities
- `Angle<Unit>` — angles always carry their unit (Radians or Degrees) at the type level
- `Pose { position: Vec3, rotation: Quaternion }` — orientations as quaternions internally, converted at hardware boundaries

**Control modes** (extension over the reference):

```python
class ControlMode(Enum):
    JOINT_VELOCITY = "joint_velocity"      # KUKA-style
    JOINT_POSITION = "joint_position"
    CARTESIAN_POSE = "cartesian_pose"      # neuroArm-style
    CARTESIAN_FORCE = "cartesian_force"    # contact tasks
```

**`RobotCommand`** carries `control_mode` plus the appropriate payload, enabling robot-agnostic policy code.

**`StepResult`** carries observations sized by capability:

- `joint_state` (always)
- `tool_pose` (always, derived if needed)
- `forces` (Vec3, present when sensor available)
- `stereo_frames` (when stereo cameras present)
- `mono_frame` (always)

### 8.3 KUKA KR 6-2 → neuroArm

KUKA KR 6-2 is the development platform. neuroArm is the target. The migration path:

1. KUKA bridge built first (Simulink wrapper or direct KSS interface)
2. Policy trained against KUKA in real and SOFA sim
3. neuroArm bridge added by implementing `Environment` for neuroArm
4. Policy fine-tuned for neuroArm-specific control mode (likely Cartesian pose, not joint velocity)

The architecture above the `Environment` boundary does not change between robots.

---

## 9. Simulation Strategy

### 9.1 SOFA's Reduced Role

In v0.8, SOFA was the world model's physics prior. In v2.5, the foundation-model world model (DINO-WM) replaces that role. SOFA remains for four narrower purposes:

1. **Inverse Dynamics Model training data.** SOFA provides ground-truth action labels paired with rendered video frames. The IDM trains here.
2. **Fast forward model training data.** SOFA provides paired `(sensory_t, motor_t, sensory_{t+1})` triples for next-state prediction supervision.
3. **RL refinement environment.** Once an imitation policy exists, SOFA provides a safe environment to refine it via RL with contract-based rewards.
4. **Evaluation harness.** Stress tests, edge cases, safety probes, deformation regression tests.

SOFA does **not** need to perfectly replicate surgical tissue dynamics, because the deployed world model is learned from real surgical video — not from SOFA. The sim-to-real gap on tissue dynamics matters less than in v0.8.

### 9.2 Domain Randomization

Applied to SOFA scenes when generating IDM, forward-model, and RL refinement scenarios:

- Tissue material properties (compliance, friction)
- Lighting and camera intrinsics (matching real surgical cameras)
- Tool geometry variations
- Anatomy variants (when MRI distributions are available)

Domain randomization protects against IDM and forward-model overfitting to a narrow visual or dynamic regime.

---

## 10. Safety Architecture

Inherited from v0.8 with one substantive addition: **forward-model physical implausibility detection** as a Layer 1 sync-gate signal (see §6.4).

| Layer | Phase | Type | Source of truth |
|---|---|---|---|
| 1 | Sync gate (100–500 Hz) | Formally verified | Hardware limits, geometric intersection, forward-model implausibility |
| 2 | Async assessment (10–20 Hz) | Calibrated learned | Active entity state, world model deformations, surgeon tags |
| 2 | Sync command gate phase 2 | Formally verified against Layer 2 surface | Pre-computed safety surface from async assessment |

The safety evaluator is **substrate-output-agnostic**: it does not care whether a command came from policy, planner, sim, or surgeon teleoperation. Every command is gated.

Safety head training remains separate from policy training. Conformal prediction calibrates uncertainty in async assessment.

---

## 11. Development Phases

| Phase | Capability | Foundation Model Adaptations | Robot |
|---|---|---|---|
| **0** | Infrastructure: Environment interface, schema, foundation model checkpoints | All foundation models loaded, frozen | None |
| **1** | Self-supervised perception + entity binding on surgical video | Slot attention + V-JEPA 2 fine-tune | None |
| **2** | World model + IDM + fast forward model | DINO-WM trained, IDM in SOFA, forward model in SOFA | KUKA in sim |
| **3** | Imitation policy on surgical video pseudo-actions | Substrate trained via behavioral cloning | KUKA in sim |
| **4** | KUKA hardware bring-up, real-data fine-tuning | All components fine-tuned with real KUKA observations | KUKA real |
| **5** | Hopfield activated at inference (no consolidation yet) | Encoder/decoder bridge trained for the Hopfield interface | KUKA real |
| **6** | Suturing on phantom; offline consolidation between cases | Replay → slow-weight fine-tuning | KUKA real |
| **7** | Cadaveric narrow-task autonomy | World model under real anatomy + registration error | KUKA real |
| **8** | neuroArm port | `NeuroarmRealEnvironment` implementation, control mode adaptation | neuroArm |
| **9+** | Procedural autonomy + RL refinement (Dreamer-style) | Substrate composes lower-level behavior internally | neuroArm |

Each phase adds data, training, and evaluation. None requires architectural rewrites.

**Validation gates between phases are specified in `RISKS_AND_VALIDATION.md`.** Phase 0 is gated on validation of the load-bearing IDM transfer bet.

---

## 12. Rejected Patterns

Carried over from v0.8 (still valid):

- **Fixed skill library** as the autonomy center — does not scale with surgical variability
- **Inference-time RAG** — behavior-changing knowledge enters via training, not retrieval
- **Pure end-to-end VLA** — lacks entity state, multi-loop control, formal safety boundaries

Carried over from v2.0 (still valid):

- **From-scratch training of perception or world models** — foundation models exist, use them
- **Annotation-heavy supervised pretraining** — self-supervised + imitation is sufficient and cheaper
- **SOFA as the world model** — pretrained video world models generalize better than physics priors

New rejections in v2.5:

- **Entity store with discrete fields** (`semantic_class`, `is_vessel`, `tool_type`, etc.) — violates the v0.8 §4 embedding-first principle. Identity is a region in continuous embedding space; class membership is a similarity query.
- **Episodic memory as an append-only buffer or retrieval database** — fast-weight memory belongs in a network with high plasticity (modern Hopfield), not in a database. Recall is by pattern completion, not indexed lookup.
- **VL-JEPA and SAM 2 on the critical path before they earn it** — defer until a specific input requirement makes them load-bearing.

---

## 13. Open Questions

The most important risks and open questions are tracked in `RISKS_AND_VALIDATION.md`. Summarized here:

1. **IDM transfer accuracy.** The load-bearing bet. Validate before downstream work commits.
2. **Foundation model surgical specificity.** Quantify how much surgical-domain fine-tuning is needed before features are useful for slot attention and world modeling.
3. **Pseudo-action quality at fine temporal scale.** What temporal smoothing makes behavioral cloning stable.
4. **Forward-model sim-to-real gap.** Does sim-only forward-model training give clinically useful implausibility signals on real surgery, or does it pollute safety with false positives?
5. **Hopfield capacity at case length.** Empirical validation needed; surgeries are long.
6. **KUKA control mode mapping.** Affects the `ControlMode` selected during dev.
7. **neuroArm integration depth.** What does the `NeuroarmRealEnvironment` need to expose to match the KUKA-trained policy's expectations?
8. **Reward signal beyond imitation.** The architecture has no path beyond expert demonstrations until Phase 9. The plan for closing this gap needs spec.
9. **Evaluation framework.** Metrics, gates, and acceptance criteria for each phase.
10. **Data contract.** Hours of surgical video available, procedures covered, quality, consent — required to size training plans.

---

## 14. Final Position

v2.5 is a hybrid system:

- **Self-supervised foundation models** (DINOv2, V-JEPA 2, DINO-WM) for perception, entity discovery, and world modeling
- **Three-timescale memory** — activations (working memory), Hopfield fast weights (case-long), model slow weights (cross-case)
- **Imitation from surgical video** via inverse-dynamics-extracted pseudo-actions
- **Fast forward model** for sub-millisecond control refinement and physical implausibility detection
- **Engineered safety evaluator** with formal verification at the sync gate
- **One environment interface** spanning sim, KUKA, and neuroArm

The system is built on the bet that 2025–2026 self-supervised foundation models, plus IDM-bridged imitation, are strong enough that surgical autonomy can be bootstrapped from mono surgical video without annotation, supervised perception labels, or multimodal sensor ground truth. The architecture remains testable, auditable, and safety-bounded throughout that bet.

**The bet has known risks.** They are catalogued in `RISKS_AND_VALIDATION.md` with concrete first-month experiments to validate or refute the load-bearing assumptions before downstream work commits.

**Trainable today. Scalable as data accumulates. Robot-agnostic at the interface boundary.**
