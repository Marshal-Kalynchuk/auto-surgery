# Brain-Aligned Architecture (Research Vision)

**Status:** Research vision — not the engineering spec
**Canonical spec:** `../library/ARCHITECTURE.md`
**Relationship:** This document captures a brain-region-aligned reorganization of the canonical architecture. It is preserved as a long-term research direction, not as the current implementation plan. Several useful corrections from this doc (three-timescale memory, embedding-first entity record, Hopfield as fast-weight memory, fast forward model framing) were back-ported into the canonical spec; the brain-region renaming was not, because it does not change implementation and oversells the alignment claim.
**Preserves principles from:** `../library/ARCHITECTURE.md` v0.8 §4 (embedding-first state representation)

---

## 1. Overview

The architecture is organized around **brain regions defined by connectivity**, not around foundation-model boxes. One cortical substrate type (transformer modules trained with predictive coding), specialized into regions by what each module reads from and writes to. Long-term memory and the world model are the same substrate at different timescales. A modern Hopfield network plays the role of the hippocampus. A small fast forward model plays the role of the cerebellum. Engineered safety predicates play the role of brain-stem reflexes.

What v3 keeps from v2.0:

- Self-supervised + imitation learning as the only training paradigms
- IDM-extracted pseudo-actions from SOFA → behavioral cloning on surgical video
- Mono surgical video as the primary data source
- The `Environment` interface as the sim/real parity boundary
- Engineered safety gate, formally verified, separate from learned components

What v3 changes from v2.0:

- No "entity store" with discrete fields. Active scene state lives in association-cortex activations; longer memory in hippocampal fast weights and cortical slow weights.
- No separate "world model module." The world-model function is a head on the cortex, sharing weights with the planner.
- Two perception streams (ventral + dorsal), explicitly mapped to cortical regions.
- LeWM replaced by a cerebellar-style forward model with the same role.
- VL-JEPA and SAM 2 deferred from the critical path until justified.
- DINOSAUR slot attention is a readout from association cortex, not a separate module.

---

## 2. Operating Constraints

Unchanged from v2.0:

| Constraint | Implication |
|---|---|
| **Mono surgical video only** for the foreseeable future | No multimodal sensor ground truth at training time |
| **No annotation budget** | All training is self-supervised, imitation, or sim-supervised |
| **No robot data yet** | Cannot pretrain on robot-collected multimodal data |
| **Real surgeons, not robots, in source data** | Action labels do not exist; must be inferred via IDM |
| **KUKA KR 6-2 for development** | Different from final neuroArm target |
| **Clinical safety required** | Hard guarantees cannot depend on learned models alone |

---

## 3. Design Philosophy

Six principles, in priority order:

1. **Brain-aligned organization.** Regions are defined by connectivity over a shared substrate type, not by being separate kinds of network.
2. **Self-supervised everywhere.** Predictive coding is the universal loss for cortical regions. The cerebellum's "supervised by reality" signal (next-frame prediction error) is the only non-self-supervised signal.
3. **Memory = weights.** No retrieval buffers, no entity stores with fields. Memory lives at three timescales: activations (working memory), fast weights (hippocampus), slow weights (cortex).
4. **Embedding-first state.** Per-entity state is continuous and addressable, never reduced to discrete labels. Carried forward verbatim from v0.8 §4.
5. **Hard safety stays engineered.** Learned components inform but cannot override formally verified safety predicates.
6. **One environment interface.** Sim and real implement the same boundary so code above it runs unchanged on KUKA, neuroArm, or simulation.

---

## 4. Brain-Region Mapping

The architecture is a connectivity graph of specialized modules. All cortical modules share the same architecture type (a transformer block with predictive coding loss); they differ only in connectivity and consequent specialization.

| Brain analog | Architecture component | Substrate type | Connectivity role |
|---|---|---|---|
| Ventral visual stream | Identity perception encoder | Pretrained transformer (DINOv2-class), surgically fine-tuned | Feeds association cortex (identity-coded features) |
| Dorsal visual stream | Spatial perception encoder | Pretrained transformer (V-JEPA 2-class) + SE(3)-equivariant projection | Feeds association cortex (spatial/action-coded features) |
| Association cortex (parietal-like) | Working memory + entity binding | Cortical module | Reads both streams + tracker; writes to PFC, entorhinal |
| Entorhinal interface | Encode/decode bridge to hippocampus | Cortical module | Bidirectional with hippocampus |
| Hippocampus | Fast-weight associative memory | Modern Hopfield | Bidirectional with entorhinal |
| Prefrontal cortex (PFC) | Planning, working memory, goal management | Cortical module | Reads association + entorhinal; writes to premotor |
| Premotor / SMA | Action sequencing | Cortical module | Reads PFC + association; writes to motor cortex |
| Motor cortex | Action distribution | Cortical module | Reads premotor + cerebellum; writes to safety gate |
| Cerebellum | Forward motor model | Small fast network with prediction-error loss | Reads motor efference + sensory; writes to motor cortex + safety gate |
| Brain-stem reflexes | Engineered safety gate | Formally verified rule set | Reads sensory + cerebellar; gates motor output |

Cross-attention is the wiring between cortical modules. Each module has its own weights but shares the same architecture and learning rule.

---

## 5. Connectivity Diagram

```
    Ventral stream ─┐
                    ├─→ Association cortex ──→ Entorhinal ⇄ Hippocampus
    Dorsal stream  ─┘          │                              (Hopfield,
                               │                               fast weights)
                               ▼
                              PFC ─────────→ Premotor ──→ Motor cortex
                               ▲                              │
                               │                              ▼
                               │                          Cerebellum
                               │                          (forward model)
                               │                              │
                               │ (replay during              ▼
                               │  consolidation)         Safety gate
                               │                              │
                               └──────────────────────────────┘
                                   (efference + reflex
                                    inputs back into PFC)
```

A surgical case at runtime is feedforward + recurrent flow through this graph. Consolidation (offline, post-case) is replay back through the same paths.

---

## 6. Memory at Three Timescales

All implemented as model weights or activations. No buffers.

| Timescale | Substrate | Mechanism | Brain analog |
|---|---|---|---|
| Activations (frame–seconds) | Recurrent state in cortical modules + tracker | Forward computation, no weight updates | Working memory (parietal) |
| Fast weights (case-long) | Modern Hopfield network | Online outer-product updates per frame; pattern completion for retrieval | Hippocampus |
| Slow weights (cross-case) | Cortical modules | Predictive coding + offline consolidation from fast weights (eventually generative replay) | Cortex |

Three properties this gives the system:

- **Object permanence** is two-sided: working-memory activations track entities across visible frames; hippocampal pattern completion recovers them after long occlusion or interruption.
- **Long-term time** lives in the cortical weights as accumulated dynamics priors, refined by replay from hippocampal fast weights.
- **Querying** the hippocampus is by similarity / pattern completion, not by indexed lookup. The entorhinal interface provides the encode/decode bridge.

---

## 7. Component Architecture

### 7.1 Perception Streams

Two parallel encoders, both self-supervised, both fine-tuned on surgical video.

**Ventral (identity) stream.** DINOv2-class pretrained transformer fine-tuned via masked feature prediction on surgical video. Output: per-patch identity-coded features (what is this).

**Dorsal (spatial) stream.** V-JEPA 2-class video transformer + SE(3)-equivariant projection layer (Equiformer or SE(3)-Transformer). Output: spatial/action-coded features (where, how, under what dynamics).

Both streams output features that feed association cortex. No labels required for either stream.

### 7.2 Association Cortex

Cortical module. Receives both perception streams. Maintains:

- **Active entity state** as variable-K continuous embedding bundles (slot-attention readout over fused stream features, with SAVi-style temporal recurrence).
- **Entity tracker** that matches new observations to existing active entities by embedding similarity and spatial prior.
- **Occlusion handling** by holding entity activations alive while unobserved, with predicted state from local recurrent dynamics.

Per-entity record (preserved from v0.8 §4 — embeddings + bookkeeping only, no labels):

| Field | Type | Notes |
|---|---|---|
| `observation_emb` | continuous vector | Latest projected slot/feature bundle |
| `state_emb` | continuous vector | Running fusion via the module's recurrence |
| `pose` | continuous SE(3) | Geometry, not a label |
| `observation_confidence` | continuous scalar | Replaces any "observed: bool" flag |
| `id` | opaque identifier | Tracker bookkeeping, not semantic |
| `last_seen_t` | timestamp | Bookkeeping |

Semantic identity (forceps vs. cottonoid vs. vessel) is a *region in continuous embedding space*, not a stored class. Queries by similarity, not field lookup.

### 7.3 Entorhinal Interface and Hippocampus

**Entorhinal interface** is a small cortical module that maps association activations into Hopfield keys/values, and reads back via pattern completion to feed PFC.

**Hippocampus** is a modern Hopfield network (Ramsauer et al., 2020 line of work). Storage is by outer-product updates at inference time; recall is by attention-style energy minimization with continuous queries. Capacity is exponential in dimension, naturally pattern-completes from partial cues.

Within a case, the hippocampus accumulates episode-level associations. Across cases, those associations are consolidated into cortical slow weights via offline replay (see §8.3).

### 7.4 Prefrontal Cortex (PFC)

Cortical module. Inputs from association cortex (current scene) and entorhinal (relevant case episodes). Outputs:

- Behavior contracts (hard + soft constraints, objective)
- Modulation signals (caution, attention focus, time pressure)
- Search-derived directive plans (via rollouts, see §7.4.1)

The PFC's predictive-coding head **is** the world model. Future-state prediction is a readout from the same module, sharing weights with the planning function. There is no separate "world model network."

#### 7.4.1 Search

PFC runs MCTS / MPPI / CEM over its own predictive-coding head's rollouts. Each rollout is `(association state, candidate action sequence) → predicted future association state`. Search budget is bounded by the planner's 0.5–2 Hz rate.

### 7.5 Premotor / SMA and Motor Cortex

Two cortical modules in series.

**Premotor / SMA.** Reads PFC directives + association state. Sequences actions over horizon ~1 s. Outputs intermediate motor goals.

**Motor cortex.** Reads premotor goals + cerebellar feedback. Outputs the action distribution sent to the safety gate. Trained via behavioral cloning on IDM pseudo-actions (see §8.2).

Together these implement the policy substrate. Splitting them into two modules matches biological organization and gives the cerebellar feedback path a clean entry point.

### 7.6 Cerebellum

Small fast forward model. Architecture: a few-layer transformer or MLP, *not* a cortical module. Trained in SOFA via supervised next-state prediction:

```
cerebellum(sensory_t, motor_command_t) → predicted_sensory_{t+1}
```

Loss: MSE on actual `sensory_{t+1}`. This is the only place the architecture uses a "supervised by reality" signal — and the supervision is free.

Two consumers:

- **Motor cortex** reads cerebellar predictions to refine action choice (efference copy + forward model).
- **Safety gate** reads cerebellar implausibility signals (large prediction errors → physical implausibility flag).

Cerebellar training in v0 is sim-only (SOFA with domain randomization). Real-data fine-tuning is added in phase 4.

### 7.7 Safety Gate (Brain-Stem Reflexes)

Engineered, not learned. Two-phase, identical in structure to v2.0 §10:

| Layer | Phase | Type | Source of truth |
|---|---|---|---|
| 1 | Sync gate (100–500 Hz) | Formally verified | Hardware limits, geometric intersection, cerebellar implausibility |
| 2 | Async assessment (10–20 Hz) | Calibrated learned | Association state, PFC deformation predictions, surgeon tags |
| 2 | Sync gate phase 2 | Formally verified against the layer-2 surface | Pre-computed safety surface from async assessment |

The safety gate is **substrate-output-agnostic**: it does not care whether a command came from the policy, the planner, simulation, or surgeon teleoperation. Every command is gated.

---

## 8. Training Strategy

### 8.1 Universal Loss

All cortical modules share one self-supervised loss family: **predictive coding** in feature space (V-JEPA-style). For module `M` with input distribution `x ~ X_M` and context `c`, the loss is:

```
L_M = || predictor_M(c) - encoder_M(x) ||_loss_metric
```

Region-specific behavior emerges from connectivity (which `x`, which `c`), not from region-specific losses.

The cerebellum is the only exception: it uses `MSE(predicted_sensory, actual_sensory)` because next-frame supervision is free.

The policy substrate (premotor + motor cortex) is additionally trained via **behavioral cloning** on IDM pseudo-actions. This is on top of, not instead of, the predictive-coding loss.

### 8.2 The Inverse Dynamics Bridge

Unchanged from v2.0 §7.2. The technique that breaks the chicken-egg problem of having no robot action labels:

```
Step 1: Train IDM in SOFA simulation
        IDM(frame_t, frame_{t+1}) → action_t   (small dataset, supervised)

Step 2: Apply IDM to surgical video
        Per-frame transition → pseudo-action

Step 3: Train motor cortex via behavioral cloning
        association_state + directive → pseudo-action

Step 4: When real robot data exists, replace pseudo-actions with real actions
```

VPT-style. Works because the action representation (delta tool pose, gripper state) transfers from sim to real video at the right level of abstraction.

### 8.3 Consolidation Roadmap

Memory consolidation from fast weights (hippocampus) into slow weights (cortex) is added incrementally:

| Phase | Consolidation mechanism |
|---|---|
| **v0** | None. Pure imitation + predictive coding. Each case is independent. |
| **v1** | Hippocampal Hopfield active at inference, no cross-case consolidation. Episodes lost between cases. |
| **v2** | Offline supervised consolidation: replay hippocampal contents into cortical modules via standard fine-tuning between cases. |
| **v3** | Generative replay (Dreamer-style): PFC's predictive-coding head generates rollouts; motor cortex trains on imagined trajectories with reward signals. RL with dreaming. |

The architecture supports the full progression. **Only v0 is required for initial deployment.**

### 8.4 Data and Tier Structure

| Tier | Components trained | Method | Data |
|---|---|---|---|
| **0: Backbone pretraining** | Ventral + dorsal stream encoders | Use existing self-supervised checkpoints | None (already trained) |
| **1: Surgical adaptation** | Stream encoders (fine-tune) | Predictive coding | Mono surgical video |
| **2: Cortical training** | Association cortex, entorhinal, PFC, premotor, motor cortex | Predictive coding (universal loss) | Mono surgical video |
| **3: Cerebellum + IDM** | Cerebellum, IDM | Supervised in SOFA | SOFA renders with action labels |
| **4: Policy** | Motor cortex (behavioral cloning head) | BC on IDM pseudo-actions | Surgical video × IDM pseudo-actions |
| **5: Hippocampus** | Modern Hopfield (no gradient training); entorhinal interface (predictive coding) | Pattern completion at inference; predictive coding for the bridge | Surgical video |

No supervised perception annotation. No multimodal sensor ground truth. Cerebellum uses sim-only supervision (free). Policy uses pseudo-actions (free).

---

## 9. System Loops and Data Flow

| Loop | Rate | Module(s) |
|---|---|---|
| Surgeon interface | event-driven | External — issues goals, tags, corrections to PFC |
| PFC planning | 0.5–2 Hz | PFC search over its own rollouts |
| Motor pathway (premotor → motor cortex → safety gate) | 5–20 Hz | Reactive policy substrate |
| Fast controller | 100–500 Hz | Classical (MPC + visual servoing + impedance/admittance) |
| Cerebellar forward model | 100–500 Hz | Predictive feedback into motor cortex + safety gate |
| Safety evaluator | 10–20 Hz async + 100–500 Hz sync gate | Two-phase, every command gated |

The fast controller is classical, not learned. It tracks short-horizon target trajectories from the motor cortex output. It takes geometric features directly from the dorsal stream for low-latency servoing.

---

## 10. Robot Interface

Unchanged from v2.0.

### 10.1 Environment Protocol

```python
class Environment(Protocol):
    def reset(self, config: EnvConfig) -> SceneGraph: ...
    def step(self, action: RobotCommand) -> StepResult: ...
    def get_sensors(self) -> SensorBundle: ...
    def get_scene(self) -> SceneGraph: ...
```

Implemented by:

- `SimEnvironment` (SOFA-backed; cerebellum + IDM training; RL refinement env)
- `KukaRealEnvironment` (KUKA KR 6-2 hardware, dev platform)
- `NeuroarmRealEnvironment` (neuroArm hardware, target platform)

Code above this boundary runs unchanged across all three.

### 10.2 Schema Design

Standardized shapes drawn from `reference_kuka/`:

- `Vec3 { x: f64, y: f64, z: f64 }` — 3D positions, forces, velocities
- `Angle<Unit>` — angles always carry their unit (Radians or Degrees) at the type level
- `Pose { position: Vec3, rotation: Quaternion }` — orientations as quaternions internally, converted at hardware boundaries

```python
class ControlMode(Enum):
    JOINT_VELOCITY = "joint_velocity"      # KUKA-style
    JOINT_POSITION = "joint_position"
    CARTESIAN_POSE = "cartesian_pose"      # neuroArm-style
    CARTESIAN_FORCE = "cartesian_force"    # contact tasks
```

`RobotCommand` carries `control_mode` plus the appropriate payload, enabling robot-agnostic policy code.

`StepResult` carries observations sized by capability:

- `joint_state` (always)
- `tool_pose` (always, derived if needed)
- `forces` (Vec3, present when sensor available)
- `stereo_frames` (when stereo cameras present)
- `mono_frame` (always)

### 10.3 KUKA → neuroArm Migration

1. KUKA bridge built first (Simulink wrapper or direct KSS interface)
2. Architecture trained against KUKA in real and SOFA sim
3. neuroArm bridge added by implementing `Environment` for neuroArm
4. Motor cortex fine-tuned for neuroArm-specific control mode (likely Cartesian pose, not joint velocity)

Architecture above the `Environment` boundary does not change between robots.

---

## 11. Simulation Strategy

SOFA's role narrows further from v2.0. Three uses:

1. **Cerebellum training data.** SOFA renders + ground-truth next-frame transitions. The cerebellum trains here via supervised next-state prediction.
2. **Inverse Dynamics Model training data.** Same SOFA scenes provide action-labeled transitions for IDM training. The IDM is later applied to real surgical video to extract pseudo-actions.
3. **Evaluation harness.** Stress tests, edge cases, safety probes, deformation regression tests. Eventually: RL refinement environment (phase 7+).

SOFA does **not** need to perfectly replicate surgical tissue dynamics. The cortical world model (PFC's predictive head) is learned from real surgical video, not from SOFA. Sim-to-real gap on tissue dynamics matters only for cerebellum and IDM, both of which target generalizable representations under domain randomization.

Domain randomization applied to SOFA scenes:

- Tissue material properties (compliance, friction)
- Lighting and camera intrinsics
- Tool geometry variations
- Anatomy variants (when MRI distributions are available)

---

## 12. Safety Architecture

Two-phase, engineered, separate from learned components. Inherited structurally from v0.8 and v2.0; the cerebellar implausibility signal replaces LeWM's role at the sync gate.

| Layer | Phase | Type | Source of truth |
|---|---|---|---|
| 1 | Sync gate (100–500 Hz) | Formally verified | Hardware limits, geometric intersection, cerebellar implausibility |
| 2 | Async assessment (10–20 Hz) | Calibrated learned | Association state, PFC deformation predictions, surgeon tags |
| 2 | Sync gate phase 2 | Formally verified against the layer-2 surface | Pre-computed safety surface from async assessment |

The safety evaluator is **substrate-output-agnostic**. Every command is gated regardless of source. Safety training data and pipeline remain separate from policy training. Conformal prediction calibrates uncertainty in async assessment.

---

## 13. Development Phases

| Phase | Capability | Components active | Robot |
|---|---|---|---|
| **0** | Infrastructure: `Environment` interface, schemas, backbone checkpoints | Stream encoders frozen | None |
| **1** | Self-supervised perception + association cortex on surgical video | Stream encoders (fine-tune), association cortex, entorhinal | None |
| **2** | Cortical training: PFC, premotor, motor cortex predictive coding | + PFC, premotor, motor cortex | None |
| **3** | Cerebellum + IDM in SOFA | + Cerebellum, IDM | KUKA in sim |
| **4** | Imitation policy on surgical-video pseudo-actions | + Motor cortex BC head | KUKA in sim |
| **5** | KUKA hardware bring-up, real-data fine-tuning | All components fine-tuned with real KUKA observations | KUKA real |
| **6** | Hippocampus active at inference (no consolidation) | + Hopfield + active entorhinal | KUKA real |
| **7** | Offline consolidation between cases | Replay → cortical fine-tuning | KUKA real |
| **8** | RL refinement (Dreamer-style generative replay) | + Imagined trajectory training | KUKA real |
| **9** | neuroArm port | `NeuroarmRealEnvironment` + control mode adaptation | neuroArm |
| **10+** | Procedural autonomy | PFC composes lower-level behavior internally | neuroArm |

Each phase adds data, training, and evaluation. None requires architectural rewrites.

---

## 14. Rejected Patterns

Carried over from v0.8 and v2.0 (still valid):

- **Fixed skill library** as autonomy center — does not scale with surgical variability
- **Inference-time RAG** — behavior-changing knowledge enters via training, not retrieval
- **Pure end-to-end VLA** — lacks entity state, multi-loop control, formal safety boundaries
- **From-scratch training of perception backbones** — pretrained checkpoints exist, use them
- **Annotation-heavy supervised pretraining** — self-supervised + imitation is sufficient and cheaper

New rejections in v3:

- **Separate "world model" module distinct from the planner.** The brain doesn't have one; the cortex serves both functions in shared substrate.
- **Episodic memory as a retrieval buffer.** The brain doesn't have buffers; it has fast weights. Modern Hopfield is the substrate.
- **Discrete entity-store fields (`semantic_class`, `is_vessel`, etc.).** Identity is geometric in continuous embedding space (v0.8 §4 principle).
- **Single backbone serving both ventral and dorsal needs.** Two streams have different inductive biases for good reasons.

---

## 15. Open Questions

1. **Cortical module sizing.** How many parameters per region? Brain-aligned proportions or compute-dictated proportions?
2. **Cross-attention bandwidth.** How rich should cross-attention between regions be? Full attention is expensive; sparse attention may starve specific pathways.
3. **Hopfield capacity in practice.** Modern Hopfield gives exponential capacity in theory. Surgical case length and entity count must fit within practical regime; needs empirical validation.
4. **IDM transfer accuracy.** How well do SOFA-trained IDMs extract pseudo-actions from real surgical video? Validation requires ground truth from at least one robot-collected session.
5. **Backbone surgical specificity.** How much surgical-domain fine-tuning is needed before stream features are useful for cortical training?
6. **Cerebellum sim-to-real.** Does sim-only cerebellum training generalize to real robot dynamics under domain randomization, or does it need real fine-tuning earlier than phase 5?
7. **Pseudo-action quality at fine temporal scale.** Per-frame pseudo-actions are noisy. What temporal smoothing makes behavioral cloning stable?
8. **KUKA control mode mapping.** Does the Simulink bridge expose joint velocity, Cartesian pose, or both? Affects the `ControlMode` selected during dev.
9. **neuroArm integration depth.** What does `NeuroarmRealEnvironment` need to expose to match KUKA-trained policy expectations?
10. **Predictive-coding loss tuning across regions.** A single loss family does not mean a single set of hyperparameters. Which masking ratios, prediction horizons, and feature spaces work for each cortical module?

---

## 16. Final Position

v3 is a brain-aligned hybrid system:

- **Two perception streams** (ventral identity, dorsal spatial) feed the cortex
- **One cortical substrate type** (predictive-coding transformer modules) specialized into regions by connectivity
- **Three memory timescales** — activations (working memory), Hopfield fast weights (hippocampus), cortical slow weights (long-term)
- **Cerebellum-style forward model** for fast motor prediction and safety implausibility
- **Imitation from surgical video** via IDM-extracted pseudo-actions
- **Engineered safety gate** with formal verification at every command
- **One environment interface** spanning sim, KUKA, and neuroArm

The system is built on the bet that the right brain-aligned ML organization, plus 2025–2026 self-supervised representation learning, plus IDM-bridged imitation, is sufficient to bootstrap surgical autonomy from mono surgical video — without annotation, supervised perception labels, or multimodal sensor ground truth.

**Trainable today. Brain-aligned in structure. Robot-agnostic at the interface boundary.**
