# Autonomous Neurosurgery as Structured Learned Control: An Architectural Thesis

> **Status:** Theoretical position paper, not a results paper. This document proposes an architecture and a falsifiable research program; it does not claim experimental validation. Companion documents are `docs/library/ARCHITECTURE.md` (canonical engineering specification) only. External citations are tracked in `docs/library/REFERENCES.md`.

## Abstract

Autonomous surgical robotics is at a paradigmatic juncture. Traditional image-guided neurosurgery (IGNS) systems rely on rigid-body registration assumptions that are invalidated by intraoperative brain shift [Roberts et al., 1998; Bayer et al., 2017]. Conventional surgical-robotics stacks depend on brittle pipelines of task-specific modules and fixed action libraries — the paradigm exemplified by the da Vinci Research Kit (dVRK), STAR [Saeidi et al., 2022], and most clinical platforms (Mako, ROSA, Auris/Monarch). Conversely, monolithic end-to-end vision-language-action (VLA) models in the RT-1/RT-2/OpenVLA/π0 family [Brohan et al., 2022; 2023; Kim et al., 2024; Black et al., 2024] lack the inspectable entity state, multi-loop control, and engineered safety boundaries necessary for clinical deployment. This paper proposes an architectural thesis for autonomous neurosurgery: the system should be framed as a *structured learned control* problem. We present an architecture that combines self-supervised foundation models for perception and world modeling, continuous embedding-first entity state, multi-timescale memory with planner-gated consolidation, and a compositional policy substrate trained via imitation learning over inverse-dynamics-model-extracted pseudo-actions. Crucially, we separate soft learned reasoning from an engineered safety boundary whose Layer 1 invariants are formally verified and whose Layer 2 risk surface is calibrated by conformal prediction. The architecture is structured so as not to preclude higher-autonomy extensions in principle, but the clinical scope of this thesis is supervised tumor resection within the Level 2–3 range of the Yang et al. [2017] taxonomy. This document outlines the clinical problem framing, the levels-of-autonomy positioning, the methodological principles, the component-by-component rationale, an explicit comparison against named prior systems and naive baselines, an evaluation logic with falsifiable component-level and thesis-level claims, a staged research roadmap, and a Limitations and Threats to Validity section. It is intended as a falsifiable starting point for a research program, not as a results paper.

## 1. Introduction: The Clinical and Technical Problem

Neurosurgical procedures occur in a visual and physical field characterized by high-frequency non-semantic detail, semantic ambiguity, and continuous physical flux. The operating surgeon must navigate a landscape that is inherently non-rigid, prone to rapid physiological changes, and frequently obscured by instruments, smoke from electrosurgical devices, specular highlights on moist tissue, and chaotic fluid movement.

### 1.1 Brain Shift and Semantic Ambiguity

Traditional image-guided neurosurgery (IGNS) systems rely on rigid registration of pre-operative MRI/CT to the patient's skull, as exemplified by the Medtronic StealthStation and Brainlab Cranial platforms. These rigid-body assumptions break down the moment a craniotomy is performed and the dura is incised. Loss of cerebrospinal fluid, osmotic diuresis, and surgical maneuvers (retraction, resection) deform the parenchyma; cortical surface displacements of 1–2 cm have been documented in clinical cohorts [Roberts et al., 1998; Miga et al., 2003; Bayer et al., 2017]. The visual field is also fraught with semantic ambiguity: as cotton patties become saturated with blood, their visual properties converge with surrounding tissue, increasing the risk of retained surgical items; and blood itself acts as a dynamic dye that confounds RGB-based segmentation. Intraoperative imaging modalities (iMRI, intraoperative ultrasound, intraoperative CT) and biomechanical brain-shift compensation [Miga et al., 2016; Heinrich & Jenkinson, 2012] partially mitigate this, but each introduces workflow disruption, cost, or boundary-condition uncertainty that motivates a perception-driven alternative. Classical physics-based non-rigid registration (PBNRR) via Finite Element Methods (FEM) also carries prohibitive runtime latency (cubic in mesh resolution under typical solvers), boundary-condition uncertainty, and an inability to gracefully handle topological changes such as resection.

### 1.2 The Limits of Classical Robotics in Surgery

Most practical surgical robotics systems — including the da Vinci Research Kit (dVRK), STAR [Saeidi et al., 2022], and the demonstration stacks used by Krishnan, Kim, Yip, and Hannaford for sub-task automation — are built as pipelines of task-specific modules, explicit state machines, and fixed action libraries (`Grasp`, `Aspirate`, `Cauterize`). Surveys [Attanasio et al., 2021; Haidegger, 2019] confirm this is the dominant paradigm. While these systems can be made reliable for narrow demonstrations, they struggle when the surgical scene changes, when a tool or tissue behaves differently, or when two behaviors need to compose in ways that were not explicitly programmed. Surgery contains too many intermediate states for closed skill definitions, and hand-coded state machines are brittle under unexpected tissue behavior.

### 1.3 The Limits of Pure End-to-End Learning

At the other extreme, monolithic Vision-Language-Action (VLA) models — RT-1 [Brohan et al., 2022], RT-2 [Brohan et al., 2023], OpenVLA [Kim et al., 2024], Octo [Octo Model Team, 2024], π0 [Black et al., 2024], and the broader Open X-Embodiment family — attempt to map directly from pixels and text to motor commands. While these models excel at compositional generalization in general manipulation, they do not maintain inspectable per-entity state, operate at a single control frequency, and cannot provide deterministic safety guarantees that a surgical certifying body or operating surgeon can audit. A single forward pass cannot simultaneously deliver millisecond-scale servoing and second-scale strategic planning, and current VLA models do not expose a substrate where surgeon-confirmed hard constraints can be applied externally to policy outputs.

### 1.4 The Research Problem

Sections 1.1–1.3 surface a common gap: no existing paradigm simultaneously maintains *persistent entity identity under occlusion*, *cumulative procedure state across the case*, and *a deterministic safety boundary independent of the learned stack*. Surgery is a sequence of irreversible state changes — tissue resected, vessels cauterized, patties placed and retrieved, parenchyma progressively deformed — and a controller that does not represent these changes cannot reason about what to do next. Frame-by-frame semantic segmentation has no notion of identity over time and cannot represent that a tumor is half-resected, that an accumulating pool of blood implies an active bleeder outside the current view, or that a cotton patty placed three minutes ago must still be retrieved before closure. Hand-coded skill libraries (§1.2) enumerate only discrete states and compose poorly under unmodeled tissue behavior, while monolithic VLA forward passes (§1.3) expose neither inspectable per-entity state nor a substrate where surgeon-confirmed hard constraints can be enforced externally to the policy. The technical challenge is therefore to track changing entities — surgical tools, cotton patties, live tissue, evolving bleed risk, and volumetric deformation — continuously over time and under occlusion, *and* to expose an engineered safety surface that operates independently of that tracker. This requires a continuous, physically consistent representation of the surgical scene coupled with a learned compositional policy and an engineered safety boundary.

### 1.5 Levels of Autonomy and Clinical Scope

This thesis adopts the six-level surgical autonomy taxonomy of Yang et al. [Sci. Robot., 2017] (subsequently refined by Haidegger [2019]):


| Level | Description                                                                           |
| ----- | ------------------------------------------------------------------------------------- |
| 0     | No autonomy (full teleoperation)                                                      |
| 1     | Robot assistance (e.g., motion scaling)                                               |
| 2     | Task autonomy (sub-task automation under continuous surgeon supervision)              |
| 3     | Conditional autonomy (system performs whole tasks, surgeon supervises and intervenes) |
| 4     | High autonomy (system makes medical decisions; surgeon as supervisor)                 |
| 5     | Full autonomy                                                                         |


**The architecture is structured so as not to preclude Level 3–5 surgical autonomy in principle.** The component choices are engineered to support higher autonomy and high generalization, while the clinical scope of this thesis is supervised tumor resection within the Level 2–3 range. The point is that the substrate should remain extensible to higher autonomy if later research and regulation justify it, without claiming that such deployment is already the goal of this document.

### 1.6 Ethics, Regulatory Pathway, and Data Governance

Any autonomous-surgery research program must engage three external constraints that are out of scope to fully resolve here, but that constrain the architecture from the outset:

1. **Regulatory framing.** Any clinically deployable component is a Software-as-a-Medical-Device (FDA SaMD) and must conform to IEC 62304 (medical device software lifecycle) and IEC 60601-2-77 (robotically assisted surgical equipment). The clinical translation pathway anticipated for this work follows the IDEAL-D framework for surgical innovation [McCulloch et al., 2009]. The architecture's separation of learned reasoning from an engineered safety boundary (§5.6) is partly motivated by the need to localize the regulatory verification burden in the safety layer rather than across the entire learned stack.
2. **Ethics and human-subjects review.** Cadaveric and any subsequent in-human work are gated on institutional review board (IRB) approval; the staged roadmap of §8 reflects this. The shared-autonomy interface (§5.8) preserves the operating surgeon as the ultimate authority to maintain compliance with current standards of care.
3. **Data governance.** All clinical video, telemetry, and intraoperative imaging used for training is subject to PHI protection, informed consent, and de-identification protocols. The training-from-archived-video strategy of §5.5 inherits these constraints; the architecture cannot ingest data outside their envelope, and this binds the training and consolidation pipelines (§5.10).

## 2. Thesis and Conceptual Contribution

**Central Hypothesis:** Autonomous neurosurgery should be framed as a learned structured-control problem with continuous entity state, multi-timescale memory, multi-loop control, and a hard engineered safety boundary.

We assert that surgical autonomy will scale only if the system can learn continuous structure and compose behavior across contexts, while preserving hard deterministic safety at the robot boundary. Capability must expand by training on more data, not by adding code paths.

**Conceptual Pillars:**

1. **Embedding-First State.** The surgical world is represented as a variable-cardinality set of persistent entities. Identity and semantics are captured in continuous multimodal embeddings, not discrete class labels. That representation supports similarity reasoning across entities and can encode how a tumor's risk surface changes during resection.
2. **Multi-Timescale Memory.** Memory is not an inference-time retrieval database for behavior-changing knowledge. It lives at three timescales: activations (working memory for immediate tracking), Hopfield fast weights (case-long associative recall for patient-specific calibration), and model slow weights (cross-case dynamics priors consolidated offline).
3. **Multi-Loop Control.** The system is organized as perception, planning, and fast control loops operating at different time scales rather than as a single monolithic policy. This makes it possible to combine millisecond-scale servoing with slower deliberation, escalation, and consolidation.
4. **Imitation via Inverse Dynamics.** In the absence of robot action labels on archival surgical video — the gap between abundant raw video and the labeled action trajectories required for behavioral cloning, which we refer to as the *action-label gap* — the policy is bootstrapped by extracting pseudo-actions from mono surgical video using an Inverse Dynamics Model (IDM) trained in simulation, in the spirit of VPT [Baker et al., 2022]. This allows learning from large archives of human surgical video.
5. **Separation of Reasoning and Safety.** Learned components (policy, planner, world model) handle soft reasoning, affordances, and behavior composition. An engineered safety evaluator sits between every command and the robot. Its **Layer 1** invariants are formally verified; its **Layer 2** risk surface is conformally calibrated. Clinical safety in the formally verified sense is therefore localized to Layer 1; Layer 2 provides statistical guarantees, not deterministic ones (§5.6).
6. **Hybrid Identity + Safety Query.** Continuous embeddings carry identity and persistence under deformation/occlusion, while typed, interpretable scene-graph-style predicates carry safety constraints and auditability.

## 3. Related Work and Positioning

Sections 1.1–1.3 already established the clinical problem, so this section does not restate those failure modes. Instead, it positions the architecture against the closest prior paradigms, states which ideas are retained, and identifies where the thesis departs from them. Cited systems are tracked in `docs/library/REFERENCES.md`.

### 3.1 Surgical autonomy systems and skill-library paradigm

The dominant surgical-autonomy paradigm pairs classical perception with hand-coded skill libraries on top of teleoperation platforms: dVRK-based research stacks, demonstration systems by Krishnan, Kim, Yip, Hannaford, and the STAR system [Saeidi et al., 2022]; commercial platforms (da Vinci SP, CMR Versius, Auris/Monarch, Stryker Mako, Zimmer ROSA, Accuray CyberKnife, ROBODOC) embed similar designs at varying levels of autonomy. Surveys [Attanasio et al., 2021; Haidegger, 2019] document the resulting fragility: skill composition is combinatorial, and adding new behaviors requires new code paths. **We accept the perceptual front-ends and physical platforms of these systems as legitimate baselines but reject fixed skill libraries as the autonomy center.** A compositional learned policy (§5.5) is intended to subsume the skill-library role without bounding capability by the engineering budget.

### 3.2 Brain-shift compensation and intraoperative imaging

Biomechanical brain-shift compensation [Roberts et al., 1998; Miga et al., 2003, 2016; Bayer et al., 2017] and ultrasound-driven or MRF-based deformable registration [Heinrich & Jenkinson, 2012] are the dominant alternatives to rigid IGNS. Intraoperative MRI, ultrasound, and CT systems update the prior at discrete points during surgery. **The architecture treats the pre-operative MRI prior as initialization at \(t=0\) (§5.7) and replaces continuous biomechanical compensation with a learned residual world model (§5.4); intraoperative imaging is complementary, not excluded, but is not assumed available on every platform.** A reviewer should read this as a research bet on perception-driven deformation modeling, not as a claim that biomechanical methods are obsolete.

### 3.3 End-to-end Vision-Language-Action models

The strongest open VLA systems — RT-1, RT-2, OpenVLA, Octo, π0, Open X-Embodiment — demonstrate that a single transformer can map pixels and language to actions with broad cross-task generalization. Surgery-specific extensions (Surgical-VQLA, SurgicalGPT) are nascent. **The thesis rejects these as the *whole* system because they lack inspectable per-entity state, do not expose an external separation between strategic planning and millisecond-scale servoing, and expose no surface where surgeon-confirmed hard constraints can be enforced externally to the policy.** Hierarchical-control literature such as options frameworks and hierarchical RL does address the planning/control split, but it usually assumes cleaner symbolic subtask interfaces than surgery offers; this thesis borrows the separation of timescales without inheriting a fixed subtask ontology. The compositional policy substrate of §5.5 borrows VLA training methodology but is consumed by a planner and a safety gate rather than driving the robot directly.

### 3.4 Object-centric perception, SLAM, and surgical-scene representation

Alternatives to slot-attention-style entity binding include query-based detectors and trackers (DETR, MOTR), classical SLAM and reconstruction (ORB-SLAM3, BundleFusion), neural scene representations for surgical video (EndoNeRF), and surgical-domain scene graphs (4D-OR; surgical phase recognition on Cholec80, CATARACTS, M2CAI). **We adopt slot-attention-derived entity binding [Locatello et al., 2020; Kipf et al., 2022; Elsayed et al., 2022; Seitzer et al., 2023] for variable-cardinality decomposition with continuous embeddings, and treat SLAM/NeRF-style geometric reconstruction as complementary inputs to the Action stream rather than as an alternative to entity-centric state.** The choice is justified empirically in §5.2 and at risk under §10.

### 3.5 World-model alternatives

Within model-based control, the landscape includes Dreamer V1–V3 (reconstructive latent world models), MuZero (search-based planning over learned dynamics), IRIS (transformer-based world models), and V-JEPA 2 (predictive latent world models for video). DINO-WM [Zhou et al., 2024] is selected because it operates on pretrained DINOv2 features, enabling zero-shot planning over a substrate that is also used by the perception stack. **The thesis commits to DINO-WM for its alignment with a shared visual prior, while acknowledging that Dreamer-V3 and V-JEPA-2-WM are credible alternatives whose comparative evaluation on surgical video is not yet performed.** A falsification of DINO-WM in Stage 2 (§8) triggers a comparative ablation against these alternatives.

### 3.6 Memory architectures and retrieval

The closest alternatives to a Hopfield/slow-weights design are RETRO-style retrieval [Borgeaud et al., 2022], Memorizing Transformers [Wu et al., 2022], the Differentiable Neural Computer [Graves et al., 2016], and conventional retrieval-augmented generation (RAG). **We reject open-ended inference-time episodic retrieval as a path for behavior-changing knowledge in clinical use because the retrieval set itself must be frozen and evaluated before deployment; once constrained that tightly, the advantage over slow-weights distillation largely disappears. We therefore adopt modern Hopfield fast weights [Ramsauer et al., 2021] as case-long storage and offline distillation into slow weights as the consolidation path.** Hopfield capacity at case length is an open empirical question (§2.8 and §3.4); a failed capacity gate falls back to a learned key-value memory exposing the same interface.

### 3.7 Safety-by-engineering: formal methods and surgical primitives

The canonical safety-by-engineering frameworks are Control Barrier Functions [Ames et al., 2019], Hamilton–Jacobi reachability [Bansal et al., 2017], and the Simplex / runtime-assurance architecture [Sha, 2001]. The surgical-robotics tradition contributes virtual fixtures and active constraints [Abbott et al., 2007; Davies et al., 2007; Bowyer et al., 2014]. Conformal prediction has been increasingly applied to safe planning [Lindemann et al., 2023; Tibshirani et al., 2019]. **The two-phase safety evaluator of §5.6 is positioned as a hybrid: Layer 1 invariants are formally verified in the spirit of Simplex and CBFs; the Layer 2 risk envelope is learned and conformally calibrated.** Virtual fixtures are subsumed as a special case of the SafetySurface (a static, surgeon-set element). The thesis does not claim to obsolete CBFs or HJ reachability; it claims that high-dimensional, deformation-aware no-go geometry is more naturally expressed as a learned-and-calibrated surface than as a hand-designed barrier function — an empirical claim that §7 makes falsifiable.

### 3.8 Naive Baselines and Why They Are Insufficient

A reviewer is entitled to ask why each component is necessary. The following table enumerates the strawman baselines that the architecture must outperform; concrete falsification experiments are described in §7 and §3.


| #   | Naive baseline                                                                                                                         | Operational rebuttal                                                                                                                        | Falsifies thesis if …                                                                                                                                                                    |
| --- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Pure end-to-end behavioral cloning from teleop video, trained from scratch; no foundation pretraining, no entity state, no safety gate | No path to surgeon-auditable state, no safety substrate independent of the policy, single control frequency                                 | … BC-from-scratch matches the proposed policy on compositional generalization *and* a post-hoc audit recovers entity state automatically (unlikely but not impossible)                   |
| 2   | Hand-coded finite state machine with classical perception                                                                              | Combinatorial in skills × scene contexts; brittle under unmodeled tissue behavior; matches §3.1                                             | … skill-library autonomy demonstrably scales to neurosurgical complexity within a fixed engineering budget                                                                               |
| 3   | Pure RL in simulation with sim-to-real transfer, no demonstrations                                                                     | Reward design for surgery is unsolved; soft-tissue sim-to-real gap is large; sample-inefficient                                             | … contract-based sim rewards alone produce policies that transfer to phantom/cadaver tasks competitively with imitation                                                                  |
| 4   | Supervised semantic segmentation + classical motion planning                                                                           | Frame-by-frame; no continuous deformation state; no compositional behavior                                                                  | … per-frame segmentation + classical planning achieves comparable directive-level success on a defined neurosurgical task suite                                                          |
| 5   | LLM/VLM as planner with tool-use API over classical perception (e.g., GPT-style controller)                                            | No millisecond-scale servoing; planner-policy latency stack incompatible with surgical control; no auditability of in-context-learned state | … an LLM-planner stack achieves comparable directive success with formally bounded worst-case latency and surgeon-auditable state                                                        |
| 6   | Full high-fidelity FEM/PBNRR digital twin at runtime                                                                                   | Latency cubic in mesh resolution; topological-change handling is brittle (resection); boundary conditions uncertain                         | … FEM-based deformation modeling can be made real-time and robust to topology change at clinical fidelity                                                                                |
| 7   | Monolithic VLA (RT-2 / OpenVLA / π0) fine-tuned on surgical data                                                                       | No inspectable entity state; no formal safety surface; no sub-second planning ↔ servoing separation                                         | … a monolithic VLA can be augmented with an external safety-projection layer that is functionally equivalent to §5.6 — at which point it is structurally equivalent to this architecture |


The thesis's central claim is that **structured learned control** (entities + multi-timescale memory + compositional policy + engineered safety) outperforms the cheapest baseline that achieves the same observable behavior. If any row's "falsifies thesis if …" condition is met empirically, the corresponding pillar of §2 must be revised.

### 3.9 Research Findings Integration

The deep-research corpus in `docs/research/architecture/research-findings.md` yields additional architecture constraints:

- **Modularity and adaptability**: modular PPC stacks are the default against rigid state-machine behavior.
- **Two-layer safety with explicit authority**: hard Layer-1 invariants plus adaptive Layer-2 risk gating are kept distinct.
- **Identity-first, query-second representation**: embeddings are required for occlusion robustness; explicit safety predicates remain the audit surface.
- **Loop-separation and timing contracts**: planner, safety, and servo loops remain bandwidth-separated for stability.
- **Tiered memory with capacity-aware fallback**: Hopfield fast memory is paired with case-scale structures when retention or latency demands exceed practical limits.
- **Action-label-gap plan**: IDM pseudo-action extraction remains the load-bearing bridge from unlabeled video to policy.
- **Governance and HMI as first-class**: consent/de-identification/logging and takeover-state clarity are design constraints, not post-hoc process items.

## 4. Methodology: Architecture Overview

The system operates as a hierarchy of loops around a shared entity knowledge store. The major system layers are:

1. **Perception:** Dual streams (Action and Semantic) extract geometric and semantic features from video and sensors.
2. **Entity Binding:** Variable-cardinality slot attention assigns persistent identities to observed objects, creating active entity states.
3. **Memory:** A three-timescale architecture (activations, Hopfield fast weights, slow weights) with planner-gated consolidation.
4. **World Model:** An embedding-conditioned model (DINO-WM) predicting future entity states and interactions.
5. **Policy Substrate:** A learned compositional policy that consumes entity state, directives, and contracts to emit short-horizon target trajectories.
6. **Slow Planner:** A deliberative module that translates surgeon goals into directives, manages memory consolidation, and handles escalation.
7. **Fast Controller:** A classical controller (MPC, visual servoing) tracking substrate trajectories at high frequency.
8. **Safety Evaluator:** A two-phase system (async assessment and sync command gate) providing hard safety guarantees on the engineered Layer 1 invariants and statistically calibrated guarantees on the learned Layer 2 surface.
9. **Environment Interface:** A unified protocol ensuring sim/real parity across KUKA, neuroArm, and simulation.

### 4.1 Notation and Symbols

The thesis uses a small set of formal symbols. We collect them here for reference; component-level subsections of §5 introduce additional structure where required. Units are SI throughout unless noted.

| Symbol                                             | Meaning                                                                                | Type / units                                                                                                                                                   |
| -------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $t$                                                | Continuous time                                                                        | seconds                                                                                                                                                        |
| $E_t = e_{1,t}, \dots, e_{k_t,t}$                  | Set of entity states at time $t$                                                       | variable cardinality $k_t \in \mathbb{N}$                                                                                                                      |
| $e_{i,t} \in \mathbb{R}^d$                         | Continuous embedding for entity $i$                                                    | $d$ on the order of a foundation-model embedding (typ. $d = 384$–$1024$)                                                                                       |
| $\xi_{i,t}$                                        | Decoded geometric attributes of entity $i$ (3D pose in SE(3), bounding geometry)       | SE(3) pose + bounding primitive                                                                                                                                |
| $\rho_{i,t}$                                       | Decoded interaction digest fields (contact, applied force, deformation, recent events) | structured record (see §5.3)                                                                                                                                   |
| $a_t \in \mathcal{A}$                              | Policy action at time $t$                                                              | continuous Cartesian delta pose $\Delta T \in \mathfrak{se}(3)$, gripper state $g \in [0,1]$, instrument-specific channels (cautery duty cycle, suction state) |
| $u_t$                                              | Joint-space command emitted by the fast controller after kinematic projection          | joint-vector in robot configuration space (rad)                                                                                                                             |
| $\mathcal{D}$                                      | Active language directive(s) issued by the planner / surgeon                           | structured directive object (text + parameter slots)                                                                                                           |
| $\mathcal{C}$                                      | Active behavior contract set                                                           | finite set of typed constraints (force bounds in N, spatial bounds as SDFs, deadlines in s, hard/soft flag) — see §5.5                                         |
| $\pi(a_t \mid E_t, \mathcal{D}, \mathcal{C})$      | Policy substrate distribution over actions                                             | conditional distribution over $\mathcal{A}$                                                                                                                    |
| $f_\theta$                                         | World model dynamics: $\hat{E}*{t+\Delta t} = f*\theta(E_t, a_t)$                      | learned forward map on entity embeddings                                                                                                                       |
| $\mathcal{S}_t$                                    | SafetySurface — calibrated risk envelope at time $t$                                   | per-entity signed distance field (SDF) over Cartesian workspace, with risk score $r: \mathbb{R}^3 \to [0,1]$                                                   |
| $\mathcal{N}_t \subset \mathbb{R}^3$               | No-go geometry — sublevel set of $\mathcal{S}_t$ at the deployed risk threshold        | open subset of the workspace                                                                                                                                   |
| $\mathcal{M}_{\text{safe}, t} \subset \mathcal{A}$ | Action-space safe manifold induced by $\mathcal{S}_t$ and Layer 1 invariants           | measurable subset of $\mathcal{A}$                                                                                                                             |
| $\text{Proj}*{\mathcal{M}*{\text{safe}, t}}$       | Projection onto $\mathcal{M}_{\text{safe}, t}$ used by the sync command gate           | measurable map $\mathcal{A} \to \mathcal{M}_{\text{safe}, t} \cup \bot$, where $\bot$ denotes hard stop                                                        |


### 4.2 Methodological Principles

- **Self-supervised first:** No component requires labeled data as a first-class dependency. Annotation is a refinement, not a prerequisite.
- **Latency contracts:** Interface contracts include explicit rates, deadlines, and deterministic arbitration requirements; adaptive or learned loops must be designed around, not over, the safety gate cadence.
- **Embedding-first state:** Per-entity state is continuous and addressable, never reduced to discrete labels.
- **Memory = weights and activations:** No inference-time episodic retrieval buffers for behavior-changing knowledge; memory lives in network activations, fast weights, and slow weights.
- **Foundation models as substrate:** Use pretrained checkpoints (DINOv2, V-JEPA 2, DINO-WM) to leverage broad visual priors, bypassing the need for from-scratch training on scarce surgical data.
- **Imitation from observation:** Surgical video is the demonstration set, bridged by IDM pseudo-actions.
- **Engineered safety:** Layer 1 of the safety evaluator is formally verified; Layer 2 is conformally calibrated. Learned components inform but cannot override the projection onto $\mathcal{M}_{\text{safe}, t}$.
- **Sim/real parity:** One environment interface (`Environment` protocol) spans simulation and real hardware so that code above the boundary runs unchanged. Time-synchronization requirements (≤ 1 ms drift between modalities) for real-robot phases are treated as a design constraint.

## 5. Component-by-Component Rationale

### 5.1 Perception (Action and Semantic Streams)

- **What it does:** Extracts low-latency geometric features (Action stream via SE(3)-equivariant layers) and high-resolution semantic features (Semantic stream via foundation models like SAM 2 / V-JEPA 2).
- **Why it exists:** Control needs millisecond-scale geometry (pose, contact frames); planning needs rich semantics (anatomy labels, segmentation).
- **Why this form:** Splitting streams avoids forcing one network to compromise between latency and semantic depth. SE(3) layers bake in 3D rigid motion symmetries, stabilizing representations of tool pose and contact frames without needing to learn them from scratch. To resolve timing discrepancies before fusion into the Entity Binding stage, the asynchronous semantic stream uses latency-aware feature queues that project high-resolution embeddings forward in time using the Action stream's high-frequency geometric updates.
- **What can go wrong:** The streams might misalign during fusion, or the semantic stream may introduce too much latency if not properly pipelined.
- **Connection:** Feeds features into the entity binding stage.

### 5.2 Entity Binding and Continuous State

- **What it does:** Uses variable-cardinality slot attention to discover and track entities, representing them as continuous multimodal embeddings. Formally, at time $t$, the surgical scene is modeled as a state space comprising a set of continuous entity states $E_t = e_{1,t}, e_{2,t}, \dots, e_{k,t}$. Each entity $e_{i,t} \in \mathbb{R}^d$ is a dense vector embedding that encapsulates semantic identity, 3D pose, bounding geometry, and inferred physical properties.
- **Why it exists:** Surgery requires persistent handles for tools, tissue, and fluids to allow for planning, safety constraints, and surgeon communication.
- **Why this form:** *Tumor resection illustrates why continuous state matters.* A tumor is a changing volume and risk surface, not a binary `present`/`absent` class. Continuous embeddings allow the model to learn similarity and difference along the resection trajectory. *Physically overlapping entities illustrate why per-entity decomposition is correct.* A vein draped over a tumor must be treated as two entities with different affordances (preserve vs. resect) and constraints, even if they are mechanically coupled.
- **What can go wrong:** *Entity splitting is a known failure mode of slot attention.* A heterogeneous entity (e.g., a multi-lobed tumor) might be over-segmented into multiple slots.
- **Occlusion stress:** During prolonged smoke/blood occlusion or abrupt re-entry, slot identity can fragment or duplicate. Memory-aware re-identification is therefore part of entity binding; unresolved identity is treated as low-confidence until reconverged.
- **Connection:** Provides the foundational state representation consumed by all downstream learned components.

### 5.3 Entity Knowledge Store and Planner-Gated Consolidation

- **What it does:** Maintains an *interaction digest* $\rho_{i,t}$ for each entity, updated at the planner's rate (0.5–2 Hz) based on world model prediction errors and risk signals. The interaction digest is a structured record with at minimum: contact partners, accumulated applied force history (windowed), observed deformation extent, recent event tags (`bleed`, `cauterize`, `retract`, `aspirate`), and a confidence score for each field. When the deviation between expected and observed entity states exceeds a calibrated threshold, the planner triggers a write to the Hopfield fast-weight store [Ramsauer et al., 2021]. Post-operatively, these episodic fast weights are distilled into the model's slow weights through offline *generative replay* — synthetic rollouts produced by the world model conditioned on stored fast-weight states, used as additional training data for slow-weight updates.
- **Why it exists:** To accumulate patient-specific calibration (e.g., tissue compliance, bleeding tendencies) during a procedure and consolidate cross-case priors offline.
- **Why this form:** The planner has strategic context and can decide what is worth remembering better than threshold-based event detection alone, modulating the system's attention budget in a manner loosely analogous to prefrontal modulation of memory consolidation. *Why not RAG-style retrieval?* Behavior-changing knowledge in clinical use must be subject to pre-deployment evaluation; retrieval at inference time admits unverified examples into the control loop. *Why not the Differentiable Neural Computer or Memorizing Transformer?* These provide larger working memories but do not naturally support the planner-gated, write-on-prediction-error semantics that this architecture uses to bound the write rate. They are credible fallbacks if Hopfield capacity at case length proves inadequate (§2.8 / §3.4); the interface is designed to admit a key–value memory swap.
- **What can go wrong:** The planner might fail to consolidate critical safety-relevant interactions, leading to repeated mistakes on the same tissue. Hopfield capacity may be insufficient at full case length (open question §9).
- **Capacity policy:** Planner writes to Hopfield are rate-limited by confidence and utility. If retrieval latency or recall precision degrades at longer case lengths, the interface supports a key-value/vector fallback path with compatible write/read semantics.
- **Connection:** Fuses with current observations to create the full entity embedding consumed by the policy substrate and the world model.

### 5.4 Unified Embedding-Conditioned World Model

- **What it does:** Predicts future entity states and interactions, $\hat{E}*{t+\Delta t} = f*\theta(E_t, a_t)$, using DINO-WM [Zhou et al., 2024] over DINOv2 features, conditioned on entity embeddings and per-entity dynamics adapters.
- **Why it exists:** To enable latent rollouts for the planner, support directive search, and provide interpretable readouts (predicted forces, deformations) for the Layer 2 SafetySurface (§5.6).
- **Why this form:** Foundation-model world models learn dynamics implicitly from video, capturing soft-tissue and bleeding behaviors that analytic simulators cannot easily model. Because dynamics are conditioned on entity embeddings, the world model can infer compliance and deformation by relating SE(3)-structured action geometry to stereo-evolved surface state. The forward-pass cost is fixed by encoder shape and is not a function of underlying mesh resolution — so the per-step latency is constant in the *anatomical complexity* dimension that dominates FEM cost (cubic in mesh resolution under typical solvers); the asymptotic comparison should be read as O(1) vs. O(N³) in mesh resolution, not as a unitless O(1) claim.
- **Why DINO-WM over alternatives:** Among comparable world models we considered Dreamer V3 [Hafner et al., 2023] (reconstructive latent), MuZero [Schrittwieser et al., 2020] (search-based planning over learned dynamics), IRIS [Micheli et al., 2023] (transformer-based), and V-JEPA 2 used directly as a predictive world model [V-JEPA 2 team, 2025]. We chose DINO-WM because it operates over the same DINOv2 features used by the perception stack, enabling a shared visual prior across perception, world model, and policy training. This is a research bet, not a settled choice; if DINO-WM fails the Stage-2 evidence threshold (§8), Dreamer V3 and V-JEPA-2-WM are the canonical comparator ablations.
- **What can go wrong:** The model may hallucinate physically impossible deformations or fail to predict complex fluid dynamics (bleeding). Sim-to-real gap on tissue compliance is non-trivial (§2.5).
- **Benchmark gap:** No single standardized benchmark currently spans deformation, topology change, and fluid dynamics with aligned compute-cost reporting on neurosurgical scenes; Stage-2 therefore requires an explicit stress-suite and explicit comparator protocol.
- **Connection:** Feeds prediction errors back to the planner for memory consolidation (§5.3); provides rollouts for directive search (§5.6 SafetySurface) and for the policy substrate (§5.5).

### 5.5 Compositional Policy Substrate

- **What it does:** A hybrid SSM–Transformer policy emitting short-horizon target trajectories, $\pi(a_t \mid E_t, \mathcal{D}, \mathcal{C})$, conditioned on entity state $E_t$, the active language directive $\mathcal{D}$, and the active behavior-contract set $\mathcal{C}$.
- **Behavior contracts $\mathcal{C}$:** A behavior contract is a typed constraint with an enforcement semantics. The minimal schema is `{kind, params, hard_or_soft, deadline}` where `kind ∈ {force_bound, spatial_bound, motion_speed_bound, deadline, no_go_zone}`, `params` are SI-typed (force in N, distance in mm, speed in mm/s, time in s), and `hard_or_soft` flags whether violation triggers a sync-gate intervention or a soft penalty in the policy loss. *Soft contracts modulate the policy at training and deployment; hard contracts are enforced by the safety evaluator (§5.6) and are independent of the policy's own training objective.* This separation prevents policy optimization from relaxing safety bounds.
- **Why it exists:** To generate flexible, generalized surgical behavior without relying on brittle, hand-coded state machines.
- **Why this form:** Real surgical behavior is compositional. A learned substrate shares representations across behaviors and generalizes within its training distribution. SSM layers (Mamba-2 style) provide linear-time scaling for long context; attention layers handle cross-modal fusion of language, vision, and contract tokens.
- **Why imitation rather than RL:** Reward design for surgery is unsolved; high-dimensional, soft-tissue interactions produce sparse, non-stationary, and potentially unsafe reward signals. Imitation from inverse-dynamics-extracted pseudo-actions [VPT: Baker et al., 2022; LAPA-style methods] gives a dense supervision signal grounded in expert behavior. RL augmentation via contract-based rewards in simulation is acknowledged as a Phase-9+ extension (§4.1) but is not load-bearing for the initial thesis.
- **What can go wrong:** The substrate may fail to generalize to OOD directive combinations or exhibit behavioral drift; cumulative pseudo-action drift in BC is a known failure mode (§2.4).
- **Connection:** Driven by the planner's directives; outputs are tracked by the fast controller and projected onto $\mathcal{M}_{\text{safe},t}$ by the sync command gate.

### 5.6 Safety Evaluator (Two-Phase)

- **What it does:** Two phases at different rates form an asymmetric guard. Phase 1 (async assessment, 10–20 Hz) computes deformation-aware no-go geometry $\mathcal{N}*t$ and a per-entity risk surface $r$, publishing a SafetySurface $\mathcal{S}t$. Phase 2 (sync command gate, 100–500 Hz) checks every command against $\mathcal{S}t$ and a set of Layer 1 physical invariants. The evaluator induces a safe action manifold $\mathcal{M}{\text{safe}, t} \subset \mathcal{A}$. Any proposed $a_t \notin \mathcal{M}{\text{safe}, t}$ is either projected, $a_t' = \text{Proj}*{\mathcal{M}_{\text{safe}, t}}(a_t)$, or — if no admissible projection exists — replaced by a hard stop $\bot$.
- **SafetySurface representation.** $\mathcal{S}_t$ is implemented as a per-entity signed distance field over the workspace, augmented with a scalar risk score $r \in [0,1]$ and an associated nominal coverage level. The no-go geometry $\mathcal{N}_t$ is the sublevel set $x \in \mathbb{R}^3 : r(x) > \alpha$ at the deployed risk threshold $\alpha$, expanded by a deformation-aware halo derived from world-model rollouts.
- **Layer 1 (formally verified, engineered).** Hard physical invariants — workspace bounds, joint limits, force ceilings, stop-on-fault — are expressed as decidable predicates over robot state. These are formally verified using techniques compatible with the Simplex / runtime-assurance architecture [Sha, 2001]; they correspond to the Control Barrier Function [Ames et al., 2019] and Hamilton–Jacobi reachability [Bansal et al., 2017] traditions and provide the same kind of deterministic guarantee. Surgeon-set virtual fixtures [Abbott et al., 2007; Bowyer et al., 2014] are admitted as Layer 1 spatial bounds.
- **Audit and e-stop semantics:** Safety events are timestamped with explicit action provenance for post-op traceability. Emergency-stop transitions and arbitration decisions must remain reconstructable and are designed to terminate motion in a bounded safe state.
- **Layer 2 (learned, conformally calibrated).** The risk score $r$ over the workspace is computed from world-model rollouts of the policy's action samples. To avoid heuristic thresholds, we apply conformal prediction [Vovk et al., 2005; Angelopoulos & Bates, 2021]: a held-out calibration set of (rollout, observed outcome) pairs yields non-conformity scores from which a quantile threshold is computed for a target nominal coverage (e.g., $1 - \alpha = 0.95$). At inference, only candidate actions whose worst-case predicted deformation lies within the conformal envelope are admitted. The surgical setting is non-stationary; we follow the covariate-shift-aware variants of conformal prediction [Tibshirani et al., 2019] and the safe-control adaptations [Lindemann et al., 2023], explicitly track sliding-window calibration during a case, and treat distribution-shift coverage as an open empirical question (§9, §4.5). A separate safety-calibration protocol will specify the details; until that protocol is established, Layer 2 is *advisory*, not vetoing, on novel tissue distributions.
- **Why it exists:** To provide deterministic guarantees on Layer 1 invariants and statistically calibrated guarantees on the Layer 2 risk surface, without coupling either to policy optimization.
- **Why this form vs. CBFs, HJ reachability, or pure virtual fixtures:** Hand-designed Control Barrier Functions and Hamilton–Jacobi value functions provide rigorous safety guarantees but are difficult to specify directly over a deformable, partially observed surgical scene with shifting topology. Static virtual fixtures cannot represent deformation-aware no-go zones that move with the tissue. Our approach **does not replace these methods**; it composes with them. Layer 1 is structurally a Simplex-style supervisor with CBF-like invariants over robot state; Layer 2 lifts the same idea to a high-dimensional, learned, conformally calibrated envelope over workspace risk. The empirical claim in §7 is that the Layer 1 + Layer 2 composition produces tighter, more useful no-go geometry than any of the individual classical methods alone on a defined task suite.
- **What can go wrong:** The Layer 2 surface depends on the world model being well-calibrated on real tissue — a known open question (§2.5, §4.5). Conformal coverage guarantees degrade under distribution shift if the calibration set is not refreshed. The sync gate may be too conservative, blocking necessary surgical actions; the false-positive / false-negative trade-off must be measured against simulated and (later) phantom adversarial baselines. Layer 1 does *not* protect against Layer 1 specification errors — invariants must be audited as a system-level deliverable.
- **Connection:** Sits between all command sources (policy, controller, surgeon) and the robot hardware. Receives world-model rollouts from §5.4, contracts $\mathcal{C}$ from §5.5, and surgeon-imposed Layer 1 constraints from the HMI of §5.8.

### 5.7 System Initialization and Bootstrapping

- **What it does:** Reconciles the pre-operative MRI prior with the initial surgical scene at t=0 upon incision, seeding the initial entity embeddings $E_0$.
- **Why it exists:** Although rigid registration degrades post-incision, pre-operative data provides the initial geometric and semantic priors (e.g., hidden tumor boundaries) before they can be physically observed. Intraoperative imaging modalities (iMRI, iUS) — when available — are admitted as supplementary observations through the same entity-binding pathway.
- **What can go wrong:** Initial registration error propagates into entity geometric attributes; large brain shift between MRI acquisition and incision degrades the prior more than expected.
- **Connection:** Seeds entity binding (§5.2) and the entity knowledge store (§5.3); influences but does not substitute for continuously updated learned residuals from the world model (§5.4).

### 5.8 Human-Machine Interaction (HMI) and Handover

- **What it does:** Provides the shared autonomy interface enabling the surgeon to inject Layer 1 constraints (workspace bounds, virtual fixtures) and behavior contracts $\mathcal{C}$, issue directives $\mathcal{D}$, and execute a graceful mechanical handover when the Phase 2 sync gate triggers a hard stop.
- **Why it exists:** Clinical safety at Levels 2–3 (§1.5) requires the operating surgeon to remain the ultimate authority. The architecture supports shared control without abrupt impedance discontinuities at transition.
- **Handover protocol:** Before automation begins a step, the system provides an explicit intent preview and requires surgeon acknowledgment. During take-over requests, the interface presents multimodal cues (visual + audible + haptic) and state-lock indicators to prevent authority ambiguity and reduce transients.
- **What can go wrong:** Handover discontinuities can introduce force transients; surgeon-injected directives may conflict with active contracts (conflict resolution policy must be specified).
- **Connection:** Source of Layer 1 spatial bounds for the safety evaluator (§5.6) and of directives consumed by the policy substrate (§5.5).

### 5.9 Hardware and Actuation Constraints

- **What it does:** Projects the policy's target trajectories into feasible joint-space commands $u_t$, explicitly handling kinematic singularities, payload limits, and physical actuation latency on physical platforms — KUKA KR 6-2 in the near term, neuroArm in the long-horizon Stage-5 program.
- **Why it exists:** The architecture must respect the physical envelope of the robot. Payload, reach, repeatability, and joint-rate limits constrain the achievable action set $\mathcal{A}$ and the worst-case latency available to the safety evaluator. The control-mode mapping between KUKA and neuroArm is a non-trivial deliverable (Stage 4 → Stage 5 gate).
- **What can go wrong:** Singularity handling can produce trajectory discontinuities; payload limits not surfaced to the policy substrate produce infeasible commands; clock drift between perception and actuation breaks alignment (≤ 1 ms drift requirement).
- **Connection:** Sits between the fast controller (§5.5 outputs) and the robot hardware, downstream of the sync command gate (§5.6).

### 5.10 Data Provenance and Pipeline

- **What it does:** Manages the secure, HIPAA-compliant lifecycle of surgical data from edge inference to cloud consolidation, including automated sanitization and de-identification of video and telemetry. The full data lifecycle, manifest schema, RBAC model, and PHI-gating are specified by the system design (see §5.10).
- **Why it exists:** Offline consolidation of episodic fast weights into cross-case slow weights (§5.3) requires large-scale aggregation across cases and centers, which mandates a robust, auditable, consent-conformant data pipeline. Training-from-archived-video at all stages inherits these constraints.
- **What can go wrong:** PHI leakage in derived features; consent boundary violations when consolidating across institutions; insufficient data volume to support cross-case priors.
- **Connection:** Feeds the offline consolidation pipeline (§5.3) and the IDM/policy training pipelines (§5.5).

## 6. Cross-Cutting Trade-offs and Rejected Patterns

The architecture makes three deliberate trade-offs that drive the rest of the design. Each is a research bet, not a settled position; the §7 evaluation logic and the §10 limitations are the conditions under which the bets would be revised.

- **Data over code.** We accept higher data requirements and harder debugging for the policy substrate in exchange for compositional generalization that fixed skill libraries cannot achieve. The data cost is real; §4.3 surfaces the unquantified data contract as a known gap.
- **Continuous over discrete.** We accept the cost of reasoning over continuous embeddings rather than typed scene graphs to capture surgical-tissue nuance. The cost is debuggability and human-readability; partial mitigation is provided by per-entity decoders that produce inspectable readouts (§5.2).
- **Separated safety.** We accept the engineering overhead of a separate safety evaluator with two distinct guarantee classes (formal verification at Layer 1, conformal calibration at Layer 2) so that task optimization cannot compromise clinical safety in the verified layer. The cost is that Layer 2 inherits the world-model's failure modes (§5.6, §10.2).

## 7. Evaluation and Validation Logic

The thesis is structured to be falsifiable both at the level of individual load-bearing assumptions and at the level of its central claim. Concrete acceptance gates, time budgets, and probability estimates are specified in the companion validation; this section states the validation logic.

### 7.1 Component-level falsifiable claims

The evaluation protocol is architecture-level and module-specific: each load-bearing claim is tied to an ablation hypothesis and a stopping condition rather than a post-hoc optimization result.

1. **IDM transfer accuracy.** *Claim:* Pseudo-actions (Δ tool pose, gripper state) extracted by an inverse-dynamics model trained in SOFA simulation are sufficiently aligned with expert surgeon kinematics to support behavioral cloning. *Validation:* Per-frame RMSE on Δ translation (mm), Δ rotation (rad), and gripper state, and per-sequence DTW distance, against ground-truth actions on robot-collected surgical sessions; concrete thresholds in §3.1. *Falsification:* RMSE > 4× inter-surgeon variance baseline or correlation < 0.3 — see the "weak" gate in §3.1.
2. **Foundation-model surgical specificity.** *Claim:* DINOv2 and V-JEPA 2 features, lightly fine-tuned on surgical video, support stable variable-cardinality entity binding and downstream task heads. *Validation:* Tracking stability and segmentation quality on surgical benchmarks such as SurgT, EndoVis, Cholec80, and CATARACTS. **Threshold caveat.** The expectation we work toward is *EAO ≥ 0.6 and DSC ≥ 0.85* on a defined evaluation suite; these numbers should be read as engineering targets adopted from common practice on adjacent surgical benchmarks, not as claims about clinical outcomes. Linking these targets rigorously to clinical endpoints (e.g., margin-safe resection rate) requires a formal outcome study and is explicitly out of scope for this thesis. *Falsification:* Sustained EAO < 0.4 or DSC < 0.7 after the fine-tuning protocol.
3. **Compositional generalization.** *Claim:* The policy substrate executes novel combinations of language directives that are unseen as combinations during training, on a defined directive grammar. *Validation:* Zero-shot success rate on held-out directive combinations in simulation and on phantoms (FLS peg transfer, JIGSAWS-style suturing, then neurosurgical phantom tasks). *Falsification:* Held-out compositional success indistinguishable from per-directive memorization; the policy's compositional generalization curve fails to rise above a behavioral-cloning baseline trained on the same data without language conditioning.
4. **Safety evaluator reliability.** *Claim (Layer 1):* Layer 1 invariants admit a formal verification proof (workspace bounds, joint limits, force ceilings, deadline-bounded stop). *Validation:* Theorem-prover or model-checker proof against the published predicate set; surrogate empirical validation by adversarial-policy fuzzing. *Claim (Layer 2):* The conformally calibrated SafetySurface attains its nominal coverage on a held-out calibration distribution, with quantified degradation under defined distribution shift. *Validation:* Empirical coverage vs. nominal, per §4.5. *Falsification:* Verified failure of any Layer 1 predicate under the published threat model; or empirical Layer 2 coverage substantially below nominal under in-distribution evaluation. The thesis explicitly **does not** claim "zero safety violations" as an absolute; it claims (a) zero observed violations of Layer 1 invariants under the specified adversarial protocol, and (b) statistical coverage at a stated confidence on Layer 2.

### 7.2 Falsification protocol and stage gates

Component claims in this thesis are treated as load-bearing until disproven by pre-registered ablations. For each claim:

- We define an explicit falsifiable hypothesis with a clinically meaningful MCID (for example TRE, force-threshold violations, insertion error, or compositional failure rate).
- We run ablations by toggling or replacing one module at a time while holding all non-target components constant.
- We apply transparent stopping rules (efficacy, futility, safety) to avoid post-hoc interpretation and to prevent repeated-risk testing without clear evidence gain.
- If Layer 1 cannot be preserved under a hypothesis at the specified risk bound, downstream evidence is blocked and the architectural decision is revisited.

This converts non-improvements into strong negative evidence instead of anecdote and keeps the thesis behaviorally falsifiable throughout staging.

### 7.3 Falsifiability

The central claim of §2 is that **structured learned control** outperforms the cheapest baseline in the §3.8 table that achieves the same observable behavior. The thesis is falsified at the architectural level if any of the following obtains:

- A monolithic VLA fine-tuned on surgical data, augmented with an external safety-projection layer functionally equivalent to §5.6, achieves comparable directive-level success and clinical-safety metrics on the evaluation suite of §7.1 — at which point the structural separation argued for in §2 is empirically equivalent to the baseline.
- A skill-library autonomy stack matches the proposed system's compositional generalization and OOD handling on a defined neurosurgical task suite within a fixed engineering budget. (This is the §3.8 row #2 falsifier.)
- Pseudo-action extraction (the §7.1 claim 1) reaches the "weak" gate of §3.1 and no alternative supervision route (e.g., contract-based RL in sim, action-conditioned VLA on surgical video) recovers the lost training signal — in which case the imitation-from-IDM pillar of §2 is broken and the architecture must be redesigned.

### 7.4 Evaluation-suite alignment with prior benchmarks

Where possible, the validation harnesses align with public surgical datasets and tasks (FLS peg transfer, JIGSAWS suturing, Cholec80 phase recognition, CATARACTS, EndoVis tool tracking, SurgT soft-tissue tracking) so that component-level performance is comparable to the wider community's baselines. Neurosurgery-specific tasks beyond these are considered novel and require purpose-built phantom or cadaveric protocols whose specification is part of the Stage-4 / Stage-5 deliverable (§8).

## 8. Research Roadmap

This roadmap sequences the scientific agenda, detailing which bets must be validated first and what evidence unlocks the next layer. It is a staged scientific program, not a software implementation plan. Per-stage acceptance gates and probability estimates are specified in §3 and §5; per-stage training schedules are described in the roadmap itself. The 9-month core window below is intentionally aggressive and assumes agent-accelerated implementation, continuous funding, surgical-data access, and IRB approvals at the corresponding stages.

### Stage 1 — Foundational Validation and Representation Learning (Months 0–2)

- **Objective:** Validate the IDM transfer bet (the §1 load-bearing assumption) and establish continuous entity representation.
- **Milestones:** train IDM in SOFA and extract pseudo-actions from mono surgical video; fine-tune DINOv2 / V-JEPA 2 on surgical video and implement variable-cardinality slot attention.
- **Evidence threshold:** IDM transfer at least at the "moderate" gate (§3.1); slot attention temporal coherence per §3.2.
- **Stop condition:** IDM transfer at the "weak" gate halts downstream work pending architecture redesign.

### Stage 2 — World-Model Reliability and Fast Control (Months 2–4)

- **Objective:** Establish the embedding-conditioned world model (DINO-WM) and the SOFA-trained forward model used by the safety evaluator.
- **Milestones:** train DINO-WM on surgical video transitions; train the fast forward model for physical-implausibility detection in SOFA; pre-register comparator world-model ablations (Dreamer V3, V-JEPA 2 as WM) for execution if Stage-2 evidence is weak.
- **Evidence threshold:** DINO-WM short-horizon prediction quality on tissue deformation and tool-tissue contact at a defined task suite; forward model false-positive rate < 10% on routine surgical motion at the threshold that catches injected synthetic anomalies (§3.5).

### Stage 3 — Policy Substrate Capability and Safety Calibration (Months 4–6)

- **Objective:** Train the compositional policy substrate via behavioral cloning over IDM pseudo-actions and calibrate the two-phase safety evaluator.
- **Milestones:** BC training of the policy substrate; implementation of Layer 1 invariants (verified) and Layer 2 conformal calibration; publication of a safety-calibration protocol (currently a deliverable per §4.5).
- **Evidence threshold:** Compositional generalization above a directive-conditioned BC baseline on FLS-peg-transfer and JIGSAWS-style suturing (in simulation). Layer 1 verified; Layer 2 nominal coverage met on held-out calibration set. **No "zero safety violations" claim is made absolutely**; the gate is *no observed Layer 1 violation* under the specified adversarial protocol and *empirical Layer 2 coverage at the stated nominal level*.

### Stage 4 — Sim-to-Real Transfer and Deformable Dynamics (Months 6–9)

- **Objective:** Bridge to physical hardware (KUKA KR 6-2 envelope) and handle soft-tissue interactions, with multimodal time-sync (≤ 1 ms drift).
- **Milestones:** deploy on KUKA via the unified `Environment` interface; fine-tune components with real observations; validate suturing and resection directive language on physical phantoms.
- **Evidence threshold:** Successful execution of multi-step directives on phantoms; world model accurate on real tissue proxies; no Layer 1 safety violations on phantom protocol.

### Stage 5 — Clinical Progression and Procedural Autonomy (Post-9-Month Extension)

- **Objective:** Advance to cadaveric validation and Level-3-style procedural autonomy on neuroArm; activate offline memory consolidation and generative replay. This is a follow-on extension beyond the 9-month thesis window.
- **Milestones:** port `Environment` to neuroArm; offline consolidation pipeline live with PHI-compliant data handling; cadaveric narrow-task autonomy under IRB.
- **Evidence threshold:** handling of real anatomical variation, registration errors, and procedural directives at clinically auditable safety and efficacy.

**Roadmap honesty.** The current 9-month probability of Stage-3 completion is moderate only if the agent pipeline and data access stay unblocked; the 9-month probability of Stage-4 KUKA transfer is lower; Stage 5 is intentionally excluded from the core thesis window and is reserved for a follow-on program. See §6 for explicit probability estimates. This thesis publishes the architecture and the validation logic; outcomes from the staged program will be reported in subsequent papers.

## 9. Open Questions and Research Bets

The architecture relies on several open research bets that are tracked, but not yet resolved, by the staged program:

- **Pseudo-action quality at fine temporal scale.** What degree of temporal smoothing is required to make behavioral cloning stable from IDM pseudo-actions? (§2.4)
- **Streaming variable-cardinality entity binding.** Can slot attention reliably maintain entity identities without over-segmentation in a dynamic, bleeding surgical field, particularly across cautery-smoke occlusions? (§2.2)
- **Hopfield capacity at case length.** Can modern Hopfield networks store and recall case-long associative memories without catastrophic forgetting or latency spikes? (§2.8, §3.4)
- **Reward signal beyond imitation.** How does the system transition from imitation to RL (contract-based rewards) for behaviors imitation cannot reach? (§4.1)
- **Conformal coverage under distribution shift.** Layer 2 coverage guarantees in non-stationary surgical scenes — what is the calibration-set composition, sliding-window protocol, and degradation profile under defined shift? (§4.5, deliverable: safety-calibration protocol)
- **Cross-component gradient flow.** Are modules trained jointly, in stages, or end-to-end fine-tuned? Stability tradeoffs are unspecified. (§4.4)
- **Data contract.** How many hours of surgical video, across how many procedures and centers, are required for the policy substrate to reach Stage-3 evidence thresholds? (§4.3)

### 9.8 Standards-driven evaluation discipline

Research findings now require evaluation discipline for every ablation. The thesis explicitly ties architectural decisions to:

- Risk-aware ablation protocols with pre-specified hypotheses and MCID thresholds.
- Benchmarks and task suites that include phantoms, simulation stressors, and metrics aligned to neurosurgical failure modes.
- Publication-ready documentation standards (ARRIVE, SPIRIT-AI, CONSORT-AI, and DECIDE-AI where applicable), so negative findings (non-necessity of a module) are also treated as valid disproof outcomes.

## 10. Limitations and Threats to Validity

A PhD-grade theoretical paper is obliged to state the conditions under which the proposed architecture would fail or be inappropriate. The following are explicit, not exhaustive.

### 10.1 Limitations of scope

- **Autonomy ceiling.** The architecture targets Levels 2–3 (§1.5). It is not a position on Levels 4–5 surgical autonomy. Decisions about clinical judgment, patient triage, or non-procedural reasoning are out of scope.
- **Procedure scope.** Validation is sequenced through generic surgical tasks (FLS, JIGSAWS) before any neurosurgical task. Initial neurosurgical evaluation is on phantoms; cadaveric and any in-human work is gated on IRB approval and is not assumed to occur on the timeline of §8.
- **No clinical-outcome claim.** The thesis does not claim that achieving the §7 component thresholds yields improved patient outcomes. Linking technical metrics to clinical endpoints requires a formal outcome study that is out of scope here.

### 10.2 Threats to internal validity

- **IDM transfer is the single load-bearing bet.** If §3.1 returns the "weak" gate, the policy story collapses; no other component is sufficient to recover it without redesign.
- **Sim-to-real gap on tissue.** Soft-tissue compliance, tool-tissue contact, and bleeding dynamics in SOFA differ measurably from real tissue. The world model and forward model both inherit this gap; their utility on novel tissue is an empirical question (§2.5).
- **Distribution shift between training video and clinical surgery.** Available surgical video is selected (teaching cases, well-recorded). Real intraoperative variability is wider; OOD behavior of every learned component is a threat (§2.9).
- **Adversarial robustness is unspecified.** Adversarial inputs (corrupted video frames, sensor noise) are not part of the current safety threat model. They must be added before any clinical claim.
- **Cumulative pseudo-action drift in BC.** Errors compound over long horizons; this bounds policy quality even if IDM transfer is moderate (§2.4).

### 10.3 Threats to external validity

- **Single-platform development.** Stage 1–4 validation is on KUKA-class hardware; neuroArm port is Stage 5. Generalization across surgical robots is not demonstrated, only argued for via the unified `Environment` interface.
- **Single-center data.** Without a multi-center data plan, claims about generalization to surgical-style variation are weak.
- **Surgical specialty.** Decisions calibrated for neurosurgery may not transfer to other specialties without revalidation.

### 10.4 Falsifiability of the thesis as a whole

The thesis is falsified at the architectural level if any one of the conditions in §7.2 obtains. In particular, if a monolithic VLA fine-tuned on surgical data, augmented with an external safety-projection layer functionally equivalent to §5.6, achieves comparable directive-level success and clinical-safety metrics on the §7.1 evaluation suite, then the *structural* separation argued for in §2 is empirically equivalent to the baseline and the central claim is falsified — even though the resulting end-to-end behavior may still be useful.

### 10.5 Open architectural gaps acknowledged in companion docs

The validation companion (§4) explicitly enumerates current gaps that this thesis inherits: the absence of an in-architecture reward signal until late stages (§4.1), the absence of a per-phase evaluation framework that this document partly addresses (§4.2), the absence of a quantified data contract (§4.3), the underspecification of cross-component gradient flow (§4.4), the conformal-calibration protocol gap (§4.5), and the deferred surgeon-language directive interface (§4.6). These are not solved here; they are surfaced so that reviewers can assess what part of the architecture is *load-bearing yet unspecified*.

## 11. Conclusion

This thesis proposes a **structured learned control** paradigm for autonomous neurosurgery, positioned at Levels 2–3 of the Yang-et-al. autonomy taxonomy. It rejects three dominant alternatives — fixed skill libraries, monolithic VLA models, and FEM-only biomechanical compensation — on operational grounds, and engages each with named prior systems and explicit naive baselines (§3, §3.8). It commits to four conceptual pillars (§2): embedding-first state, multi-timescale memory, imitation via inverse dynamics, and the separation of soft learned reasoning from an engineered, partly formally verified safety boundary. It states component-level falsifiable claims (§7.1), a thesis-level falsification condition (§7.2, §10.4), and a staged research program (§8) whose acceptance gates are inherited from the companion validation logic. It does **not** claim experimental results: the architecture is a hypothesis, not a system. The contribution is a self-consistent, falsifiable starting point for a multi-year research program; the program's outcomes — successful or otherwise — will be reported in subsequent papers grounded in the validation logic specified here.