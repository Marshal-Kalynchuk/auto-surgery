# Architecture Thesis Deep Research Findings (Parallel)

This file consolidates the deep-research outputs generated for `docs/library/ARCHITECTURE_THESIS.md` into one digestible overview.

## Source artifacts

- `modular-classical-stack-vs-end-to-end.json`
- `layered-safety-risk-scoring.json`
- `object-centric-perception-surgical-occlusion.json`
- `deformable-world-models.json`
- `memory-architectures-hops-kv.json`
- `multi-loop-control-frequency.json`
- `inverse-dynamics-imitation-learning.json`
- `embedding-first-entity-state.json`
- `evaluation-falsifiability-surgical-robotics.json`
- `data-governance-phi.json`
- `surgical-hmi-handover-autonomy.json`
- `hardware-actuation-constraints.json`

## 1) Modular vs classical stack

**Finding:** Classical neurosurgical stacks are brittle under brain shift and unmodeled dynamics because they are often registration-centric and state-machine-like. Modular perception-planning-control pipelines are consistently better positioned to adapt continuously via real-time sensing and replanning.

**Implication for thesis:** The design emphasis should remain on modular replacement boundaries and adaptive perception/plan feedback, not rigid preoperative-task scripting.

## 2) Layered safety and risk scoring

**Finding:** A two-layer safety architecture is strongly supported: hard safety invariants in Layer-1 plus runtime risk gating in Layer-2. This combination improves auditability, emergency-stop semantics, and override clarity.

**Implication for thesis:** Keep deterministic safety behavior at the lowest layer and treat adaptive autonomy as risk-gated policy, especially for surgical settings with strict traceability demands.

## 3) Object-centric perception under occlusion

**Finding:** Slot/semantic object-centric representations give interpretable scene structure, while memory-aware tracking is more robust during long occlusion and re-entry. In high-artefact neurosurgical scenes, hybridization is favored.

**Implication for thesis:** Combine a semantic front-end for structure with short/medium-term identity memory to reduce tool/tissue track instability from bleeding, smoke, and overlap.

## 4) Deformable world models

**Finding:** No single public benchmark currently unifies DINO-WM, Dreamer, JEPA, and retrieval baselines on neurosurgical deformation + topology + fluid dynamics with standardized compute-cost reporting.

**Implication for thesis:** Treat DINO-WM evaluation as a staged research protocol decision rather than a final benchmark claim; focus on defining a thesis-specific stress suite with explicit error modes (deformation, topology change, fluid dynamics).

## 5) Memory architectures (fast/slow + MHN + key-value)

**Finding:** Single memory options have clear trade-offs: fast/slow weights are low latency but shallow retention; modern Hopfield-style associative memory has strong recall properties but scales in latency; vector/key-value stores scale long-horizon memory with strong recall and low query latency if indexed well.

**Implication for thesis:** A tiered memory stack is the most defensible architecture for multi-hour cases under sub-200 ms control safety budgets.

## 6) Multi-loop frequency design

**Finding:** Robust systems separate planning and servo loops, with servo at much higher frequency than planning, and hardware loops at the highest deterministic rates. Safety interlocks should sit at the fastest, most verifiable layer.

**Implication for thesis:** Design loop rates explicitly as part of interfaces between modules, with end-to-end latency budgets carried through every interface contract.

## 7) Imitation learning through inverse dynamics

**Finding:** IDM-based pseudo-action extraction from video enables Behavioral Cloning from Observation (BCO) where labeled actions are missing, especially useful in historical/phased data where only video exists. Safety filters are essential for deployment.

**Implication for thesis:** Prioritize IDM data pipelines for bootstrapping policy supervision, then couple with safety wrappers (e.g., CBF/QP filters) before any real-world control validation.

## 8) Embedding-first state + discrete safety querying

**Finding:** Continuous embeddings support stronger identity persistence under occlusion/pose variation, but explicit discrete semantic labels remain necessary for auditability and safety predicates.

**Implication for thesis:** Use a hybrid: embeddings for tracking and uncertainty-aware identity, discrete scene-graph or ontology-backed structure for constraint specification and human-readable rules.

## 9) Evaluation and falsifiability

**Finding:** Publication-ready rigor is strongest when studies define architecture-level ablations with pre-registered stopping rules, explicit MIDs/MCIDs, and benchmark alignment to avoid over-claiming.

**Implication for thesis:** Architect evaluations so negative findings are informative and falsify component necessity claims rather than being treated as incidental failures.

## 10) Data governance and compliance

**Finding:** A production-safe governance posture needs explicit consent handling, de-identification strategy, security, retention policy, and cross-center sharing plans (often via federated/secure aggregation patterns).

**Implication for thesis:** Keep governance artifacts versioned alongside experimental outputs; they should be explicit, testable, and referenced by protocol documents.

## 11) HMI and handover authority

**Finding:** Safe autonomy transitions depend on unambiguous surgeon authority and handover states with intuitive multimodal cues. Without clear authority semantics, automation can create delay and confusion.

**Implication for thesis:** Treat HMI as a first-class safety module, with explicit control-transition signaling, state clarity, and hard/soft stop semantics.

## 12) Hardware and actuation constraints

**Finding:** Hard constraints are best reserved for limits with direct safety impact (workspace, force, collision, actuator limits, kinematics), while learned priors should only tune behavior inside those guardrails.

**Implication for thesis:** Enforce hardware limits as hard constraints in the control layer and use residual/learning-based terms only for compensation and performance shaping.

## Cross-cutting recommendations

1. Use a layered, compositional architecture:
   - continuous embedding/identity layer
   - graph/symbolic safety-query layer
   - fast memory + deliberative memory tiers
   - multi-loop control with explicit timing contracts
2. Reframe every major module with an ablation hypothesis and a falsifiable stopping rule before implementation.
3. Keep governance and safety evidence paired with technical claims in the same section-level artifact.
