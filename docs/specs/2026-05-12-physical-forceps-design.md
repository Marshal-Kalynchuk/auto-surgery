# Physical Forceps in SOFA — Design

| Field | Value |
|---|---|
| Date | 2026-05-12 |
| Status | Approved (design); implementation pending |
| Piece | 2 of 5 (in the simulation-pipeline redesign) |
| Depends on | `docs/specs/2026-05-12-command-schema-and-env-step-contract-design.md` (piece 1) |
| Supersedes | `src/auto_surgery/env/sofa_scenes/forceps.py` (visual-only factory) and `build_forceps_action_applier` in `src/auto_surgery/env/sofa_tools.py` (`OglModel.translation` setter) |
| Backward compatibility | **None.** Piece 1 is the new floor; this piece consumes its `RobotCommand` / `SensorBundle` directly. |

---

## 1. Context and scope

Piece 1 made `RobotCommand(CARTESIAN_TWIST in CAMERA)` the action a real robot, an IDM, and the SOFA env all speak. That command is meaningless until the forceps in SOFA is something a twist can actually act on. Today the forceps is a single `OglModel` whose `translation` field is mutated each tick — it has no rigid-body state, no collision geometry, no jaw articulation, and does not deform the brain when it touches it.

Piece 2 replaces the visual-only forceps with a **physically simulated rigid tool** that:

1. Is driven by camera-frame `Twist` commands and emits the `ToolState` fields piece 1 declared (`pose`, `twist`, `jaw`, `wrench`, `in_contact`).
2. Collides with and deforms the brain Tet-FEM tissue via the existing penalty contact pipeline.
3. Renders as a recognisable surgical forceps (DejaVu shaft + two visible jaws) so the rendered video is plausible IDM training material.
4. Exposes `tool_jaw_target` as a per-tick scalar that visibly opens and closes the jaws — sufficient to train a future `ForcepsJawIDM` actuation head from the same simulated clips.

This is **piece 2 of 5**:

1. Command schema + environment-step contract *(piece 1, complete)*
2. Physical forceps in SOFA *(this spec)*
3. Human-like, scene-aware motion generator
4. Domain randomization framework
5. Recording pipeline

A later piece — provisionally **piece 6**, gated on a policy training run that emits jaw-closure as a control acting on tissue — adds physically articulated jaws and grasping constraints. Piece 2 deliberately does not block on it.

---

## 2. Goals

1. Produce a single `Rigid3d` forceps body that consumes `CARTESIAN_TWIST in CAMERA` commands and reports a `ToolState` consistent with what the camera sees.
2. Deform the existing brain Tet-FEM mesh through penalty contact when the tool tip presses into tissue. Reuse the scene's existing `FreeMotionAnimationLoop` + `MinProximityIntersection` + `PenaltyContactForceField` stack — no new collision plugin.
3. Render the forceps using the real DejaVu meshes (`body_uv.obj`, `clasper1_uv.obj`, `clasper2_uv.obj`), with visible jaw open/close driven by `tool_jaw_target`.
4. Produce a typed `ToolSpec` field on `SceneConfig` so future tool catalog growth (DR-driven mesh swaps, additional tool types) is additive — a new yaml + a new literal in the union.
5. Keep physics math (twist transforms, hinge geometry) in pure functions outside SOFA; the action applier and observer are thin orchestrators over those pure functions.
6. Provide a one-time, deterministic, committed asset-prep step that produces decimated collision proxies from the DejaVu visuals.

## 3. Non-goals

1. **Articulated jaw dynamics, grasping, or per-jaw contact wrench.** Jaws are kinematic visuals in piece 2. Tracked as piece 6, triggered by a policy that emits jaw-closure as a control acting on tissue.
2. **Constraint-based contact response** (LCP / `GenericConstraintSolver`). The existing penalty pipeline is sufficient for IDM training video. Re-opens only if penalty interpenetration becomes visibly problematic.
3. **Multiple tool variants in v1.** `ToolSpec.tool_id` is a `Literal["dejavu_forceps"]` in v1, widening only when a downstream training run needs additional geometric variation that appearance DR can't cover.
4. **Sub-stepped FEM integration.** Couple sim-tick to control-tick at piece 1's default 250 Hz; re-open if instability appears.
5. **Force / impedance control of the tool.** Piece 1 explicitly omits `CARTESIAN_FORCE`; the tool body is velocity-source, not force-controlled.
6. **Mesh swap / domain randomization** of the forceps appearance. `VisualOverrides` is a piece-2 hook with no piece-2 sampler; piece 4 fills it in.

---

## 4. Foundational design decisions

### 4.1 Velocity-source rigid shaft

The shaft is a single `MechanicalObject` of `template="Rigid3d"`. Each tick the action applier writes the shaft's 6-DoF **velocity** DOF to the commanded twist (transformed from camera frame to scene frame and shifted from tool-tip frame to shaft-origin frame). SOFA integrates this over `dt ≈ 4 ms` during `animate()`. Penalty contact may slightly deflect the integrated pose — this is the intended behaviour and is what makes the rendered video physically plausible.

Why velocity-source over kinematic (write pose directly) or compliant (mass + spring/damper toward target):

- **Kinematic** would set pose exactly each tick and let tissue deform around it; this makes the tool tunnel through hard structures and would make the action-video correspondence too clean to be realistic.
- **Compliant** would let the tool drift off-command under contact, breaking the action-observation correspondence the IDM relies on.
- **Velocity-source** sits between: the tool *mostly* tracks command but visibly slows/deflects when pressed hard, matching real surgical video. Action labels stay close enough to observed motion to keep IDM training clean.

### 4.2 Visual-only jaws

The two DejaVu claspers (`clasper1_uv.obj`, `clasper2_uv.obj`) are loaded as separate `OglModel` children of the shaft body, parented via `RigidMapping`. Each tick the applier writes a **per-clasper transform** onto each visual model — a rotation about the hinge axis by the lerped jaw angle — without giving the jaws their own `MechanicalObject` or collision model.

Justification:

- The base IDM the simulator trains is **tool-agnostic tip-twist** (decision from piece 1). Jaw state is not part of its action space. Tool-specific actuation (jaw open/close, blade angle, cautery state) lives in **separate tool-conditioned actuation heads** trained on top — these heads learn from *visible* jaw motion, not from physical jaw-tissue interaction.
- Therefore the simulation only needs the jaws to *look* like they open and close. The simplified design provides that for free with a per-tick visual transform.
- Per-jaw contact wrench, simulated grasping, and tissue resistance to closure are all properties needed by a *policy* that commands jaw closure on tissue. They are not properties needed by an *IDM head* observing jaw motion in video. The simpler design therefore does not block any IDM training path.

When the policy training stack needs grasping consequences (piece 6's trigger condition), the visual jaws are replaced by articulated rigid sub-bodies. Until then, the additional complexity has no caller.

### 4.3 Real DejaVu meshes + decimated collision proxy

The forceps geometry is the DejaVu surgical-tool asset, already shipped as three separate meshes (`body_uv.obj` 73k tris, `clasper1_uv.obj` 16k tris, `clasper2_uv.obj` 16k tris) with matching textures (`instru.png`, `instru_clasper.png`).

The visuals load the full-resolution meshes. **The collision model uses a decimated proxy** for the tool-tip end of the shaft — the ~20% of the shaft's z extent closest to where the claspers attach (this is what actually contacts tissue) — target ~500 triangles, produced once by an asset-prep script and committed under `assets/forceps/`. The remaining ~80% of the shaft (the handle / arm) has no collision proxy; it is visual-only.

Justification:

- 73k-tri collision against a Tet-FEM brain is unworkable; ~500 tris is well within SOFA's comfort zone for `MinProximityIntersection`.
- The upper shaft is held above the operating field in surgical clips and rarely contacts tissue. Cutting it from the collision proxy is free realism, free compute.
- Real assets give photorealism the IDM benefits from; procedurally generated tools would look fake and would defeat the point of training on internet surgical video.
- Asset prep is deterministic and committed (not generated at scene-load) so CI is reproducible and the proxy is reviewable.

### 4.4 Penalty contact, reusing the existing pipeline

No new collision components. The shaft's collision proxy joins the scene's existing `FreeMotionAnimationLoop` + `BruteForceBroadPhase` + `BVHNarrowPhase` + `MinProximityIntersection` + `PenaltyContactForceField` chain. The brain side of the contact is unchanged. Penalty stiffness for tool-tissue contact is tuned during the visual smoke test and persisted alongside the assembly params.

---

## 5. Architecture

### 5.1 Unit decomposition

Four new / rewritten units in `src/auto_surgery/env/` plus one offline script:

1. **`sofa_scenes/forceps_assets.py`** *(new)* — pure-Python loader. Knows the DejaVu asset paths and the repo-local collision proxy paths. Exposes a `ForcepsMeshSet` dataclass holding paths for shaft visual, shaft collision proxy, clasper visuals, and textures. No SOFA dependency; unit-testable.
2. **`sofa_scenes/forceps.py`** *(rewritten)* — SOFA scene factory. Takes a `ForcepsMeshSet`, a `ForcepsAssemblyParams`, and an initial pose. Builds the rigid-body sub-tree (Section 5.2). Returns a typed `ForcepsHandles` dataclass exposing exactly the SOFA objects the applier and observer need — no scene introspection by string name elsewhere.
3. **`sofa_tools.py`** *(replace `build_forceps_action_applier`)* — two factories:
   - `build_forceps_velocity_applier(handles, camera_pose_provider, initial_jaw_target)` — consumes `RobotCommand`, transforms `CARTESIAN_TWIST` camera→scene, writes shaft velocity DOF, writes per-clasper visual transforms. Handles `enable` / `cycle_id` gating. `initial_jaw_target` (from `ToolSpec.initial_jaw`) seeds the applier's `last_jaw_target` cache so the very first command with `tool_jaw_target=None` has a defined value to hold.
   - `build_forceps_observer(handles, jaw_target_ref)` — reads `pose`, `twist`, `jaw`, `wrench`, `in_contact` and returns a `ToolState`. `jaw_target_ref` is a closure that reads the applier's cached `last_jaw_target`.
4. **`schemas/scene.py`** *(new)* — `ToolSpec`, `VisualOverrides`, `SceneConfig` Pydantic models (Section 5.4).
5. **`scripts/prep_forceps_collision_meshes.py`** *(new, offline)* — one-time deterministic asset-prep tool. Reads the DejaVu OBJs and writes decimated collision proxies into `assets/forceps/`.

Pure-function helpers live alongside the applier and are unit-testable without SOFA:

- `_twist_camera_to_scene(twist_camera: Twist, T_scene_camera: Pose) -> np.ndarray  # (6,)`
- `_shaft_origin_twist_from_tip_twist(twist_tip_scene: np.ndarray, T_shaft_world: Pose, tool_tip_offset_local: np.ndarray) -> np.ndarray  # (6,)`
- `_clasper_visual_transform(assembly: ForcepsAssemblyParams, jaw_target: float, side: Literal["left", "right"]) -> np.ndarray  # 4×4 homogeneous`

### 5.2 SOFA component tree

Added under the existing root, beside the brain subtree:

```
root
├── (existing brain subtree — unchanged)
└── Forceps                                            # group node
    ├── EulerImplicitSolver                            # tool ODE solver (independent of brain)
    ├── CGLinearSolver  iterations=25 tolerance=1e-9 threshold=1e-9
    ├── Shaft                                          # rigid body
    │   ├── MechanicalObject  template="Rigid3d"
    │   │                     position=[x y z qx qy qz qw]  velocity=[0]*6
    │   ├── UniformMass       totalMass=0.05            # small; we velocity-write each tick
    │   ├── UncoupledConstraintCorrection               # cheap, fine with penalty contact
    │   ├── ShaftVisual                                # visual child
    │   │   ├── MeshOBJLoader        filename=${DEJAVU}/body_uv.obj
    │   │   ├── OglModel             texturename=${DEJAVU}/instru.png
    │   │   └── RigidMapping         input=@.. output=@.
    │   ├── ShaftCollision                             # collision child (tool-tip end only)
    │   │   ├── MeshOBJLoader        filename=${REPO}/assets/forceps/shaft_tip_collision.obj
    │   │   ├── MechanicalObject     template="Vec3d"
    │   │   ├── TriangleCollisionModel
    │   │   ├── LineCollisionModel
    │   │   ├── PointCollisionModel
    │   │   └── RigidMapping         input=@../.. output=@.
    │   ├── ClasperLeft                                # visual-only child
    │   │   ├── MeshOBJLoader        filename=${DEJAVU}/clasper1_uv.obj
    │   │   ├── OglModel             texturename=${DEJAVU}/instru_clasper.png
    │   │   └── RigidMapping         input=@../.. output=@.   # mapped through a Python-driven transform
    │   └── ClasperRight                               # visual-only child (mirror of ClasperLeft)
    │       ├── MeshOBJLoader        filename=${DEJAVU}/clasper2_uv.obj
    │       ├── OglModel             texturename=${DEJAVU}/instru_clasper.png
    │       └── RigidMapping         input=@../.. output=@.
```

Notes:

- The **Shaft body** is the only thing with an independent rigid DOF and collision geometry. Both claspers are rendered as `OglModel`s that follow the shaft; their per-jaw rotation is applied as a transform written by the action applier each tick (Section 6.2). They do not have their own `MechanicalObject` and do not participate in contact.
- The tool has **its own `EulerImplicitSolver` + `CGLinearSolver`**, separate from the brain's. SOFA propagates cross-subtree contact forces through the collision response layer regardless; decoupling the solvers means tool integration cannot stall the FEM solve and vice versa.
- `UncoupledConstraintCorrection` is required by `FreeMotionAnimationLoop` to project constraint corrections back; uncoupled is the right variant for our penalty-contact setup.
- `Triangle` + `Line` + `Point` collision models on the shaft tip — all three are needed for robust point-triangle, edge-edge, and point-point checks against the brain surface, which `MinProximityIntersection` requires.
- Mass `0.05 kg` for the shaft is small but non-zero. We velocity-write so absolute mass barely matters for tracking, but it must be non-zero for the rigid mapping and contact response to be well-defined.

### 5.3 `ForcepsHandles`

The scene factory returns a typed dataclass. The applier and observer take this handle — they never look up SOFA objects by string name.

```python
@dataclass(frozen=True)
class ForcepsHandles:
    shaft_mo: Any                          # MechanicalObject (Rigid3d) — the only rigid DOF
    shaft_collision_mos: tuple[Any, ...]   # Triangle, Line, Point collision models (for wrench/contact readout)
    clasper_left_visual: Any               # OglModel — receives per-tick visual transform
    clasper_right_visual: Any              # OglModel — receives per-tick visual transform
    assembly: ForcepsAssemblyParams        # mesh-derived constants (hinge axis, scale, etc.)
```

### 5.4 `SceneConfig` and `ToolSpec`

Piece 1 deferred `SceneConfig` shape to piece 2. The minimal v1 shape:

```python
class ToolSpec(BaseModel):
    """Tool definition. v1 supports DEJAVU_FORCEPS; future tools add new literals."""
    model_config = {"extra": "forbid"}
    tool_id: Literal["dejavu_forceps"]
    initial_pose_scene: Pose
    initial_jaw: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_overrides: VisualOverrides | None = None    # piece-4 hook

class VisualOverrides(BaseModel):
    """Per-episode visual perturbations applied by piece 4 (domain randomization)."""
    model_config = {"extra": "forbid"}
    body_tint_rgba: tuple[float, float, float, float] | None = None
    clasper_tint_rgba: tuple[float, float, float, float] | None = None
    # piece 4 may add lighting, specularity, etc.

class SceneConfig(BaseModel):
    model_config = {"extra": "forbid"}
    brain_scene_path: Path
    tool: ToolSpec
    camera_extrinsics_scene: Pose
    camera_intrinsics: CameraIntrinsics
```

Tool assembly constants (mesh paths, hinge origin, hinge axis, jaw angles, scale, penalty stiffness) are **properties of `tool_id`**, not of an episode. They live in `assets/forceps/dejavu_default.yaml` and are loaded by `forceps_assets.py` when the tool is instantiated.

This is what makes `tool_id` extensible: adding `dejavu_scissors` is a new yaml + a new literal in the union + (if needed) new actuation-head training, with no schema migration.

### 5.5 `ForcepsAssemblyParams`

Constants derived once from the DejaVu meshes by inspection, committed to yaml:

```python
@dataclass(frozen=True)
class ForcepsAssemblyParams:
    scale: float                           # uniform scale: DejaVu units → scene units
    hinge_origin_local: np.ndarray         # (3,) hinge center in shaft-local coords
    hinge_axis_local: np.ndarray           # (3,) unit hinge axis in shaft-local coords
    jaw_open_angle_rad: float              # clasper rotation at tool_jaw_target = 0.0
    jaw_closed_angle_rad: float            # clasper rotation at tool_jaw_target = 1.0
    tool_tip_offset_local: np.ndarray      # (3,) tool-tip position in shaft-local coords
    penalty_stiffness: float               # contact stiffness against brain FEM
```

Initial values (from mesh-bbox analysis, refined during the visual smoke test):

| Param | Initial value | Source |
|---|---|---|
| `scale` | TBD by smoke test | Match shaft length to brain wound scale |
| `hinge_origin_local` | `(0, 0, 0)` | DejaVu body bbox z ∈ [-88.9, 1.9]; tip end ≈ origin |
| `hinge_axis_local` | `(1, 0, 0)` | Claspers mirror across XZ plane in their local frames |
| `jaw_open_angle_rad` | `+0.30` | Visual inspection target |
| `jaw_closed_angle_rad` | `0.0` | Closed when claspers are coincident |
| `tool_tip_offset_local` | `(0, 0, ~+9.4 × scale)` | Clasper bbox z extent |
| `penalty_stiffness` | `1000.0` | Standard SOFA penalty value; tuned in smoke test |

---

## 6. Action applier semantics

The applier runs **once per control tick, before `animate()`**. One function call per `RobotCommand`. Five steps in order:

### 6.1 Safety gate (cycle_id + enable)

Identical contract to piece 1 §6.

- Validate `command.cycle_id > last_accepted_cycle_id`. If stale, write `SafetyStatus.command_blocked=True`, `block_reason="stale_cycle_id"`, **set shaft velocity to zero**, **hold last clasper transforms**, return.
- Validate `command.enable is True`. If disabled, same handling with `block_reason="disabled"`.
- Otherwise advance `last_accepted_cycle_id` and proceed.

Setting shaft velocity to zero on gate (rather than just blocking metadata) means the safety stop is **physically meaningful** in the rendered video, not just a flag in the manifest.

### 6.2 Camera-frame twist → shaft-origin twist

Pure-function pipeline:

1. `T_scene_camera = camera_pose_provider()` — read the scene's camera pose this tick.
2. `twist_scene_at_tool_tip = _twist_camera_to_scene(command.cartesian_twist, T_scene_camera)`
   — apply the 6×6 rigid adjoint $\mathrm{Ad}(T_{\text{scene}\leftarrow\text{camera}})$.
3. `T_shaft_world = read shaft_mo.position`
4. `twist_scene_at_shaft_origin = _shaft_origin_twist_from_tip_twist(twist_scene_at_tool_tip, T_shaft_world, assembly.tool_tip_offset_local)`
   — rigid shift from the tool-tip reference point to the shaft body's origin DOF (the angular part is unchanged; the linear part picks up `ω × r`).

The twist is **applied at the tool tip** because that is the surgically meaningful reference frame (the IDM learns video → tool-tip motion). The shift to the shaft body's origin is bookkeeping.

### 6.3 Write shaft velocity DOF

`shaft_mo.velocity = twist_scene_at_shaft_origin` (6-vec `[vx, vy, vz, wx, wy, wz]`).

### 6.4 Resolve jaw target and write clasper visual transforms

- Resolve `current_jaw = command.tool_jaw_target if command.tool_jaw_target is not None else last_jaw_target` (hold semantics from piece 1).
- Cache `last_jaw_target = current_jaw`.
- Compute `T_left = _clasper_visual_transform(assembly, current_jaw, "left")` (4×4 homogeneous in shaft-local frame).
- Compute `T_right = _clasper_visual_transform(assembly, current_jaw, "right")`.
- Decompose each 4×4 into `translation: (3,)` and `rotation: (3,)` (Euler XYZ in radians) — these are the data fields `OglModel` exposes for per-tick transforms — and write them onto the corresponding clasper `OglModel`. Scale stays at `(1, 1, 1)`.

The clasper transforms are written into the visual model directly — they bypass the rigid-mapping pipeline because the claspers do not have rigid DOFs. The `RigidMapping` to the shaft body remains; the visual transform composes on top of the rigid-mapped pose, so claspers track the shaft *and* open/close.

### 6.5 No mutation of brain state

The applier never touches brain DOFs. Tool-brain coupling happens through SOFA's collision response layer during `animate()`.

---

## 7. Observation extraction

`build_forceps_observer(handles, jaw_target_ref)` is called by the backend *after* `animate()` returns. It produces a `ToolState` (piece 1 schema). One source per field:

| `ToolState` field | Source |
|---|---|
| `pose: Pose` | `handles.shaft_mo.position` (7-vec) → `Pose` |
| `twist: Twist` | `handles.shaft_mo.velocity` (6-vec) → `Twist`. Reports the **integrated** velocity, which differs from the commanded twist when contact deflects the tool. This is what the camera sees. |
| `jaw: float` | `jaw_target_ref()` returns the applier's cached `last_jaw_target`. Kinematic visual, so commanded ≡ actual. |
| `wrench: Vec3` | `handles.shaft_mo.force` snapshot after `animate()`. Linear reaction force on the rigid DOF; equals net contact force in steady state. Piece 1's `wrench` is force-only (no moment), matching this. |
| `in_contact: bool` | `True` if the scene's `ContactManager` lists any contact whose collision MO is in `handles.shaft_collision_mos`. |

The observer is purely a query. It does not own state, does not advance time, and does not emit any other `SensorBundle` field — the camera image, timestamp, and `SafetyStatus` are built by other observer functions in the backend's per-tick sensor pipeline (piece 1's factoring).

---

## 8. Asset preparation

`scripts/prep_forceps_collision_meshes.py` — one-time deterministic tool, re-runnable, output committed to `assets/forceps/`.

Inputs:
- `${DEJAVU_ROOT}/scenes/liver/data/dv_tool/body_uv.obj` (73k tris)

Outputs:
- `assets/forceps/shaft_tip_collision.obj` (~500 tris, tool-tip end of the shaft by z extent in body-local frame)
- `assets/forceps/dejavu_default.yaml` — `ForcepsAssemblyParams` constants. **Not produced by this script.** Committed once by hand, edited by hand after smoke-test calibration. The script only produces the decimated `.obj`.

Procedure:

1. Load `body_uv.obj`.
2. Crop to the tool-tip end: keep only vertices (and the triangles they index) with `z > z_max - 0.2 * (z_max - z_min)` — i.e. the 20% of the z range closest to the claspers. For the canonical DejaVu body with `z ∈ [-88.9, 1.92]`, this keeps `z > -16.24` (≈ 18 units near the tip).
3. Decimate to target triangle count using quadric edge collapse (`pymeshlab.filter_meshing_decimation_quadric_edge_collapse` or equivalent).
4. Validate: closed surface, consistent normals, no degenerate triangles. If validation fails, raise — do not silently produce a broken proxy.
5. Write to `assets/forceps/shaft_tip_collision.obj`.

CI: the script runs in CI to verify reproducibility (the output should match the committed file byte-for-byte modulo trivial header differences). If the DejaVu input changes, CI flags the diff; the committed proxy is regenerated by hand.

---

## 9. Testing strategy

Four layers, ordered by speed.

### 9.1 Pure-function unit tests (no SOFA)

- `_twist_camera_to_scene`: canned camera poses + canned twists → expected scene twists.
- `_shaft_origin_twist_from_tip_twist`: rigid-shift correctness for non-trivial $(T_{\text{shaft}}, r)$ pairs.
- `_clasper_visual_transform`: lerp + hinge rotation at `jaw ∈ {0, 0.5, 1}` → expected per-clasper transforms; symmetry between left and right.
- `ForcepsAssemblyParams` yaml loader: valid configs parse, malformed configs raise.
- `ForcepsMeshSet` path resolver: returns existing paths; missing paths raise.

### 9.2 SOFA component tests (forceps subtree only, no brain, no rendering)

- Build the forceps via the scene factory in isolation; assert `ForcepsHandles` fields populated and typed.
- Step with zero twist → shaft pose stays within ε of initial after N ticks.
- Step with constant translational twist → shaft moves at commanded velocity within integration error.
- Step with `enable=False` → shaft velocity stays zero, claspers hold.
- Step with stale `cycle_id` → command rejected, `SafetyStatus.command_blocked=True`, motion blocked.
- Sweep `jaw_target ∈ {0, 0.5, 1}` → clasper visual transforms match expected angles, left/right symmetric.

### 9.3 Integration test (forceps + brain, no rendering)

- Build the full scene with the new forceps subtree.
- Drive the tool into brain tissue along a fixed twist for N ticks.
- Assert `in_contact: False → True` transition occurs.
- Assert `‖wrench‖` grows when pressing.
- Assert at least one brain tetrahedral DOF near the contact point deforms by more than a threshold.

### 9.4 Visual smoke test (slow, manual, not in CI by default)

- Run a short capture with a hand-authored twist sequence (move tip in, sweep, retract; intersperse jaw opens/closes).
- Inspect by eye.
- This is the **calibration acceptance gate** for `ForcepsAssemblyParams` — the values committed to `dejavu_default.yaml` are the values that produced a passing smoke test.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Mesh scale mismatch between DejaVu and brain scene.** | `scale` is the first thing the smoke test calibrates. Documented as a known calibration step in the asset-prep README. |
| **Penalty stiffness tuning.** Too soft → tool tunnels through brain; too stiff → instability. | Tuned in the smoke test, persisted in `dejavu_default.yaml`. Default `1000.0` is a sensible SOFA starting point. |
| **Decimation quality.** Quadric decimation can produce degenerate edges or non-manifold output that breaks `MinProximityIntersection`. | Asset-prep script validates output (closed surface, consistent normals). Fall back to a manually-authored proxy mesh if decimation fails on a future tool. |
| **Initial camera-pose ordering at reset.** Applier reads camera pose every tick, but the first call happens before any `animate()`. | Backend reset sequence places the camera node before constructing the applier; documented in the backend reset contract. |
| **Cross-subtree contact stiffness mismatch.** Tool subtree has its own solver; if penalty force is large, the tool's ODE may integrate it differently from the brain's. | Penalty contact computes force from geometry, not from solver state — both subtrees see the same contact force at the same instant. Solver-decoupling is safe with penalty; would not be safe with constraint-based contact (which is why we deferred that to piece 6). |

---

## 11. Deferred items with trigger conditions

Tracked verbatim so we don't lose them across pieces:

| Item | Trigger condition for re-opening |
|---|---|
| `ForcepsJawIDM` and other tool-specific actuation heads | A training run wants jaw / blade / cautery action labels — pure ML work, no sim change needed; this spec already emits `jaw_state` per tick. |
| Physical jaw simulation (articulated rigid jaws + grasping constraints) | Training a *policy* (not IDM) that emits jaw-closure as a control acting on tissue. **Piece 6.** |
| Per-jaw contact wrench in `ToolState` | An IDM head or policy explicitly consumes per-jaw force. |
| Tool catalog beyond `dejavu_forceps` | A training run needs geometric tool diversity beyond what appearance DR provides. New tool = new yaml + new literal in `ToolSpec.tool_id`. |
| Constraint-based contact response (LCP / `GenericConstraintSolver`) | Penalty interpenetration becomes visibly problematic in training data. |
| Sub-stepped FEM integration | 250 Hz control rate produces visible instability in tool-brain contact. |
| Tool segmentation (SAM 2) feeding tool-conditioned actuation heads | A downstream component explicitly needs per-pixel tool masks. `ARCHITECTURE.md` §11 defers this until load-bearing. |

---

## 12. Implementation completion checklist

Piece 2 lands when:

- [ ] `scripts/prep_forceps_collision_meshes.py` is committed and produces a deterministic `assets/forceps/shaft_tip_collision.obj`.
- [ ] `assets/forceps/dejavu_default.yaml` is committed with smoke-test-calibrated `ForcepsAssemblyParams`.
- [ ] `src/auto_surgery/env/sofa_scenes/forceps_assets.py` is implemented with full unit coverage.
- [ ] `src/auto_surgery/env/sofa_scenes/forceps.py` is rewritten to build the rigid-body subtree and returns `ForcepsHandles`.
- [ ] `src/auto_surgery/schemas/scene.py` defines `ToolSpec`, `VisualOverrides`, `SceneConfig`.
- [ ] `build_forceps_velocity_applier` and `build_forceps_observer` are implemented and replace `build_forceps_action_applier` in `sofa_tools.py`.
- [ ] `_NativeSofaBackend` wires the new applier and observer into reset / step.
- [ ] Pure-function unit tests pass.
- [ ] SOFA component tests pass.
- [ ] Integration test (forceps presses brain → contact + wrench + deformation) passes.
- [ ] Visual smoke test passes by eye; calibration committed.

---

## 13. Pointer to piece 3

Piece 2 provides a physically simulated forceps the simulation can command via tip-twist and observe via `ToolState`. **Piece 3 — Human-like, scene-aware motion generator** — replaces the current zero-default sine-wave action generator with a generator that emits plausible surgeon-like twist trajectories *informed by the scene state*: approach + dwell + retract + sweep patterns, anchored on contact events, with tuneable smoothness and frequency content matched to real surgical motion. Piece 3 reads from `ToolState` (notably `in_contact` and `pose`) to close the loop between scene state and emitted motion. Piece 3's design doc opens after this one is reviewed and committed.
