# Scene-Aware Motion Generator — Design

| Field | Value |
|---|---|
| Date | 2026-05-12 |
| Status | Approved (design); **implemented** with collapsed primitives (see §3.1). **SOFA path:** generator emits `CARTESIAN_POSE` in `SCENE` frame (`cartesian_pose_target`); SOFA applier uses pose-error servo (see `docs/UNITS.md`). Legacy `evaluate()` still exposes `twist_camera` for jaw / finish bookkeeping. |
| Piece | 3 of 5 (in the simulation-pipeline redesign) |
| Depends on | `docs/specs/2026-05-12-command-schema-and-env-step-contract-design.md` (piece 1), `docs/specs/2026-05-12-physical-forceps-design.md` (piece 2) |
| Supersedes | `src/auto_surgery/env/action_generators.py` (sine + random-walk joint generators) and the `_build_action` / `--joint-*` CLI in `src/auto_surgery/recording/brain_forceps.py` |
| Backward compatibility | **None** for command schema (piece 1). Motion YAML may still use legacy primitive-weight / duration key names; `MotionGeneratorConfig` normalizes them at load time (see `src/auto_surgery/schemas/motion.py`). |

---

## 1. Context and scope

Piece 1 made `RobotCommand(CARTESIAN_TWIST in CAMERA)` the action a real robot, an IDM, and the SOFA env all speak. Piece 2 made the forceps a physical rigid body that a twist can actually drive. Both are necessary; neither is sufficient. The remaining gap is the **source of those twist commands**: today the only generators are a joint-sine and a joint-random-walk, both of which emit `JOINT_POSITION` and neither of which reads any scene state. The resulting video is a forceps drifting on top of a static brain — not training material for an IDM that has to generalise to internet surgical footage.

Piece 3 replaces that with a **scene-aware motion generator** that:

1. Emits `RobotCommand(CARTESIAN_TWIST in CAMERA)` at the control rate (250 Hz default), one command per `env.step()`.
2. Produces surgeon-like trajectories: bounded-jerk velocity profiles, recognisable phases of motion (`Reach`, `Hold`, `Drag`, `Brush`, `Grip`, `ContactReach`), and visible jaw open/close coordinated with motion.
3. Reads `ToolState.in_contact`, `ToolState.pose`, and the camera extrinsics from the per-tick `StepResult` to react to what is actually happening in the scene — `Reach` early-terminates on contact, `Grip` handles contact-aware jaw handling, and `ContactReach` ends once contact is reliable.
4. Is parameterised by a typed `MotionGeneratorConfig` and a typed `SceneConfig.target_volumes`, both surfaced through pydantic YAML so the recorder CLI, piece 4's domain randomisation, and unit tests share the same configuration surface.
5. Records the realised primitive sequence (with parameters and contact-event timing) alongside the video/command stream, so piece 5's recorder can ship paired (motion-label, video) artefacts.

This is **piece 3 of 5**:

1. Command schema + environment-step contract *(piece 1, complete)*
2. Physical forceps in SOFA *(piece 2, approved; implementation pending)*
3. Human-like, scene-aware motion generator *(this spec)*
4. Domain randomization framework
5. Recording pipeline

---

## 2. Goals

1. Replace `src/auto_surgery/env/action_generators.py` (joint-only, sensor-blind) with a Cartesian-twist generator that consumes per-tick `StepResult` and emits `RobotCommand(CARTESIAN_TWIST in CAMERA)`.
2. Produce trajectories built from a small, typed set of motion primitives with minimum-jerk velocity profiles, so the rendered video shows recognisable surgical motion phases instead of constant-velocity drift.
3. Ground the trajectories in scene state via `SceneConfig.target_volumes` — small regions in scene frame that mark where the surgeon's tool is doing work — without coupling the generator to any SOFA internals.
4. Provide a per-episode `MotionGeneratorConfig` that exposes every randomisation knob as a typed pydantic field, so piece 4 can drive variation through the same surface that unit tests and CLI defaults use.
5. Keep the generator a pure-Python module: no SOFA dependency, no mesh access, no env protocol coupling beyond the typed `StepResult` / `SceneConfig` types defined in piece 1.
6. Ship the typed surface as multi-scene-ready (list-shaped `target_volumes`, generic `tissue_scene_path`) even though v1 only commits the brain scene's `SceneConfig`.

## 3. Non-goals

1. **Joint-space motion.** All emitted commands are `CARTESIAN_TWIST`. Joint-space is a piece-1 bridge concern.
2. **Stochastic-process motion priors** (OU / colored noise / DMPs). The primitive + minimum-jerk model produces enough variety from sampled parameters; spectrum-shaped noise is deferred until evidence of motion-monotony from IDM training.
3. **Multi-axis "surgeon style" vectors.** v1 randomises each primitive's parameters independently. A higher-order style sampler (slow vs fast, tremulous vs steady) is deferred until evidence the per-parameter distribution is too narrow.
4. **Hierarchical / task-level primitives.** No `ResectTumor` macro that composes sub-primitives. v1 is flat; the FSM walks one primitive at a time.
5. **Anatomy-aware target selection.** `TargetVolume.label` is shipped as `Literal["tumor", "vessel", "general"]`, but v1 uses uniform weighting across volumes regardless of label. Per-label biases are a piece-4 hook.
6. **Replay of a fixed motion plan across scene seeds.** Recorded manifests are introspection-only in v1.
7. **Per-tool primitive variants.** v1 ships one primitive set sized for `dejavu_forceps`. Tool-specific primitives (e.g., a `Cut` for scissors) are gated on the tool catalog expanding (piece 2 §11 trigger).

### 3.1 Implementation snapshot (2026-05)

Runtime code in `src/auto_surgery/motion/` uses a **collapsed** primitive set: `Reach`, `Hold`, `ContactReach`, `Grip`, `Drag`, `Brush` (see `src/auto_surgery/motion/primitives.py`).

| Original spec name | Implemented as |
|--------------------|----------------|
| `Approach` | `Reach` |
| `Dwell` | `Hold` |
| `Retract` | `Drag` (scene-mm distances) |
| `Sweep` / `Rotate` | `Brush` / `Grip` |
| `Probe` | `ContactReach` + `Grip` |

Canonical motion YAML keys are `weight_reach`, `reach_duration_range_s`, `hold_duration_range_s`, `drag_duration_range_s`, `brush_duration_range_s`, etc. (`configs/motion/default.yaml`). Legacy keys are still accepted at load time and normalized in `MotionGeneratorConfig`.

The FSM in `src/auto_surgery/motion/fsm.py` ends `Reach` early on a contact rising edge when `end_on_contact=True`; other primitives end on elapsed time only.

---

## 4. Foundational design decisions

### 4.1 Parametric primitives + contact-reactive FSM

Trajectories are built from a typed set of motion primitives, each a frozen dataclass with explicit parameters. A per-episode sequencer picks an ordered sequence of primitives at reset; a per-tick finite-state machine walks the sequence, advancing primitive-to-primitive on either time expiry or a primitive-specific sensor event (most commonly `tool.in_contact` rising edge).

Why primitives over stochastic process or spline waypoints:

- **Stochastic process** (OU / coloured noise tuned to surgical-motion spectrum) requires a recorded surgeon corpus to fit the spectrum to. We have no such corpus. The primitive approach uses hand-specified, well-understood velocity profiles that produce plausible motion without needing data.
- **Spline waypoint** planner pre-plans the entire episode, which makes contact-reactivity (`Reach` early-terminating, `ContactReach`/`Grip` contact handling) an awkward replanning step. Primitive FSM treats contact reactivity as a first-class transition.
- **Primitive FSM** matches the verbal model of surgical motion ("reach/hold/drag/brush/grip/contact"), with contact-anchored early termination where implemented. Each primitive is a pure function `(state, elapsed) → Twist` that is unit-testable without SOFA. The FSM is a separate unit, also unit-testable with synthetic `StepResult` streams.

### 4.2 Minimum-jerk velocity profile

Every primitive that animates over a duration uses the **minimum-jerk fifth-order polynomial** (Flash & Hogan 1985):

$$\mathrm{position}(\tau) = p_0 + (p_1 - p_0) \cdot (10\tau^3 - 15\tau^4 + 6\tau^5), \qquad \tau = t / T$$

$$\mathrm{velocity}(t) = (p_1 - p_0) \cdot \frac{30\tau^2 - 60\tau^3 + 30\tau^4}{T}$$

Boundary conditions: zero velocity at $\tau=0$ and $\tau=1$, peak velocity at $\tau=0.5$ equal to $1.875 \cdot \|p_1 - p_0\| / T$. The profile has bounded jerk, bounded acceleration, and zero start/end velocity — i.e., everything the rendered video needs to read as surgical motion rather than as a robot arm executing waypoints.

Why minimum-jerk over raised cosine, trapezoidal, or S-curve:

- It's the canonical model of human limb motion in the motor-control literature. Closest to what real surgeon footage shows.
- Closed-form polynomial. Analytical derivative gives the per-tick twist directly; no numerical differentiation, no per-tick integration of position-tracking error.
- Two-point boundary conditions (start pose, end pose, duration) — same uniform shape across every primitive that has a target pose sample. Brush/Grip uses derived phase timings in addition to those base profile constraints.
- Bounded jerk is what the IDM expects from real surgical footage. Trapezoidal velocity profiles produce visible "corners" in the rendered motion that would teach the IDM the wrong distribution.

### 4.3 `SceneConfig.target_volumes` (multi-scene-ready)

Piece 2 introduced `SceneConfig`. Piece 3 amends two of its fields:

- **Rename** `brain_scene_path → tissue_scene_path`. The current name is brain-specific; we want the typed surface ready for kidney, liver, uterus, and any other DejaVu scene without renaming twice.
- **Add** `target_volumes: list[TargetVolume]` (`min_length=1`). Each `TargetVolume` is a small region in scene frame (sphere or bbox) with a semantic label (`"tumor" | "vessel" | "general"`) that marks where surgical work happens. Tissue-seeking primitives (`Reach`, `ContactReach`, `Grip`) sample target points inside one of these volumes; non-tissue primitives (`Brush`, `Drag`) operate in camera frame and don't read the volumes.

v1 ships exactly the brain scene's `SceneConfig` with one `TargetVolume` (label `"general"`) covering the exposed cortex. Adding kidney/liver/uterus later is a pure config addition: a new YAML file, a new `tissue_scene_path`, and a new list of `target_volumes`. No code change. Piece 4's domain randomisation can perturb the `target_volume.center_scene` per episode.

The amendment to piece 2 lands in the same PR as piece 3's implementation. Piece-2 logic is unaffected; only the typed surface changes.

### 4.4 Stateful generator with `next_command(step_result)`

The public entry point is one class — `SurgicalMotionGenerator` — with `reset(initial_step) → RobotCommand` and `next_command(last_step) → RobotCommand`. The generator owns its FSM state, sequencer state, RNG state, and cycle-id counter. The recorder loop becomes a per-tick `cmd = gen.next_command(last_step); last_step = env.step(cmd)`.

Why stateful per-tick:

- Matches the recorder loop shape already in `recording/brain_forceps.py`. Zero recorder-side change beyond passing the prior `StepResult` into `next_command`.
- Contact-reactivity is first-class — the generator sees `tool.in_contact` on the tick that contact happened and can advance to the next primitive immediately.
- Lazy primitive selection: the sequencer picks the next primitive only when the FSM asks for one, so each primitive is parameterised against the *current* tool pose (after physics has acted on it) rather than against a pre-planned pose that may no longer be reachable.
- Trivial to unit-test: feed synthetic `StepResult` instances, assert emitted commands.

### 4.5 Jaw scripted by the same temporal model as motion

Each primitive optionally carries `jaw_target_start: float | None` and `jaw_target_end: float | None`. The **generator** interpolates between them using the **same minimum-jerk τ** that drives the primitive's motion and emits the resulting scalar as `RobotCommand.tool_jaw_target` on every tick. The piece-2 applier just consumes the scalar. `None` on either endpoint means "use the last commanded jaw". This unifies jaw scripting with motion scripting: one temporal model, one interpolator, no separate jaw FSM.

Justification: per piece 2 §4.2, jaws are visual-only kinematics in v1 — opening and closing is purely an animation. The IDM (and its future jaw-actuation head) trains on the *visible* jaw motion in the video, which is what the interpolated jaw target produces frame-by-frame. There's no physical advantage to a separate jaw control loop, and a separate one would add complexity for no gain.

### 4.6 Per-primitive parameter sampling at reset (no style vector in v1)

Aggregate-behaviour diversity across episodes comes from sampling primitive **kinds** (weighted choice from the medium primitive set) and primitive **parameters** (durations, distances, angles, jaw values, target points) from per-knob distributions on `MotionGeneratorConfig`. No multi-axis "surgeon style" vector; primitives are sampled independently.

This is YAGNI-correct: the IDM training distribution we want is *diverse* motion, not *coherent-within-episode-but-diverse-across-episodes* motion. A multi-axis style sampler is a future enhancement gated on evidence the per-parameter distribution is too narrow for IDM training.

---

## 5. Architecture

### 5.1 Unit decomposition

A new package `src/auto_surgery/motion/` with five focused modules plus the public `__init__`. Pure-Python; no SOFA dependency at any layer.

| Module | Purpose | Depends on |
|---|---|---|
| `motion/profile.py` | Minimum-jerk polynomial helpers; pure scalar math | numpy |
| `motion/primitives.py` | Frozen dataclass per primitive variant; `evaluate(elapsed, state, last_step) → PrimitiveOutput` pure function per variant | `profile.py`, `schemas/commands.py`, numpy |
| `motion/sequencer.py` | Episode-scoped primitive sampler; consumes `MotionGeneratorConfig` + `SceneConfig` + RNG; returns the next primitive given last tool state | `primitives.py`, `schemas/scene.py` |
| `motion/fsm.py` | Per-tick finite-state machine walking the active primitive; handles contact-event transitions and time-based completion | `primitives.py`, `sequencer.py`, `schemas/results.py` |
| `motion/generator.py` | Public entry point: `SurgicalMotionGenerator` class with `reset()` and `next_command()` | all above + `schemas/commands.py`, `schemas/results.py` |
| `motion/__init__.py` | Re-exports `SurgicalMotionGenerator`, `MotionGeneratorConfig`, primitive dataclasses, `TargetVolume` | — |

The five-module split exists because each piece has a single responsibility that is independently testable:

- `profile.py` is math, no domain. Tested with canned numbers.
- `primitives.py` is the per-primitive twist formula. Tested by feeding canned poses + elapsed time → expected twist.
- `sequencer.py` is the "what comes next" decision. Tested by mocking the RNG and asserting sampled sequence shape.
- `fsm.py` is the "when does the active primitive end" decision. Tested by feeding synthetic `StepResult` streams (in particular, contact rising/falling edges).
- `generator.py` is the public wiring. Tested end-to-end with a fake `StepResult` source.

### 5.2 Type definitions

#### 5.2.1 `SceneConfig` amendments (piece-2 surface)

```python
# src/auto_surgery/schemas/scene.py

class TargetVolume(BaseModel):
    """Region of surgical interest in scene frame.

    Tissue-seeking primitives (`Reach`, `ContactReach`) sample target points inside
    a TargetVolume. Non-tissue primitives operate in camera frame and ignore
    target_volumes.
    """
    model_config = {"extra": "forbid"}
    label: Literal["tumor", "vessel", "general"]
    center_scene: Vec3
    half_extents_scene: Vec3            # bbox half-extents; sphere uses isotropic radius from x
    shape: Literal["sphere", "bbox"] = "sphere"


class SceneConfig(BaseModel):
    model_config = {"extra": "forbid"}
    tissue_scene_path: Path             # renamed from brain_scene_path (piece-2 amendment)
    tool: ToolSpec
    camera_extrinsics_scene: Pose
    camera_intrinsics: CameraIntrinsics
    target_volumes: list[TargetVolume] = Field(..., min_length=1)
```

Both renames / additions are a tiny amendment to piece 2's design; piece-2 logic does not depend on either field's name or list-ness.

#### 5.2.2 Primitive dataclasses (implemented)

The authoritative definitions live in `src/auto_surgery/motion/primitives.py`. Summary:

```python
class PrimitiveKind(StrEnum):
    REACH = "reach"
    HOLD = "hold"
    CONTACT_REACH = "contact_reach"
    GRIP = "grip"
    DRAG = "drag"
    BRUSH = "brush"

Primitive = Reach | Hold | ContactReach | Grip | Drag | Brush
```

`evaluate(active, last_step) → PrimitiveOutput` emits twists in **camera frame** for the command contract; the generator converts to scene frame where needed for shaping, then back to camera frame in `RobotCommand`.

#### 5.2.3 `MotionGeneratorConfig`

Authoritative schema: `src/auto_surgery/schemas/motion.py`. Weights and duration ranges use **canonical** names aligned with `PrimitiveKind` (`weight_reach`, `reach_duration_range_s`, `hold_duration_range_s`, `drag_duration_range_s`, `brush_duration_range_s`, …). Legacy YAML keys from earlier drafts (`weight_approach`, `approach_duration_range_s`, …) are normalized at model load.

```python
class MotionGeneratorConfig(BaseModel):
    model_config = {"extra": "forbid"}
    seed: int = 0
    primitive_count_min: int = 8
    primitive_count_max: int = 20

    weight_reach: float = 1.0
    weight_hold: float = 0.5
    weight_contact_reach: float = 0.7
    weight_grip: float = 0.8
    weight_drag: float = 0.6
    weight_brush: float = 0.4

    reach_duration_range_s: tuple[float, float] = (0.6, 1.5)
    hold_duration_range_s: tuple[float, float] = (0.3, 0.8)
    drag_duration_range_s: tuple[float, float] = (0.4, 0.9)
    drag_distance_range_mm: tuple[float, float] = (3.0, 12.0)
    brush_duration_range_s: tuple[float, float] = (0.6, 1.4)
    brush_arc_range_rad: tuple[float, float] = (0.15, 0.6)
    # … reserved / future knobs (rotate_*, probe_*) …
    target_orientation_jitter_rad: float = 0.26
    jaw_value_range: tuple[float, float] = (0.0, 1.0)
    jaw_change_probability: float = 0.4
```

#### 5.2.4 Active-primitive FSM state

```python
# src/auto_surgery/motion/fsm.py

@dataclass
class _ActivePrimitive:
    primitive: Primitive
    started_at_tick: int
    started_at_pose_scene: Pose
    started_at_jaw: float
    duration_s: float
    elapsed_s: float = 0.0
    contact_was_in: bool = False
```

#### 5.2.5 Realised primitive record

```python
# src/auto_surgery/motion/generator.py

@dataclass(frozen=True)
class RealisedPrimitive:
    primitive: Primitive
    started_at_tick: int
    ended_at_tick: int
    early_terminated: bool                  # True iff contact event ended it
```

The generator exposes `realised_sequence: tuple[RealisedPrimitive, ...]` (completed primitives only) plus `finalize(last_step)` to close out the active primitive at episode end. The recorder calls `finalize` then dumps `realised_sequence` alongside the video as `motion_plan.json`. This satisfies piece 5's need to ship paired (motion-label, video) artefacts.

#### 5.2.6 Public class

```python
class SurgicalMotionGenerator:
    def __init__(
        self,
        motion_config: MotionGeneratorConfig,
        scene_config: SceneConfig,
    ) -> None: ...

    def reset(self, initial_step: StepResult) -> RobotCommand: ...

    def next_command(self, last_step: StepResult) -> RobotCommand: ...

    def finalize(self, last_step: StepResult) -> None: ...

    @property
    def realised_sequence(self) -> tuple[RealisedPrimitive, ...]: ...
```

---

## 6. Primitive semantics

Each primitive variant defines `evaluate(active: _ActivePrimitive, last_step: StepResult) → PrimitiveOutput` where:

```python
@dataclass(frozen=True)
class PrimitiveOutput:
    twist_camera: Twist
    jaw_target: float
    is_finished: bool
```

### 6.1 Minimum-jerk profile helpers (`profile.py`)

```python
def min_jerk_position_scalar(tau: float) -> float:
    """tau = elapsed/duration in [0,1]. Returns fraction of total travel completed."""
    if tau <= 0.0: return 0.0
    if tau >= 1.0: return 1.0
    return 10*tau**3 - 15*tau**4 + 6*tau**5

def min_jerk_velocity_scalar(tau: float, duration_s: float) -> float:
    """Derivative of min_jerk_position_scalar with respect to wall time."""
    if tau <= 0.0 or tau >= 1.0: return 0.0
    return (30*tau**2 - 60*tau**3 + 30*tau**4) / duration_s
```

Boundary properties (relied on by primitives and verified by unit tests):

- `min_jerk_position_scalar(0) == 0`, `min_jerk_position_scalar(1) == 1`.
- `min_jerk_velocity_scalar(0, T) == 0`, `min_jerk_velocity_scalar(1, T) == 0`.
- Velocity peaks at $\tau = 0.5$, equal to $1.875/T$.

### 6.2 Per-primitive twist formulas

| Primitive | Linear twist (scene frame) | Angular twist (scene frame) | Contact reaction |
|---|---|---|---|
| `Reach(target_pose_scene)` | $(\mathrm{target\_pos} - \mathrm{started\_pos}) \cdot v(\tau, T)$ | $\log(R_{\mathrm{target}} \cdot R_{\mathrm{started}}^{-1}) \cdot v(\tau, T)$ | `is_finished=True` on `tool.in_contact` rising edge if `end_on_contact=True`. |
| `Hold` | $0$ | $0$ | None |
| `Drag(distance_mm)` | incremental tangential delta from preferred workspace direction: `distance_mm / duration_s * v(tick_dt, duration_s)` (feedback-corrected by normal-force term) | $0$ | None |
| `Brush(amplitude_mm, frequency_hz)` | tangential sinusoidal path in workspace | tangential rotational component encoded as an equivalent scene-frame angular velocity | None |
| `ContactReach` | See below | See below | Contact or timeout finish |
| `Grip` | Piecewise path: `ContactReach` + lift-up + settle-back | piecewise hold + settle | Contact start flips to the lift phase; primitive finishes after hold/lift/release duration |

The output twist is computed in scene frame, then transformed to camera frame at output time using `last_step.cameras[0].extrinsics`. The action applier in piece 2 §6.2 transforms it back to scene frame. The round trip is intentional: the published command matches what an IDM seeing the same camera frame would predict, which is the IDM training-time invariant.

For `Drag` and `Brush`, direction vectors are specified in scene frame after rotation, with tangential fallback around the tool camera-frame axis when needed. The generated angular velocity is transformed to camera frame at output time (the generator's external contract is `Twist` in camera frame).

**Camera-frame convention**: this spec assumes OpenCV convention — camera `+z` along the optical axis (into the scene), `+x` right, `+y` down. "Camera `−z`" therefore means "away from the scene toward the viewer".

### 6.3 `ContactReach` and `Grip` behavior

`ContactReach` has one active phase: directional search until contact or timeout. Its evaluator moves toward a target sample defined by `direction_hint_scene`, `max_search_mm`, and `peak_speed_mm_per_s`; contact and geometry distance thresholds can both trigger finish.

`Grip` is a scripted wrapper around a `ContactReach` approach:

1. `ContactReach` phase (approach toward the workpiece).
2. Jaw closing phase (`jaw_close_duration_s`).
3. Lift phase (`lift_distance_mm`, `lift_duration_s`).
4. Release phase (`release_after_s`).
5. Final settle (remaining duration).

The wrapper keeps a single primitive clock and reports finished only after the hold/lift/reopen/settle path is complete.

### 6.4 Jaw interpolation

For all primitives:

```python
jaw_start = active.primitive.jaw_target_start if active.primitive.jaw_target_start is not None else active.started_at_jaw
jaw_end = active.primitive.jaw_target_end if active.primitive.jaw_target_end is not None else jaw_start
jaw_target = jaw_start + (jaw_end - jaw_start) * min_jerk_position_scalar(tau)
```

Both endpoints None → constant `started_at_jaw`. One endpoint None → constant at the non-None one (or constant at `started_at_jaw` if start is None and end is specified).

---

## 7. Sequencer logic

### 7.1 Reset

```python
def reset(self, initial_step: StepResult) -> None:
    self._rng = np.random.default_rng(self.motion_config.seed)
    self._planned_count = int(self._rng.integers(
        self.motion_config.primitive_count_min,
        self.motion_config.primitive_count_max + 1,
    ))
    self._issued = 0
```

The first primitive (returned by the first `next_primitive` call) is deterministically a `Reach` toward the centre of `scene_config.target_volumes[0]` with the configured `reach_duration_range_s` midpoint. This makes episode start a predictable glide into the operating zone rather than a random twitch — useful for video readability and as a deterministic anchor for test fixtures.

### 7.2 `next_primitive(last_step, last_jaw) → Primitive | None`

If `self._issued >= self._planned_count`, return None — the FSM converts this into an indefinite zero-twist `Hold` (the episode has run out of planned motion but `env.step` may still be called).

Otherwise:

1. Sample primitive kind: weighted choice from `{Reach, Hold, Drag, Brush, Grip, ContactReach}` using `motion_config.weight_*`.
2. Build the primitive variant with sampled parameters (see §7.3).
3. Sample jaw endpoints:
   - With probability `motion_config.jaw_change_probability`: `jaw_target_end = uniform(jaw_value_range)`, `jaw_target_start = None` (use `started_at_jaw`).
   - Otherwise: `jaw_target_start = jaw_target_end = None` (hold).
4. Increment `self._issued`.
5. Return the primitive.

### 7.3 Per-kind parameter sampling

| Kind | Parameter sampling |
|---|---|
| `Reach` | `target_pose_scene.position` = uniform point inside a `TargetVolume` sampled uniformly from `scene_config.target_volumes`. `target_pose_scene.rotation` = current tool rotation perturbed by Euler angles drawn from `Uniform(-target_orientation_jitter_rad, target_orientation_jitter_rad)` per axis. `duration_s` = uniform from `reach_duration_range_s`. `end_on_contact = True`. |
| `Hold` | `duration_s` = uniform from `hold_duration_range_s`. |
| `Drag` | `distance_mm` = uniform from `drag_distance_range_mm`; `direction_hint_scene` uses contact-space sampling preference; `duration_s` = uniform from `drag_duration_range_s`. |
| `Brush` | `amplitude_mm` and `frequency_hz` are fixed in implementation today; `duration_s` = uniform from `brush_duration_range_s`. |
| `Grip` | `approach = _build_contact_reach(last_step)`; `lift_distance_mm`, `lift_duration_s`, `jaw_close_duration_s`, `release_after_s` are fixed; `duration_s = sampled_range(reach_duration_range_s) + 1.8`. |
| `ContactReach` | `direction_hint_scene = None`; `max_search_mm = 10.0`; `peak_speed_mm_per_s = 15.0`; `duration_s` = uniform from `reach_duration_range_s`. |

The `Brush` axis bias prevents sweeps along the optical axis (which would look like zooming, not brushing) by drawing the in-plane component first and adding a small out-of-plane component. The exact bias formula is unit-tested.

### 7.4 Lazy sampling

The sequencer is **lazy**: `next_primitive` is called only when the FSM asks for one (i.e., when the prior primitive finishes). This means the sampled target pose is always relative to the *current* tool pose after physics has acted on it, which yields more coherent motion than pre-planning at reset would.

---

## 8. FSM logic

### 8.1 Step

```python
def step(self, last_step: StepResult, last_command_jaw: float) -> _ActivePrimitive:
    if self._active is None or self._active_finished(last_step):
        self._record_finished_if_any(last_step.sim_step_index)
        next_p = self._sequencer.next_primitive(last_step, last_command_jaw)
            if next_p is None:
                next_p = Hold(
                duration_s=1e9,
                jaw_target_start=None,
                jaw_target_end=None,
            )
        self._active = _ActivePrimitive(
            primitive=next_p,
            started_at_tick=last_step.sim_step_index,
            started_at_pose_scene=last_step.sensors.tool.pose,
            started_at_jaw=last_command_jaw,
            duration_s=next_p.duration_s,
        )
    self._active.elapsed_s += last_step.dt
    return self._active
```

### 8.2 Finish detection

```python
def _active_finished(self, last_step: StepResult) -> bool:
    active = self._active
    if active is None:
        return False

    contact_now = last_step.sensors.tool.in_contact
    contact_rising = contact_now and not active.contact_was_in
    active.contact_was_in = contact_now

    match active.primitive:
        case Reach(end_on_contact=True) if contact_rising:
            return True
        case _:
            return active.elapsed_s >= active.duration_s
```

The match arms are intentionally minimal: the only early-contact finish path is `Reach` with `end_on_contact=True`. All other primitives finish on elapsed time alone.

### 8.3 Realised-primitive logging

When `_active_finished` returns True, the FSM appends a `RealisedPrimitive(primitive, started_at_tick, ended_at_tick=last_step.sim_step_index, early_terminated=...)` to its internal log. `early_terminated` is True iff the finish was caused by a contact event rather than time expiry.

---

## 9. `SurgicalMotionGenerator` per-tick algorithm

### 9.1 Construction

```python
def __init__(self, motion_config, scene_config):
    self._motion_config = motion_config
    self._scene_config = scene_config
    self._sequencer = _Sequencer(motion_config, scene_config)
    self._fsm = _Fsm(self._sequencer)
    self._cycle_id = -1
    self._last_jaw_commanded = scene_config.tool.initial_jaw
    self._reset_called = False
```

### 9.2 `reset(initial_step) → RobotCommand`

1. `self._sequencer.reset(initial_step)`
2. `self._fsm.reset()` (clears `_active`, clears completed-primitive log)
3. `self._cycle_id = -1` (the increment inside `next_command` lands on 0 for the first command)
4. `self._last_jaw_commanded = self._scene_config.tool.initial_jaw`
5. `self._reset_called = True`
6. Return `self.next_command(initial_step)`.

The initial command emitted by `reset` has `cycle_id = 0`, matching piece-1 §6.1's "first valid command may use `cycle_id = 0`".

### 9.3 `next_command(last_step) → RobotCommand`

If `not self._reset_called`: raise (`RuntimeError("reset() must be called before next_command()")`).

1. `active = self._fsm.step(last_step, self._last_jaw_commanded)`
2. `output = _evaluate(active, last_step)` (dispatches by primitive variant)
3. `self._last_jaw_commanded = output.jaw_target`
4. `self._cycle_id += 1`
5. Build and return:

```python
RobotCommand(
    timestamp_ns=last_step.sensors.timestamp_ns + int(last_step.dt * 1e9),
    cycle_id=self._cycle_id,
    control_mode=ControlMode.CARTESIAN_TWIST,
    cartesian_twist=output.twist_camera,
    frame=ControlFrame.CAMERA,
    tool_jaw_target=output.jaw_target,
    enable=True,
    source="scripted",
)
```

`enable=True` is always set; the env's safety gate (piece 1 §6) is the only authority that decides whether motion actually applies. The generator does not gate its own commands.

### 9.4 `realised_sequence` and `finalize()`

The generator exposes:

- `realised_sequence: tuple[RealisedPrimitive, ...]` — property returning the FSM's log of **completed** primitives (each with `ended_at_tick` set to the tick on which the primitive finished). The currently-active primitive is *not* included; this keeps `RealisedPrimitive.ended_at_tick: int` unambiguous (no sentinel value).
- `finalize(last_step: StepResult) -> None` — closes out the currently-active primitive by appending its `RealisedPrimitive` record (with `ended_at_tick = last_step.sim_step_index` and `early_terminated = False`). Idempotent: subsequent calls are no-ops.

The recorder calls `gen.finalize(last_step)` once at episode end, immediately before writing `motion_plan.json`. This guarantees the manifest covers every tick of the episode.

---

## 10. Integration

### 10.1 Recorder loop (rewritten `recording/brain_forceps.py`)

```python
scene_config = load_scene_config(args.scene_config_path)
motion_config = load_motion_config(args.motion_config_path)
if args.seed is not None:
    motion_config = motion_config.model_copy(update={"seed": args.seed})

env = SofaEnvironment(...)
last_step = env.reset(EnvConfig(scene=scene_config, ...))

gen = SurgicalMotionGenerator(motion_config, scene_config)
cmd = gen.reset(last_step)

for tick in range(args.ticks):
    last_step = env.step(cmd)
    if last_step.is_capture_tick:
        recorder.write_frame(last_step)
    cmd = gen.next_command(last_step)

gen.finalize(last_step)
recorder.write_manifest(gen.realised_sequence)
```

`load_scene_config` and `load_motion_config` are trivial wrappers around `yaml.safe_load` + `SceneConfig.model_validate` / `MotionGeneratorConfig.model_validate`; both raise pydantic's `ValidationError` on malformed input (no custom error handling needed).

CLI changes (Typer):

| Removed | Added |
|---|---|
| `--joint-start`, `--joint-step`, `--joint-sine-amplitude`, `--joint-sine-frequency` | `--scene-config` (path to scene YAML, e.g., `configs/scenes/dejavu_brain.yaml`) |
| `--frames` (current "joint-only" semantics) | `--ticks` (control-tick count; frames derived from `is_capture_tick`) |
|  | `--motion-config` (path to `MotionGeneratorConfig` YAML; defaults at `configs/motion/default.yaml`) |
|  | `--seed` (overrides `motion_config.seed`) |

### 10.2 Deleted code

In the same PR:

- `src/auto_surgery/env/action_generators.py`: `SineJointPositionGenerator`, `CoherentRandomWalkGenerator`, `ActionGenerator` protocol, `build_default_action_generator`. File becomes empty or is deleted entirely.
- `src/auto_surgery/recording/brain_forceps.py`: `_build_action`, joint-CLI flags, the `JOINT_POSITION` command-construction path.
- `src/auto_surgery/training/sofa_smoke.py`: replace `build_default_action_generator` / `build_lite_command` calls with `SurgicalMotionGenerator` usage.
- `src/auto_surgery/training/sofa_forceps_smoke.py`: replace scripted `j0 = 0.05 * step_index` with `SurgicalMotionGenerator` usage.

Tests that import any deleted symbol are rewritten to construct a `SurgicalMotionGenerator` instead.

### 10.3 Committed config files

| Path | Purpose |
|---|---|
| `configs/scenes/dejavu_brain.yaml` | First and only `SceneConfig` in v1. Contains `tissue_scene_path` (the piece-2 brain scene), `tool` (the `dejavu_forceps` `ToolSpec`), `camera_extrinsics_scene`, `camera_intrinsics`, and one `target_volume` covering the exposed cortex. |
| `configs/motion/default.yaml` | Default `MotionGeneratorConfig`. Calibrated by the visual smoke test in §11.4. |

Both YAML files load via pydantic `model_validate(yaml.safe_load(...))`.

---

## 11. Testing strategy

Four layers, ordered by speed.

### 11.1 Pure-function unit tests (no SOFA, no env)

- `profile.py`:
  - `min_jerk_position_scalar(0) == 0`, `min_jerk_position_scalar(1) == 1`.
  - `min_jerk_velocity_scalar(0, T) == 0`, `min_jerk_velocity_scalar(1, T) == 0`.
  - Monotonicity of `min_jerk_position_scalar` on $[0,1]$.
  - Peak velocity at $\tau = 0.5$.
- `primitives.py`:
  - For each variant (`Reach`, `Hold`, `Drag`, `Brush`, `ContactReach`, `Grip`), feed canned `_ActivePrimitive` + canned `last_step.sensors`, assert returned twist matches analytical expectation at $\tau \in \{0, 0.25, 0.5, 0.75, 1.0\}$.
  - `Reach(end_on_contact=True)`: rising edge of `tool.in_contact` returns `is_finished=True`.
  - `Grip`: verify close/lift/release/settle phase progression and contact-aware behavior.
  - Jaw interpolation: `(jaw_start=None, jaw_end=None)` → constant `started_at_jaw`. `(0.2, 0.8)` → `0.5` at $\tau=0.5$.
- `MotionGeneratorConfig` YAML: valid configs parse; malformed configs raise.
- `SceneConfig` YAML with the amended fields: valid + `target_volumes` empty list rejected.

### 11.2 Sequencer + FSM tests (no SOFA, synthetic StepResult stream)

- Fixed seed → reproducible primitive sequence (kinds + parameters match a golden).
- Contact rising-edge mid-`Reach` causes FSM to advance to the next primitive within one tick.
- Lazy sampling: each `next_primitive` call sees the *current* `last_step.sensors.tool.pose` (verified by mocking the input stream).
- Episode exhausts after `planned_count` primitives → subsequent calls return zero-twist `Hold` commands.
- `_active_finished` returns False on every tick within a `Hold` of `duration_s = 1.0` for 999 ticks at 250 Hz, True on the 1000th.
- Contact-aware sequencing in `Grip` holds until close/lift/release windows complete.
- Contact edge handling: with synthetic stream where contact rises late, `Grip` does *not* terminate on the first flicker once scripted phases are active.
- `finalize(last_step)` appends exactly one record corresponding to the currently-active primitive with `early_terminated=False`; second call is a no-op.

### 11.3 SOFA-backed integration test (uses the piece-2 forceps + brain scene)

- Run `SurgicalMotionGenerator` driving the env for ~500 ticks at 250 Hz.
- Assert at least one tick has `tool.in_contact == True`.
- Assert at least one tick has `||tool.wrench|| > 0.1` (threshold tuned to piece-2 penalty stiffness).
- Assert `realised_sequence` contains at least one `Reach` with `early_terminated=True`.
- Assert every emitted command has `enable=True` and `control_mode=CARTESIAN_TWIST`.
- Determinism: same `seed` + same `scene_config` + same SOFA build → byte-identical `realised_sequence`.

### 11.4 Visual smoke test (slow, manual, not in CI by default)

Render a 30-second clip at 30 fps using the default `MotionGeneratorConfig` against `configs/scenes/dejavu_brain.yaml`. Acceptance criteria (visual, by eye):

- Tool moves continuously across the clip; no visible jerks at primitive boundaries.
- At least 3 distinct primitive kinds are visibly active.
- Jaws visibly open and close at least twice.
- Tool contacts brain tissue and the brain visibly deforms at least once.
- No tool teleports, no NaN poses, no FEM blow-up.

This is the **calibration acceptance gate** for the default values committed in `configs/motion/default.yaml`, mirroring piece 2 §9.4's gate for `dejavu_default.yaml`.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Primitive boundary discontinuities.** Min-jerk velocity is zero at $\tau=1$ and zero at $\tau=0$ of the next primitive, so velocity is C⁰-continuous; but acceleration may jump. | Acceptable for v1 — the tool is velocity-source, so an acceleration discontinuity in the command becomes a smooth physics response. If the smoke test reveals visible jerks, add a one-tick C¹ blend at primitive boundaries. |
| **Targets sampled inside `TargetVolume` end up below the tissue surface.** | Acceptable: `Reach(end_on_contact=True)` early-terminates before the target is reached; the velocity-source rigid body deflects naturally. Verified by integration test §11.3. |
| **`Brush` axis sampling produces sweeps that move the tool out of camera view.** | The sampler biases tangential motion in camera frame; if it causes out-of-frame motion, reduce `brush_duration_range_s` and/or tangential amplitude. |
| **RNG state desyncs if `reset()` is called mid-episode.** | `reset()` is the only legal start; subsequent `next_command()` calls use the same RNG. `next_command()` raises if called before `reset()`. |
| **`Grip` scripted hold/lift sequencing becomes unstable.** | `Grip` phases are guarded by elapsed-time bounds and contact-aware transition logic; keep `jaw_close_duration_s`, `lift_duration_s`, and `release_after_s` within tuned ranges. |
| **`SceneConfig.target_volumes` is empty.** | Pydantic `Field(..., min_length=1)`. Test §11.1 verifies rejection. |
| **CLI YAML config files don't exist yet in the repo.** | Both `configs/scenes/dejavu_brain.yaml` and `configs/motion/default.yaml` are committed in the piece-3 implementation PR, hand-edited after smoke-test calibration. |
| **The renamed `tissue_scene_path` field breaks any in-flight piece-2 work that reads `brain_scene_path`.** | Piece 2's implementation has not landed yet at the time of this spec. The rename lands in the same PR as piece 2's `SceneConfig` implementation; there is no rolling deprecation. |

---

## 13. Deferred items with trigger conditions

| Item | Re-open when |
|---|---|
| Stylistic randomisation (multi-axis "surgeon style" vector) | IDM training shows overfitting to a narrow motion distribution; evidence from validation video collections. |
| Hierarchical / task-level primitives (`ResectTumor` composing sub-primitives) | A policy-training run wants surgical-task structure as a control hierarchy. Not piece 3. |
| Anatomy-aware target selection (probe-vessel-not-tumor bias) | Scenes with both `vessel` and `tumor` labels are in use (kidney, liver) and per-label motion bias is needed. Piece 3 ships labels in `TargetVolume`; semantic weights stay flat in v1. |
| 7th+ primitive variants (`Trace`, `Idle`, `Lift`, `Reposition`) | A specific motion pattern is missing from generated clips and visibly hurts IDM data quality. |
| Per-episode realised-sequence replay (re-run a known sequence on a different scene seed) | A forward-model / world-model training run wants paired (motion, deformation) data with motion held fixed across scene seeds. |
| Velocity-magnitude scaling per primitive | Smoke-test reveals that uniform velocity ranges across primitives are noticeably wrong. |
| C¹-continuous blending between primitives | Smoke test shows visible acceleration discontinuities. |
| Multi-tool primitive variants (e.g., `Cut` for scissors) | The tool catalog expands (piece 2 §11 trigger). |

---

## 14. Piece-2 amendments tracked here

These changes amend `docs/specs/2026-05-12-physical-forceps-design.md` §5.4. Neither affects piece-2 logic; both are pure typed-surface adjustments:

1. **Rename** `SceneConfig.brain_scene_path` → `SceneConfig.tissue_scene_path` (multi-scene-ready).
2. **Add** `SceneConfig.target_volumes: list[TargetVolume]` with `min_length=1`.

Both land in the same PR as the piece-3 implementation. The piece-2 spec is updated at the same time (a one-section diff under §5.4).

---

## 15. Implementation completion checklist

Piece 3 lands when:

- [ ] `src/auto_surgery/motion/` package exists with the five modules in §5.1, each with full unit-test coverage.
- [ ] `SurgicalMotionGenerator`, `MotionGeneratorConfig`, `TargetVolume`, and the six primitive dataclasses are public exports.
- [ ] `src/auto_surgery/schemas/scene.py` is amended per §14.
- [ ] `configs/scenes/dejavu_brain.yaml` and `configs/motion/default.yaml` are committed, calibrated by smoke test.
- [ ] `src/auto_surgery/recording/brain_forceps.py` is rewritten to drive `SurgicalMotionGenerator`.
- [ ] `src/auto_surgery/env/action_generators.py` is deleted (or reduced to a stub if any non-generator code lived there).
- [ ] `training/sofa_smoke.py` and `training/sofa_forceps_smoke.py` are updated to use `SurgicalMotionGenerator`.
- [ ] All four test layers (§11.1–§11.4) pass; the visual smoke test gates calibration commit.

---

## 16. Pointer to piece 4

Piece 3 emits diverse motion within a fixed `(SceneConfig, MotionGeneratorConfig)` pair. **Piece 4 — Domain randomisation framework** — adds the per-episode sampler that perturbs the scene (mesh warp, texture tint, camera extrinsics, FEM constants, target-volume positions) and may also resample `MotionGeneratorConfig` knobs from a wider distribution. The motion generator stays unchanged; piece 4 plugs in *above* it by producing a fresh `(SceneConfig, MotionGeneratorConfig)` pair per episode. Piece 4's design doc opens after this one is reviewed and committed.
