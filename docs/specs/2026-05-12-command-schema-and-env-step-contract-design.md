# Command Schema and Environment-Step Contract — Design

| Field | Value |
|---|---|
| Date | 2026-05-12 |
| Status | Approved (design); implementation complete |
| Piece | 1 of 5 (in the simulation-pipeline redesign) |
| Supersedes | `src/auto_surgery/schemas/commands.py`, `src/auto_surgery/schemas/results.py`, `src/auto_surgery/schemas/sensors.py`, and the `Environment` protocol stub in `docs/library/ARCHITECTURE.md` §8.1 |
| Backward compatibility | **None.** All callers migrate. |

---

## 1. Context and scope

The auto-surgery simulator exists to produce training data for an inverse-dynamics model (IDM) that learns to predict surgical-tool actions from internet surgical video. The simulator must therefore (i) emit actions in a form the IDM can learn to predict from video, and (ii) drive a real surgical robot (KUKA KR 6-2 in development, neuroArm in target) without representation conversions that lose information.

The previous schemas left key shapes as free-form dicts (`mode_flags: dict[str, Any]`, `cartesian_pose: dict[str, Any]`, `modalities: dict[str, Any]`, `info: dict[str, Any]`), which let inconsistencies accumulate. This spec replaces the action–observation interface with fully-typed structures aligned to the project's actual control needs.

This is **piece 1 of 5** in a coordinated redesign:

1. **Command schema + environment-step contract** *(this spec)*
2. Physical forceps in SOFA (rigid body + collision + brain-FEM coupling + jaw actuation)
3. Human-like, scene-aware motion generator
4. Domain randomization framework (mesh warp, texture, camera, FEM constants, topology)
5. Recording pipeline (paired video × command stream for IDM training)

Each subsequent piece gets its own design doc and implementation plan. This spec is the lingua franca every later piece consumes.

---

## 2. Goals

1. Provide a single, fully-typed `RobotCommand` whose primary representation is **observable from monocular video**, so the simulator's action labels are in the same space the IDM is trained to predict from real surgical footage.
2. Provide a single, fully-typed `StepResult` / `SensorBundle` that captures everything a per-tick consumer (recorder, policy at eval time, scene-aware motion generator) actually needs — and nothing else.
3. Define a deterministic `env.step(command) → StepResult` contract with explicit invariants for cycle ordering, enable gating, and replay reproducibility.
4. Make the IDM training pipeline a structural consequence of the schema: aggregate twist between consecutive frames is exactly reconstructible from the recorded stream.
5. Preserve symmetry with the KUKA hand-controller SDK (`reference_kuka/sdk.rs` `FromHcToRobot` / `FromRobotToHCMessage`) so the KUKA bridge is a ~10-line converter, not a representation rewrite.
6. Eliminate every `dict[str, Any]` from the public action–observation surface.

## 3. Non-goals

1. **Joint-space policy training.** Joint state is recorded for bridges and audit, but the IDM and the policy operate in Cartesian twist. Joint-space modes exist in the schema only for bridge compatibility and calibration utilities.
2. **Real-time stream handling** (timeouts, missed-tick recovery, jitter compensation). These live at hardware bridges, not in the env or schema.
3. **Layer 1 / Layer 2 safety surfaces** from `docs/library/ARCHITECTURE.md` §10. The env enforces only schema-level invariants (cycle ordering, enable gate, mode validity). Higher-layer safety policies sit above the env.
4. **Force/impedance control** for v1. `CARTESIAN_FORCE` is not part of the primary mode set; can be added when contact-rich autonomous policies need it.
5. **Multi-arm coordination.** v1 is single-tool. The schema does not preclude multi-arm later but does not pre-engineer it.

---

## 4. Foundational design decisions

### 4.1 Primary action representation: camera-frame tool-tip twist

The IDM's training distribution is internet surgical footage, which has no joint state, no world frame, no camera extrinsics — only camera-relative tool motion. The action space the simulator emits must therefore be **camera-relative tool motion**, expressed at every control tick.

Concretely the primary command payload is a `Twist` (linear + angular velocity, both in $\mathbb{R}^3$) expressed in the **instantaneous recording-camera frame** at command time. Units: linear m/s, angular rad/s (axis-rate vector, $\|\omega\|$ = rate, direction = axis of rotation).

Why twist and not per-cycle delta-pose (which is what KUKA `FromHcToRobot` packs):
- Twist is a continuous physical quantity independent of any sampling interval. The simulator's sim tick, the robot's control tick, and the camera's frame tick all have different dt; using twist lets each consumer integrate over its own dt without representation drift.
- The KUKA bridge converts twist → delta-pose by multiplying by the RSI cycle dt (~4 ms). One line. Zero information loss.
- The IDM training label between video frames $t_k$ and $t_{k+1}$ is $\int_{t_k}^{t_{k+1}} \mathrm{twist}(\tau)\,d\tau$ — exactly recoverable from the recorded tick stream.

Camera-frame, not world-frame: world-frame actions require world-frame estimation from video (camera extrinsics), which is not available for internet footage. Camera-frame actions match what the IDM can recover.

Camera-frame, not image-plane: image-plane (du, dv) actions are most observable but require hand-eye + intrinsics to execute on a real robot; brittle under domain randomization that re-rolls intrinsics.

### 4.2 Tick coordination: sim_tick = control_tick

One `env.step(command)` advances both the SOFA simulation and the control loop by exactly `1 / control_rate_hz`. Default control rate is 250 Hz (4 ms), matching KUKA RSI and within FEM stability for `TetrahedronFEMForceField`.

The frame rate is decoupled by integer decimation: `frame_decimation = round(control_rate_hz / frame_rate_hz)`. Default: 250 / 30 ≈ 8 ticks per frame. A `StepResult.is_capture_tick: bool` tells the recorder which ticks carry a rendered frame.

This means:
- Caller drives the loop at the control rate (the recorder is the canonical caller).
- High-rate commands are never aggregated inside env; each command is one tick.
- Frame rendering cost is paid only on capture ticks.
- IDM dataset rows are produced on capture ticks; the command stream between rows is fully recoverable for label aggregation.

### 4.3 Hand-eye calibration lives at the bridge, not in the schema

Camera-frame twist cannot be applied to a real-robot actuator directly; it must be transformed by the fixed hand-eye matrix $T_{\text{robot} \leftarrow \text{camera}}$ at the hardware boundary:

$$\mathrm{twist}_{\text{robot}} = \mathrm{Ad}(T_{\text{robot} \leftarrow \text{camera}}) \cdot \mathrm{twist}_{\text{camera}}$$

That transform is per-rig, one-time, and known from a chessboard/charuco calibration procedure. It is **not** part of `RobotCommand`. It is a private field of each robot bridge. The IDM, the policy, and the sim never see it.

This is necessary regardless of the action representation we chose (world-frame would need world-to-robot; image-plane would need hand-eye + intrinsics). Camera-frame twist makes the calibration burden minimal and explicit.

### 4.4 Tool-jaw actuation is a target scalar, not a rate

`tool_jaw_target: float | None` in `[0.0, 1.0]` where 0 = fully open, 1 = fully closed. Mirrors KUKA `tool_actuation_override` semantics. Not a rate because:
- Jaw state is directly observable from a single video frame; jaw rate is not.
- Real tool gripper controllers expose target-position interfaces, not velocity interfaces.
- `None` means "hold current state," allowing twist-only ticks without forcing the caller to repeat the jaw value.

### 4.5 Cycle ordering and enable gating live in the schema

`cycle_id: int` is required and strictly increasing per command stream. `enable: bool` is required and defaults to `False`. Both gates are enforced at the env (not just at the bridge) so the sim and real robot share one structural safety boundary in the type system.

---

## 5. Type definitions

All types live in `src/auto_surgery/schemas/`. All use `pydantic.BaseModel` with `model_config = {"extra": "forbid"}`. All free-form `dict[str, Any]` fields are removed.

### 5.1 Geometric primitives

```python
class Vec3(BaseModel):
    """Right-handed, SI units. Position (m), velocity (m/s), force (N)."""
    x: float
    y: float
    z: float


class Quaternion(BaseModel):
    """Hamilton convention, unit-norm enforced at validation."""
    w: float
    x: float
    y: float
    z: float


class Pose(BaseModel):
    """Rigid pose in some declared frame."""
    position: Vec3
    rotation: Quaternion


class Twist(BaseModel):
    """6-DoF instantaneous velocity in a declared frame.

    linear:  m/s
    angular: rad/s as axis-rate vector (||omega|| = rate, direction = axis)
    """
    linear: Vec3
    angular: Vec3
```

### 5.2 Mode and frame enums

```python
class ControlFrame(StrEnum):
    CAMERA = "camera"          # primary: instantaneous recording-camera frame at command time
    TOOL_TIP = "tool_tip"      # body-fixed at tool tip
    SCENE = "scene"            # world / sim frame; internal use
    ROBOT_BASE = "robot_base"  # bridge-side only


class ControlMode(StrEnum):
    CARTESIAN_TWIST = "cartesian_twist"   # primary mode (sim, IDM, policy)
    CARTESIAN_POSE = "cartesian_pose"     # bridge (neuroArm-native)
    JOINT_VELOCITY = "joint_velocity"     # bridge (KUKA-internal)
    JOINT_POSITION = "joint_position"     # bridge (calibration / homing)
```

### 5.3 Command

```python
class RobotCommand(BaseModel):
    model_config = {"extra": "forbid"}

    # Identity / ordering
    timestamp_ns: int
    cycle_id: int
    control_mode: ControlMode

    # Primary payload (exactly one populated, matching control_mode)
    cartesian_twist: Twist | None = None
    cartesian_pose_target: Pose | None = None
    joint_velocities: dict[str, float] | None = None    # rad/s per joint name
    joint_positions: dict[str, float] | None = None     # rad per joint name

    frame: ControlFrame = ControlFrame.CAMERA           # frame for cartesian payload

    # Tool actuation (separate DoF from twist; target-state, not rate)
    tool_jaw_target: float | None = Field(default=None, ge=0.0, le=1.0)

    # Safety / enable (mirrors KUKA FromHcToRobot)
    enable: bool = False

    # Bridge/audit metadata (never an IDM/policy target)
    source: str = "sim"   # "sim" | "policy" | "tele_op" | "scripted"
```

Validation rules:
- Exactly one payload field corresponding to `control_mode` is non-None; others are None. Validator enforces this and rejects mismatches.
- `frame` is meaningful only for `CARTESIAN_TWIST` / `CARTESIAN_POSE`; ignored for joint modes.
- `cycle_id >= 0`.
- `Quaternion` is normalized at validation (within $10^{-6}$); otherwise rejected.

Semantics of `tool_jaw_target = None`: env retains the last commanded jaw target across ticks. Initial value at `reset()` is the scene's `initial_jaw` configured in `SceneConfig` (defined in piece 2; default fully open = 0.0).

### 5.4 Observation types

```python
class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class ToolState(BaseModel):
    """Tool tip state in camera frame."""
    pose: Pose
    twist: Twist
    jaw: float                # current jaw closure in [0, 1]
    wrench: Vec3              # Cartesian force at tool tip in CAMERA frame (N)
    in_contact: bool


class CameraView(BaseModel):
    camera_id: str
    timestamp_ns: int
    extrinsics: Pose              # in SCENE frame at capture time
    intrinsics: CameraIntrinsics
    frame_rgb: bytes | None = None   # PNG bytes on capture ticks, None otherwise


class SafetyStatus(BaseModel):
    motion_enabled: bool
    command_blocked: bool
    block_reason: str | None      # short tag, e.g., "stale_cycle_id" | "disabled"
    cycle_id_echo: int


class SensorBundle(BaseModel):
    model_config = {"extra": "forbid"}
    timestamp_ns: int
    sim_time_s: float
    tool: ToolState
    cameras: list[CameraView]   # len 1 in v1; stereo by later addition
    safety: SafetyStatus


class StepResult(BaseModel):
    model_config = {"extra": "forbid"}
    sensors: SensorBundle
    dt: float                    # control-tick duration actually advanced (s)
    sim_step_index: int          # monotonic from reset()
    is_capture_tick: bool
```

### 5.5 Off-path query types

These exist only for bridges, debug viewers, and calibration; not part of the IDM/policy training loop.

```python
class JointState(BaseModel):
    positions: dict[str, float]    # rad per joint name
    velocities: dict[str, float]   # rad/s per joint name
    torques: dict[str, float] | None = None   # Nm, when measured/simulated


class Contact(BaseModel):
    """One contact point between the tool and tissue, in camera frame."""
    point: Vec3
    normal: Vec3
    force_magnitude: float   # N, signed compressive
    body_id: str             # which deformable body (e.g., "brain")
    penetration_depth: float # m, positive when interpenetrating
```

### 5.6 Episode configuration

```python
class EnvConfig(BaseModel):
    model_config = {"extra": "forbid"}
    scenario_id: str = "default"
    seed: int = 0
    scene: SceneConfig
    domain_randomization: DomainRandomizationConfig   # defined in piece 4
    control_rate_hz: float = 250.0
    frame_rate_hz: float = 30.0
    episode_max_ticks: int | None = None              # caller hint, not env-enforced
```

`DomainRandomizationConfig` is defined in piece 4; this spec treats it as an opaque typed model. Its shape does not affect the command schema or env contract.

### 5.7 Environment protocol

```python
class Environment(Protocol):
    """Sim/real parity boundary. Replaces ARCHITECTURE.md §8.1 stub."""

    def reset(self, config: EnvConfig) -> StepResult: ...
    def step(self, command: RobotCommand) -> StepResult: ...
    def get_joint_state(self) -> JointState: ...      # off-path
    def get_contacts(self) -> list[Contact]: ...      # off-path (debug/viz)
```

`reset(config)` returns the initial state as a capture tick: `sim_step_index = 0`, `is_capture_tick = True`, `dt = 0.0`, sensors fully populated, `cameras[*].frame_rgb` rendered.

### 5.7.1 Runtime conformance checklist

Use this as a minimal regression anchor for the piece-1 contract:

- `env.reset(config)` sets `sim_step_index == 0`, `is_capture_tick == True`, and `dt == 0.0`.
- `env.reset(config)` writes `control_rate_hz` through to backend step cadence (`StepResult.dt == 1 / config.control_rate_hz` on the next accepted tick).
- `is_capture_tick` transitions follow `frame_decimation = round(control_rate_hz / frame_rate_hz)`.
- On non-capture ticks, `StepResult.sensors.cameras[*].frame_rgb is None`; on capture ticks, frame payload is preserved.
- `stale_cycle_id` and `enabled=False` must both map to `command_blocked=True`, with `block_reason` set to the triggering cause and `cycle_id_echo == command.cycle_id`.
- `motion_enabled` must equal `command.enable and not command_blocked` for both real and stub backends.
- Blocked commands must be converted to deterministic no-op `ControlMode.CARTESIAN_TWIST` payloads at the env boundary.

---

## 6. Env-step contract (per-tick invariants)

Each `env.step(command)` executes the following in order:

1. **Cycle-id check.** `last_accepted_cycle_id` is initialized to `-1` at `reset()`, so the first valid command may use `cycle_id = 0`. If `command.cycle_id <= last_accepted_cycle_id`, command is rejected: motion not applied, `safety.command_blocked = True`, `safety.block_reason = "stale_cycle_id"`. Tick still advances; sensors still measured; frame still rendered if capture tick.
2. **Enable check.** If `command.enable == False`, motion not applied: twist treated as zero, jaw treated as "hold". `safety.command_blocked = True`, `safety.block_reason = "disabled"`. Tick still advances.
3. **Mode dispatch.** For `CARTESIAN_TWIST` (the primary path), the camera-frame twist is transformed to scene frame using the current camera extrinsics, then applied to the rigid forceps body for one tick (piece 2 implements the rigid-body application). Other modes route through the same applier via mode-specific converters.
4. **Sim advance.** SOFA `animate(dt)` runs exactly once with `dt = 1 / config.control_rate_hz`. Contacts resolved, FEM integrated.
5. **Sensor read.** `ToolState` (pose, twist, jaw, wrench, in_contact) read from the live SOFA scene in camera frame. `safety.motion_enabled = command.enable AND NOT safety.command_blocked`. `safety.cycle_id_echo = command.cycle_id`.
6. **Capture decision.** `is_capture_tick = (sim_step_index % frame_decimation == 0)` where `frame_decimation = round(control_rate_hz / frame_rate_hz)`. If true, render `frame_rgb` for each camera in the scene; otherwise `frame_rgb = None`.
7. **Counter advance.** `sim_step_index += 1`. `last_accepted_cycle_id = command.cycle_id` iff command was accepted.

The env never raises during normal operation; failures (bad schema, missing scene, unsupported mode) raise during `reset()` or at command-validation time, not mid-loop.

---

## 7. Determinism and replay

Given the same `EnvConfig` (same seed, scenario, rates, scene, randomization), same SOFA + plugin build, and the same command stream `[cmd_0, cmd_1, ...]`, `env.reset(config)` followed by `env.step(cmd_i)` produces a bitwise-identical `[StepResult_0, StepResult_1, ...]` sequence.

The recorder relies on this for offline re-rendering: same commands + different `domain_randomization.camera` realization → new video, same action labels.

---

## 8. IDM training pipeline integration

For inter-frame interval $[t_k, t_{k+1}]$ between consecutive capture ticks:

- Input frames: `cameras[0].frame_rgb` from `StepResult` at $t_k$ and $t_{k+1}$.
- Aggregate label:
  $$\mathrm{label}_k = \sum_{i \in [\text{tick}(t_k),\,\text{tick}(t_{k+1}))} \mathrm{cmd}_i.\mathrm{cartesian\_twist} \cdot \mathrm{step\_result}_i.dt$$
  expressed in the camera frame at $t_k$. Twist's linear part integrates directly; angular part integrates via exponential map of the axis-rate vector to a rotation increment.

Because every accepted command (twist) and every per-tick `dt` are recorded, the aggregate is exactly recoverable downstream. The recorder may pre-compute it or store the raw stream and aggregate at dataloader time; both are correct.

Twist values from rejected ticks (blocked by cycle-id or enable) are excluded from aggregation; this is detectable from `safety.command_blocked` in the per-tick `StepResult`.

---

## 9. Hardware bridge integration

### 9.1 KUKA KR 6-2 bridge

Converts `RobotCommand(CARTESIAN_TWIST in CAMERA)` → `FromHcToRobot`:

1. Transform twist: $\mathrm{twist}_{\text{robot}} = \mathrm{Ad}(T_{\text{robot} \leftarrow \text{camera}}) \cdot \mathrm{twist}_{\text{camera}}$.
2. Compute per-cycle deltas: `pos_xyz = twist_robot.linear * cycle_dt`, `rot_xyz = twist_robot.angular * cycle_dt`.
3. Fill `tool_actuation_override = tool_jaw_target` (or hold if None).
4. Fill `enable`, `cycle_id` directly from the command.
5. Pack the rest (`tool_type`, `safety_ws`, etc.) from bridge-static config.

Maps `FromRobotToHCMessage` (returned by KUKA) → `SensorBundle.tool`:
- `position`, `ori_xyz` → `ToolState.pose` (after inverse hand-eye transform to camera frame).
- `tool_force_vector` → `ToolState.wrench` (after frame transform).
- `tool_actuation_override` (echo) → `ToolState.jaw`.
- `robot_safety`, `motion_enabled` → `SensorBundle.safety`.
- `fk_angles` → `get_joint_state()` (off-path query).

### 9.2 neuroArm bridge

Converts `RobotCommand(CARTESIAN_TWIST in CAMERA)` → neuroArm Cartesian pose target:

1. Transform twist to robot base: same hand-eye Adjoint.
2. Integrate one cycle: $T_{k+1} = T_k \oplus (\mathrm{twist}_{\text{robot}} \cdot \mathrm{cycle\_dt})$ via SE(3) exponential map.
3. Submit $T_{k+1}$ as the next pose target.

Both bridges are ~10–20 lines of glue; they own real-time concerns (timeouts, heartbeats) that the env does not.

---

## 10. Explicit deletions

These existing artifacts are deleted as part of implementation, with no compatibility shim:

| File / symbol | Replaced by |
|---|---|
| `src/auto_surgery/schemas/commands.py` `RobotCommand` (current) | `RobotCommand` as defined in §5.3 |
| `src/auto_surgery/schemas/commands.py` `mode_flags: dict[str, Any]` | named fields only |
| `src/auto_surgery/schemas/commands.py` `cartesian_pose: dict[str, Any]` | typed `Pose` |
| `src/auto_surgery/schemas/commands.py` `gripper: float` | `tool_jaw_target` |
| `src/auto_surgery/schemas/results.py` `info: dict[str, Any]` | removed; explicit fields cover prior use |
| `src/auto_surgery/schemas/results.py` `SceneGraph` import | `SceneGraph` no longer part of step result |
| `src/auto_surgery/schemas/sensors.py` `modalities: dict[str, Any]` | typed `ToolState`, `cameras`, `safety` |
| `src/auto_surgery/schemas/scene.py` `SceneGraph` (per-step usage) | dropped from step path; off-path query if ever needed |
| `EnvConfig.domain_randomization: dict[str, Any]` | typed `DomainRandomizationConfig` (defined in piece 4) |
| Architecture §8.1 `Environment` protocol stub | protocol as defined in §5.7 |
| `src/auto_surgery/env/sofa_orchestration.py` `SofaEnvironment` adapter | rewritten against new protocol (no `SceneGraph`, no `info`) |
| `src/auto_surgery/env/sofa_backend.py` `_NativeSofaBackend` | rewritten against new protocol |
| `src/auto_surgery/env/action_generators.py` (legacy joint sine) | replaced by piece 3 motion generator producing `CARTESIAN_TWIST` commands |
| `src/auto_surgery/recording/brain_forceps.py` `_build_action` / `--joint-*` flags | replaced by twist-emitting CLI driven by piece 3 |

Migration is a single commit per piece, not a rolling deprecation. Callers update at the same time the schema lands.

---

## 11. Open questions deferred to later pieces

- **CARTESIAN_FORCE control mode.** Force/impedance control is not in the primary v1 path. If required later (contact-rich autonomy), add a `force_target: Vec3` payload and route through a force-controlled forceps body in SOFA.
- **Multi-arm.** Single-tool in v1. The schema does not preclude multi-arm; `RobotCommand` becomes one-per-arm with an `arm_id` discriminator when needed.
- **Stereo cameras.** `SensorBundle.cameras: list[CameraView]` accommodates them; only one element in v1.
- **Depth and segmentation modalities.** Plumbed (`frame_depth: bytes | None`) but unused in v1; activate when stereo/depth phase lands.
- **Per-tick mesh state.** Deliberately excluded from `StepResult` for performance; expose via an off-path `env.get_mesh_state()` query if forward-model training ever needs it.
- **Sim-tick / control-tick decoupling.** v1 couples them. If FEM stability ever forces a smaller sim tick than the control tick, the env adds internal sub-stepping at that time; the schema is unaffected.

---

## 12. Implementation completion checklist

- `RobotCommand` now requires `cycle_id`, `enable`, and mode-specific payloads; legacy free-form fields removed.
- `StepResult` is the only per-tick contract shape from env reset/step, exposing `sensors`, `dt`, `sim_step_index`, and `is_capture_tick`.
- `SofaEnvironment` and protocol surface now own the tick order and gating boundary (`stale_cycle_id`, `disabled`) and propagate `SafetyStatus`.
- Runtime backends populate typed `SensorBundle` directly; legacy payload buckets (`scene`, `modalities`, `sensor_observation`, `info`) are not part of step outputs.
- Downstream rollouts/trainers now consume `step.sensors` and read scene snapshots via `env.get_scene()`.

### Known deferrals

- Command-to-safety conversions outside env (bridge-side, transport, and real-time transport jitter handling).
- `CARTESIAN_FORCE` as a first-class control mode.
- Contact-rich force-specific force metrics beyond `ToolState.wrench` and `Contact`.

## 13. Pointer to piece 2

`RobotCommand(CARTESIAN_TWIST in CAMERA)` is meaningless until the forceps becomes a rigid body that can be commanded and that interacts with the brain FEM. Piece 2 — **Physical forceps in SOFA** — replaces the current `OglModel`-only `Forceps` node with:

- A `Rigid3d` `MechanicalObject` for tool pose.
- `LineCollisionModel` / `TriangleCollisionModel` on the tool surface, wired into the existing `CollisionPipeline`.
- A `RigidMapping` from rigid pose to the visual mesh and the collision mesh.
- A jaw articulation (hinged sub-body) driven by `tool_jaw_target`.
- An action applier that consumes `CARTESIAN_TWIST` commands and writes rigid-body twist into SOFA, with the camera→scene frame transform handled at the applier boundary.

Piece 2's design doc opens after this one is reviewed and committed.
