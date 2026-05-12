# Domain Randomization Framework — Design

| Field | Value |
|---|---|
| Date | 2026-05-12 |
| Status | Approved (design); implementation pending |
| Piece | 4 of 5 (in the simulation-pipeline redesign) |
| Depends on | `docs/specs/2026-05-12-command-schema-and-env-step-contract-design.md` (piece 1), `docs/specs/2026-05-12-physical-forceps-design.md` (piece 2), `docs/specs/2026-05-12-scene-aware-motion-generator-design.md` (piece 3) |
| Supersedes | `DomainRandomizationConfig` placeholder in `src/auto_surgery/schemas/manifests.py` and the `EnvConfig.domain_randomization` slot declared in piece 1 |
| Backward compatibility | **None.** Piece 1 is the new floor; this piece replaces the placeholder DR config and amends `SceneConfig` / `VisualOverrides` from pieces 2–3 in the same PR. |

---

## 1. Context and scope

Piece 1 defined a typed action / observation contract. Piece 2 made the forceps a physical body that contact-couples to the brain FEM. Piece 3 produced surgeon-like, contact-reactive motion (with primitive-sequence variety across episodes). Each of those pieces emits its episodes against *one* `(SceneConfig, MotionGeneratorConfig)` pair: the brain has the same shape, the camera sits in the same spot, the lighting is whatever SOFA's default viewer happened to apply, the texture is the same `texture_outpaint.png`, and the FEM constants are whatever was hard-coded in `brain_dejavu_forceps_poc.scn`. Only the tool trajectory varies. An IDM trained on this output would overfit instantly to the simulator's visual fingerprint and fail on internet surgical footage, which is precisely the task the architecture exists to solve.

Piece 4 closes that gap by adding a **per-episode randomisation framework** that produces a fresh, fully-realised `(SceneConfig, MotionGeneratorConfig)` pair per episode, sampled from typed distributions. The framework covers five axes the user called out explicitly — *mesh warping, texture tinting, camera orientation, deformation constants, topology* — plus lighting and background colour as a sixth IDM-critical signal. The framework itself is a pure-Python module with no SOFA dependency; the env consumes the realised `SceneConfig` as an opaque "this is the scene for this episode" description, exactly as it does today.

Two adjacent gaps get fixed in the same PR because they would otherwise rot:

1. `SceneConfig.camera_extrinsics_scene` and `SceneConfig.camera_intrinsics` exist on the typed surface but are not consumed anywhere in the SOFA backend; capture uses hard-coded defaults inside `attach_capture_camera`. Piece 4's env-integration step wires both fields through the rendered `.scn` template (§7.6, §9.2) so the realised `SceneConfig.camera_*` actually affects the rendered frame.
2. `VisualOverrides.clasper_color` is declared on the pydantic model but the forceps applier ignores it. Piece 4 applies it.

This is **piece 4 of 5**:

1. Command schema + environment-step contract *(piece 1, complete)*
2. Physical forceps in SOFA *(piece 2, approved; implementation landed in `feat: finalize Stage-1 command contract and SOFA rollout pipeline`)*
3. Human-like, scene-aware motion generator *(piece 3, approved; implementation in flight)*
4. Domain randomization framework *(this spec)*
5. Episodic recorder + paired (motion-label, video) artefacts

---

## 2. Goals

1. Provide a pure-function sampler `sample_episode(base_scene, base_motion, randomization, episode_seed) -> EpisodeSpec` that returns the realised `(SceneConfig, MotionGeneratorConfig, SampleRecord)` triple plus the episode seed.
2. Extend `SceneConfig` to be the **complete physical description of one episode**: tissue material (FEM constants), tissue topology (`SparseGridTopology n`), tissue mesh perturbation (warp), lighting, plus the existing tool / camera / target-volume fields. The env consumes one `SceneConfig`; piece 4 sits *above* it.
3. Cover the user-listed randomisation axes in v1:
   * Mesh warp (anisotropic scale + bounded translation + optional Gaussian bulge anchored at a target-volume centre).
   * Texture / material tint (per-surface RGB multipliers on tissue, forceps body, claspers, plus background colour).
   * Camera (look-at sampler centred near a target volume, with bounded position / orientation / intrinsics jitter).
   * FEM constants (Young's modulus, Poisson ratio, total mass, Rayleigh damping).
   * Topology (`SparseGridTopology n` over a small Choice set).
   * Lighting (one `DirectionalLight` + one `SpotLight` with direction / intensity / colour ranges; sixth axis added because lighting variation is the single biggest IDM-generalisation signal for real surgical footage).
4. Define a small, typed distribution language — `Range`, `LogRange`, `Choice[T]`, `Vec3Range` — sufficient for every v1 knob, mirroring piece 3's `MotionGeneratorConfig` "tuple range" pattern so the same pydantic-YAML idiom carries across pieces 3 and 4.
5. Ship a preset library at `configs/randomization/{minimal,default,aggressive}.yaml` with calibrated ranges; the recorder picks one by name. Knob-level overlays atop a preset are out of scope for v1 (see §4.7).
6. Derive every sub-RNG from a single `episode_seed` via named `numpy.random.SeedSequence` sub-streams, so a new randomisation axis added later does not shift the seeds of existing axes (reproducibility under code churn).
7. Apply sampled physics + topology + lighting + texture-tint + mesh-warp values to SOFA via a **single canonical path**: render a per-episode `.scn` file from a Jinja-style template (extending the existing `dejavu_paths.render_dejavu_scene_template` pattern) and pass the rendered path to the SOFA loader at `reset()` time. Mesh warping writes warped `.obj` files to disk and substitutes the paths inside the rendered `.scn`.
8. Wire `SceneConfig.camera_extrinsics_scene` and `SceneConfig.camera_intrinsics` into `attach_capture_camera` so the sampler's camera output reaches the rendered frame.
9. Record the realised `EpisodeSpec` (full scene, full motion, sample record, episode seed) alongside the video / command stream so piece 5 can produce paired (motion-label, video) artefacts whose **inputs** are reproducibly replayable from one JSON blob (bit-identical SOFA frames across SOFA versions is non-goal §3.10).

## 3. Non-goals

1. **Free-form deformation (FFD) lattice for tissue.** v1 ships anisotropic scale + radial bulge only. FFD is a clean v2 extension: a new `MeshPerturbation` sub-field plus a separate warp function, with no change to the env contract.
2. **Per-tick lighting flicker / camera shake / texture animation.** v1 is per-episode; everything is sampled once at `reset()` and held constant for the rest of the episode. Per-tick noise is a future axis if internet-footage diversity demands it.
3. **Pre-curated texture asset library** (e.g., `brain_texture_{01..N}.png`). v1 randomises by tint multiplier on the existing texture. Asset swaps need a curated library we don't have today.
4. **Multi-scene support** (kidney / liver / uterus). v1 commits one `SceneConfig` (`dejavu_brain`). The randomisation framework is scene-agnostic in shape; adding a scene is a YAML addition.
5. **Anatomy-aware target weighting.** `TargetVolume.label` exists on the pydantic model but v1 weights volumes uniformly when sampling camera look-at and motion targets. Per-label weights are a `CameraRandomization.lookat_target_volume_weights` hook with the v1 default being uniform.
6. **Forceps mesh warping.** The IDM is trained to recognise *real* surgical tools; randomising the tool geometry would hurt generalisation. v1 randomises only the forceps tint / clasper tint and leaves geometry untouched.
7. **Stochastic distribution shapes beyond uniform / log-uniform / categorical.** No truncated normal, no mixture-of-gaussians, no rejection sampling. The v1 trio is enough to express every IDM-relevant knob.
8. **Randomisation of safety thresholds, `control_rate_hz`, `dt`.** These are stability-critical parameters; randomising them couples FEM stability with sample quality and is out of scope.
9. **Curriculum / annealing of randomisation intensity over episodes.** v1 samples i.i.d. from fixed distributions. A future curriculum sampler can wrap `sample_episode` without changing the env contract.
10. **Replay of a fixed `EpisodeSpec` across SOFA versions.** SOFA upstream changes between releases may produce different visuals or contact responses for the same `.scn`; piece 4 does not commit to bit-identical replay across SOFA versions, only across runs within a single SOFA version.

---

## 4. Foundational design decisions

### 4.1 Pure-function sampler at the recorder layer

Piece 4 lives **above** `env.reset()`. The recorder loop is:

```
for episode_idx in range(num_episodes):
    episode_seed = master_rng.integers(0, 2**63 - 1)
    spec = sample_episode(base_scene, base_motion, randomization, episode_seed)
    env.reset(EnvConfig(scene=spec.scene, seed=spec.episode_seed, ...))
    motion = SurgicalMotionGenerator(spec.motion, spec.scene)
    cmd = motion.reset(env_step)
    for tick in range(num_ticks):
        env_step = env.step(cmd)
        cmd = motion.next_command(env_step)
        if env_step.is_capture_tick:
            recorder.write_frame(env_step)
    motion.finalize(env_step)
    recorder.write_episode_manifest(spec, motion.realised_sequence)
```

The env stays oblivious to who sampled the values; it consumes `EnvConfig.scene` as the complete description of the episode. This matches the architecture piece 3 §16 declared and makes the env trivially testable against a hand-written `SceneConfig`.

### 4.2 Fat `SceneConfig` as the one canonical episode description

Every randomisable knob lands on `SceneConfig` (or one of its sub-models). The same model is:

* The input to `env.reset()` (one `SceneConfig` per episode).
* The output of `sample_episode()` (sampled `SceneConfig.tissue_material.young_modulus_pa`, etc.).
* The artefact serialised to disk in the per-episode manifest (full reproducibility from one JSON blob).
* The input to a future "replay this exact case" or "load this exact case from a real surgery" workflow.

The alternative (a separate `DomainRandomizationConfig` sidecar consumed by the env) keeps two parameter homes alive forever and forces the env to learn DR semantics. We reject it.

The `EnvConfig.domain_randomization` field declared in piece 1 is removed (see §13.1). It was a placeholder; piece 4's chosen architecture has no use for it on `EnvConfig`.

### 4.3 Named-substream seeding

Every per-axis RNG is derived from `episode_seed` via `numpy.random.SeedSequence(entropy=episode_seed, spawn_key=tuple(map(ord, axis_name)))`. Adding a new axis later does not shift the seeds of existing axes — a property ordinal `spawn(n)` doesn't have.

The motion generator's `seed` field is treated as a derived value: `sample_episode` overwrites `MotionGeneratorConfig.seed` with the `motion` sub-stream's first `uint64` draw, so the recorder no longer needs to manage two seeds.

### 4.4 `.scn` template as the SOFA application path

All sampled scene parameters reach SOFA through one mechanism: a per-episode `.scn` file rendered from a Jinja2-style template by `randomization/scn_template.py:render_scene_template(scene, *, dejavu_root)`. The function extends the existing `dejavu_paths.render_dejavu_scene_template` pattern (which already substitutes `${DEJAVU_ROOT}`) by adding double-brace `{{ varname }}` placeholders for the new knobs. Two placeholder syntaxes coexist:

* `${DEJAVU_ROOT}` — substituted by piece 4's renderer, same semantics as today.
* `{{ varname }}` — new in piece 4, regex-substituted from a pydantic-derived context dict.

The choice of `{{ }}` is deliberate: it does not collide with `${...}` so existing DejaVu-style env-var references can coexist in the template file.

The rendered `.scn` is written to a `NamedTemporaryFile` with prefix `auto-surgery-brain-dejavu-episode-` (mirroring the existing convention) and the path is passed to SOFA's loader. The temp file is left on disk for the lifetime of the env (SOFA may re-read it during init); it is deleted in `SofaEnvironment.close()`. Per-episode rendering is microseconds; per-episode SOFA reload dominates wall time and is unchanged.

The alternative — programmatic mutation of SOFA `Data` fields after load — is rejected because mesh-path and `SparseGridTopology n` changes still require a full reload, so the speed advantage disappears, and the post-load mutation path is harder to inspect (the user can `cat /tmp/auto-surgery-brain-dejavu-episode-XXX.scn` to see exactly what SOFA loaded).

### 4.5 Mesh warp via temp `.obj` substitution

When `SceneConfig.tissue_mesh_perturbation` is non-identity, `randomization/mesh_warp.py:warp_tissue_meshes(volume_obj_path, surface_obj_path, perturbation) -> (volume_warped_path, surface_warped_path)` reads both source `.obj`s, applies the **same** warp function to both vertex arrays, and writes warped `.obj`s to `NamedTemporaryFile`s (prefix `auto-surgery-tissue-warp-`). The rendered `.scn` references the warped paths via the `{{ volume_mesh_path }}` / `{{ surface_mesh_path }}` placeholders.

Identity is the default (`scale = (1.0, 1.0, 1.0)`, `translation_scene = (0, 0, 0)`, `bulge = None`), in which case the warp step is skipped and the rendered `.scn` references the canonical DejaVu mesh paths directly. This makes "no randomisation" cheap (no I/O) and removes the warp from the smoke-path of every existing test.

### 4.6 Distribution language: `Range`, `LogRange`, `Choice[T]`, `Vec3Range`

Four primitives suffice for every v1 knob:

```python
class Range(BaseModel):
    model_config = {"extra": "forbid"}
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> "Range":
        if not self.low <= self.high:
            raise ValueError(f"Range requires low <= high; got [{self.low}, {self.high}].")
        return self

class LogRange(BaseModel):
    """Uniform in log10-space, samples are exp(log_uniform(log(low), log(high)))."""
    model_config = {"extra": "forbid"}
    low: float = Field(gt=0.0)
    high: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> "LogRange":
        if not self.low <= self.high:
            raise ValueError(f"LogRange requires low <= high; got [{self.low}, {self.high}].")
        return self

class Vec3Range(BaseModel):
    """Component-wise independent ranges."""
    model_config = {"extra": "forbid"}
    x: Range
    y: Range
    z: Range

T = TypeVar("T")

class Choice(BaseModel, Generic[T]):
    """Discrete categorical with optional weights (uniform if absent)."""
    model_config = {"extra": "forbid"}
    options: list[T] = Field(min_length=1)
    weights: list[float] | None = None

    @model_validator(mode="after")
    def _validate_weights(self) -> "Choice[T]":
        if self.weights is not None:
            if len(self.weights) != len(self.options):
                raise ValueError("Choice.weights length must match Choice.options.")
            if any(w < 0 for w in self.weights):
                raise ValueError("Choice.weights must be non-negative.")
            if sum(self.weights) == 0:
                raise ValueError("Choice.weights must have a positive sum.")
        return self
```

Each primitive exposes a typed `sample(rng: np.random.Generator) -> ScalarOrT` helper in `distributions.py`. Sampling is delegated to `numpy.random.Generator`; no fresh RNG creation inside `sample()`.

Truncated-normal and similar shapes are deferred; they can be added as additional primitives without changing the framework architecture.

### 4.7 Presets layer

Three preset files ship in `configs/randomization/`:

* `minimal.yaml` — narrow ranges. Used for smoke tests and the calibration acceptance gate; produces visibly varied frames but no FEM-instability risk.
* `default.yaml` — moderate ranges. Used by `recording/brain_forceps.py` as the recorder's default. Calibrated against the acceptance gate (§11.4).
* `aggressive.yaml` — wide ranges. Opt-in only; used for stress testing and "see how far we can push the IDM" experiments. Required to pass the acceptance gate but is not the default.

A preset is just a YAML serialisation of `EpisodeRandomizationConfig`. The recorder passes `--randomization-preset configs/randomization/default.yaml`. Overlays (e.g., "use the default preset but lock the topology") are not supported in v1; users who need overlays edit a copy of the preset.

### 4.8 Per-episode only

All sampling happens once at `sample_episode()` call time. The realised values are constant for the duration of one episode. Per-tick perturbation (lighting flicker, camera shake) is a future axis if internet-footage diversity demands it; it would land as a separate `TickRandomization` sub-config consumed by the env's per-step pipeline, not by the sampler.

---

## 5. Public types

All new pydantic models live in `src/auto_surgery/schemas/randomization.py`; helpers (sampler, mesh warp, template renderer) live in `src/auto_surgery/randomization/`. Both packages stay free of SOFA imports.

### 5.1 `SceneConfig` amendments (the "fat scene")

`SceneConfig` gains four new sub-models. Every new field is **optional** in the YAML sense — every new sub-model has a default that reproduces today's scene exactly, so a piece-3-era YAML still validates without modification.

```python
class TissueMaterial(BaseModel):
    """FEM constants for the tissue body in the rendered .scn."""

    model_config = {"extra": "forbid"}

    young_modulus_pa: float = Field(default=3000.0, gt=0.0)
    poisson_ratio: float = Field(default=0.45, ge=0.0, lt=0.5)
    total_mass_kg: float = Field(default=0.5, gt=0.0)
    rayleigh_stiffness: float = Field(default=0.1, ge=0.0)


class TissueTopology(BaseModel):
    """SparseGridTopology resolution for the tissue body."""

    model_config = {"extra": "forbid"}

    sparse_grid_n: tuple[int, int, int] = (16, 16, 16)

    @model_validator(mode="after")
    def _positive(self) -> "TissueTopology":
        if any(n <= 0 for n in self.sparse_grid_n):
            raise ValueError("sparse_grid_n components must be positive.")
        return self


class BulgeSpec(BaseModel):
    """Radial Gaussian displacement centred on a point in scene frame."""

    model_config = {"extra": "forbid"}

    center_scene: Vec3
    radius_m: float = Field(gt=0.0)
    amplitude_m: float  # signed: + outward, - inward indentation


class MeshPerturbation(BaseModel):
    """Per-episode warp applied to the tissue volume + surface meshes."""

    model_config = {"extra": "forbid"}

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    translation_scene: Vec3 = Field(default_factory=lambda: Vec3(x=0.0, y=0.0, z=0.0))
    bulge: BulgeSpec | None = None

    def is_identity(self) -> bool:
        return (
            self.scale == (1.0, 1.0, 1.0)
            and self.translation_scene == Vec3(x=0.0, y=0.0, z=0.0)
            and self.bulge is None
        )


class DirectionalLight(BaseModel):
    """One directional light source for the rendered .scn."""

    model_config = {"extra": "forbid"}

    direction_scene: Vec3  # need not be unit-norm; applier normalises
    intensity: float = Field(default=1.0, ge=0.0)
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)


class SpotLight(BaseModel):
    """One spot light source for the rendered .scn."""

    model_config = {"extra": "forbid"}

    position_scene: Vec3
    direction_scene: Vec3  # need not be unit-norm
    cone_half_angle_deg: float = Field(default=30.0, gt=0.0, le=89.0)
    intensity: float = Field(default=1.0, ge=0.0)
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)


class LightingSpec(BaseModel):
    """Lights and background for the rendered .scn."""

    model_config = {"extra": "forbid"}

    directional: DirectionalLight | None = None
    spot: SpotLight | None = None
    background_rgb: tuple[float, float, float] = (0.05, 0.05, 0.08)
```

Patched `SceneConfig`:

```python
class SceneConfig(BaseModel):
    model_config = {"extra": "forbid"}

    scene_id: Literal["dejavu_brain"] = "dejavu_brain"
    tissue_scene_path: Path | None = None
    scene_xml_path: str | None = Field(default=None, description="...")
    tool: ToolSpec = Field(default_factory=ToolSpec)
    camera_extrinsics_scene: Pose = Field(default_factory=_identity_pose)
    camera_intrinsics: CameraIntrinsics = Field(default_factory=_default_camera_intrinsics)
    target_volumes: list[TargetVolume] = Field(default_factory=_default_target_volumes, min_length=1)
    # new in piece 4:
    tissue_material: TissueMaterial = Field(default_factory=TissueMaterial)
    tissue_topology: TissueTopology = Field(default_factory=TissueTopology)
    tissue_mesh_perturbation: MeshPerturbation = Field(default_factory=MeshPerturbation)
    lighting: LightingSpec = Field(default_factory=LightingSpec)
```

### 5.2 `VisualOverrides` amendments

`VisualOverrides` gains two tint fields. `clasper_color` is also wired into the applier (it was declared in piece 2 but unused).

```python
class VisualOverrides(BaseModel):
    model_config = {"extra": "forbid"}

    body_uv_path: Path | None = None
    clasper_left_uv_path: Path | None = None
    clasper_right_uv_path: Path | None = None
    body_color: tuple[float, float, float, float] | None = None
    clasper_color: tuple[float, float, float, float] | None = None
    # new in piece 4:
    tissue_texture_tint_rgb: tuple[float, float, float] | None = None
    """Multiplicative RGB tint applied to the tissue OglModel diffuse component."""
```

The background colour lives on `LightingSpec.background_rgb`, not on `VisualOverrides`. `VisualOverrides` is a tool-scoped concept (per piece 2 §5.4) and we keep that scope.

### 5.3 Distribution primitives

Already shown in §4.6. They live in `src/auto_surgery/schemas/randomization.py` alongside the per-axis randomisation sub-models so YAML loaders see one module.

### 5.4 `EpisodeRandomizationConfig`

```python
class TissueMaterialRandomization(BaseModel):
    model_config = {"extra": "forbid"}
    young_modulus_pa: LogRange | None = None
    poisson_ratio: Range | None = None
    total_mass_kg: Range | None = None
    rayleigh_stiffness: Range | None = None


class TissueTopologyRandomization(BaseModel):
    model_config = {"extra": "forbid"}
    sparse_grid_n: Choice[tuple[int, int, int]] | None = None


class MeshPerturbationRandomization(BaseModel):
    model_config = {"extra": "forbid"}
    scale_x: Range | None = None
    scale_y: Range | None = None
    scale_z: Range | None = None
    translation_scene: Vec3Range | None = None
    bulge_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    bulge_target_volume_weights: dict[str, float] | None = None  # label -> weight, default uniform
    bulge_offset_within_volume_frac: Vec3Range | None = None     # fraction of volume.half_extents_scene, clamped to [-1, +1]
    bulge_radius_m: Range | None = None
    bulge_amplitude_m: Range | None = None


class CameraRandomization(BaseModel):
    model_config = {"extra": "forbid"}
    lookat_target_volume_weights: dict[str, float] | None = None  # label -> weight
    lookat_offset_within_volume_frac: Vec3Range | None = None     # fraction of volume.half_extents_scene, clamped to [-1, +1]
    base_offset_scene: Vec3 | None = None                          # camera offset from look-at, before jitter
    distance_jitter_scale: Range | None = None                     # dimensionless multiplier applied to ||base_offset||
    azimuth_deg: Range | None = None                               # rotate base_offset around world up
    elevation_deg: Range | None = None                             # tilt around the camera-right axis after azimuth
    roll_deg: Range | None = None
    fx_jitter_pct: Range | None = None                             # multiplicative, e.g. [-20, +20]
    fy_jitter_pct: Range | None = None
    principal_point_offset_px: Vec3Range | None = None             # only x,y components used
    resolution_choices: Choice[tuple[int, int]] | None = None


class LightingRandomization(BaseModel):
    model_config = {"extra": "forbid"}
    directional_direction_scene: Vec3Range | None = None
    directional_intensity: Range | None = None
    directional_color_rgb_tint: Vec3Range | None = None            # multiplied against (1,1,1) base
    spot_position_scene: Vec3Range | None = None
    spot_direction_scene: Vec3Range | None = None
    spot_cone_half_angle_deg: Range | None = None
    spot_intensity: Range | None = None
    spot_color_rgb_tint: Vec3Range | None = None
    background_rgb: Vec3Range | None = None


class VisualTintRandomization(BaseModel):
    model_config = {"extra": "forbid"}
    tissue_texture_tint_rgb: Vec3Range | None = None
    forceps_body_color_rgb: Vec3Range | None = None                # base alpha = 1.0
    forceps_clasper_color_rgb: Vec3Range | None = None


class EpisodeRandomizationConfig(BaseModel):
    """Top-level config consumed by sample_episode."""

    model_config = {"extra": "forbid"}

    tissue_material: TissueMaterialRandomization | None = None
    tissue_topology: TissueTopologyRandomization | None = None
    tissue_mesh: MeshPerturbationRandomization | None = None
    camera: CameraRandomization | None = None
    lighting: LightingRandomization | None = None
    visual_tint: VisualTintRandomization | None = None
```

Every axis is optional. An empty `EpisodeRandomizationConfig()` is valid and produces `EpisodeSpec.scene == base_scene` and `EpisodeSpec.motion == base_motion` (modulo the derived seed).

### 5.5 `EpisodeSpec` and `SampleRecord`

```python
class SampleRecord(BaseModel):
    """Per-axis log of what the sampler drew. Used in the per-episode manifest.

    `episode_seed` is not duplicated here; it lives on the parent `EpisodeSpec`.
    """

    model_config = {"extra": "forbid"}

    tissue_material: dict[str, float] = Field(default_factory=dict)
    tissue_topology: dict[str, tuple[int, int, int]] = Field(default_factory=dict)
    tissue_mesh: dict[str, Any] = Field(default_factory=dict)
    camera: dict[str, Any] = Field(default_factory=dict)
    lighting: dict[str, Any] = Field(default_factory=dict)
    visual_tint: dict[str, Any] = Field(default_factory=dict)


class EpisodeSpec(BaseModel):
    """Output of sample_episode: the realised episode description."""

    model_config = {"extra": "forbid"}

    episode_seed: int
    scene: SceneConfig
    motion: MotionGeneratorConfig
    sample_record: SampleRecord
```

`SampleRecord` exists so the manifest can serialise *both* the drawn values *and* (separately) the preset that produced them. It enables a future "the IDM did badly on this episode — what was sampled?" debug workflow without re-running the sampler.

### 5.6 Public function signatures

```python
# src/auto_surgery/randomization/sampler.py
def sample_episode(
    base_scene: SceneConfig,
    base_motion: MotionGeneratorConfig,
    randomization: EpisodeRandomizationConfig,
    episode_seed: int,
) -> EpisodeSpec: ...


# src/auto_surgery/randomization/mesh_warp.py
def warp_tissue_meshes(
    volume_obj_path: Path,
    surface_obj_path: Path,
    perturbation: MeshPerturbation,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Apply perturbation to both meshes; write warped .objs to temp; return paths."""


# src/auto_surgery/randomization/scn_template.py
def render_scene_template(
    scene: SceneConfig,
    *,
    dejavu_root: Path,
    template_path: Path | None = None,
) -> Path:
    """Render the per-episode .scn from a Jinja-style template. Return the rendered path."""


# src/auto_surgery/randomization/presets.py
def load_randomization_preset(path: Path) -> EpisodeRandomizationConfig:
    """yaml.safe_load + EpisodeRandomizationConfig.model_validate."""
```

`sample_episode` is the only function the recorder calls; the others are private to the env / SOFA backend.

---

## 6. Per-axis sampling semantics

All sampling lives in `randomization/sampler.py`. The sampler is **pure**: it takes the inputs, an `episode_seed`, and returns an `EpisodeSpec`. No I/O. No globals. Determinism is enforced by the named-substream seeding (§4.3).

### 6.1 Seeding hierarchy

```python
def _named_subrng(episode_seed: int, name: str) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=episode_seed, spawn_key=tuple(map(ord, name)))
    return np.random.default_rng(ss)


_AXIS_NAMES: tuple[str, ...] = (
    "tissue_material",
    "tissue_topology",
    "tissue_mesh",
    "camera",
    "lighting",
    "visual_tint",
    "motion",
)
```

Each named sub-RNG is created on demand inside the sampler. The `"motion"` sub-RNG produces a single `uint64` that overwrites `MotionGeneratorConfig.seed` in the returned `EpisodeSpec.motion`.

The `_AXIS_NAMES` tuple is a documented public constant; tests assert that the sampler creates sub-RNGs only for names in this set, so adding a new axis is a one-line tuple extension plus a sampler block.

### 6.2 Tissue material

```python
def _sample_tissue_material(
    base: TissueMaterial,
    rand: TissueMaterialRandomization | None,
    rng: np.random.Generator,
    record: dict[str, float],
) -> TissueMaterial:
    if rand is None:
        return base
    young = rand.young_modulus_pa.sample(rng) if rand.young_modulus_pa else base.young_modulus_pa
    poisson = rand.poisson_ratio.sample(rng) if rand.poisson_ratio else base.poisson_ratio
    total_mass = rand.total_mass_kg.sample(rng) if rand.total_mass_kg else base.total_mass_kg
    rayleigh = rand.rayleigh_stiffness.sample(rng) if rand.rayleigh_stiffness else base.rayleigh_stiffness
    record["young_modulus_pa"] = young
    record["poisson_ratio"] = poisson
    record["total_mass_kg"] = total_mass
    record["rayleigh_stiffness"] = rayleigh
    return TissueMaterial(
        young_modulus_pa=young,
        poisson_ratio=poisson,
        total_mass_kg=total_mass,
        rayleigh_stiffness=rayleigh,
    )
```

The `default.yaml` ranges (illustrative; final values land after the §11.4 calibration gate):

* `young_modulus_pa`: `LogRange(low=1000, high=10000)` — covers soft cortex to firmer tumour-adjacent tissue.
* `poisson_ratio`: `Range(low=0.35, high=0.48)` — near-incompressible biological tissue.
* `total_mass_kg`: `Range(low=0.3, high=0.7)`.
* `rayleigh_stiffness`: `Range(low=0.05, high=0.2)`.

### 6.3 Tissue topology

```python
def _sample_tissue_topology(
    base: TissueTopology,
    rand: TissueTopologyRandomization | None,
    rng: np.random.Generator,
    record: dict[str, tuple[int, int, int]],
) -> TissueTopology:
    if rand is None or rand.sparse_grid_n is None:
        return base
    n = rand.sparse_grid_n.sample(rng)
    record["sparse_grid_n"] = n
    return TissueTopology(sparse_grid_n=n)
```

`default.yaml` ships `Choice(options=[(12,12,12), (16,16,16), (20,20,20)], weights=[1.0, 2.0, 1.0])` — biased toward the canonical 16³ grid that the existing scene uses, with 12³ and 20³ as the diversity samples. Aggressive opens this to `(24,24,24)`.

### 6.4 Tissue mesh perturbation

```python
def _sample_mesh_perturbation(
    base: MeshPerturbation,
    rand: MeshPerturbationRandomization | None,
    target_volumes: list[TargetVolume],
    rng: np.random.Generator,
    record: dict[str, Any],
) -> MeshPerturbation:
    if rand is None:
        return base
    sx = rand.scale_x.sample(rng) if rand.scale_x else base.scale[0]
    sy = rand.scale_y.sample(rng) if rand.scale_y else base.scale[1]
    sz = rand.scale_z.sample(rng) if rand.scale_z else base.scale[2]
    translation = rand.translation_scene.sample(rng) if rand.translation_scene else base.translation_scene
    bulge = base.bulge
    if rng.random() < rand.bulge_probability:
        volume = _weighted_choice(target_volumes, rand.bulge_target_volume_weights, rng)
        frac = (
            rand.bulge_offset_within_volume_frac.sample(rng)
            if rand.bulge_offset_within_volume_frac is not None
            else Vec3(x=0, y=0, z=0)
        )
        frac = _clamp_vec3(frac, -1.0, 1.0)
        offset = _hadamard(frac, volume.half_extents_scene)
        center = _add(volume.center_scene, offset)
        radius = rand.bulge_radius_m.sample(rng) if rand.bulge_radius_m else 0.005
        amplitude = rand.bulge_amplitude_m.sample(rng) if rand.bulge_amplitude_m else 0.002
        bulge = BulgeSpec(center_scene=center, radius_m=radius, amplitude_m=amplitude)
    record["scale"] = (sx, sy, sz)
    record["translation_scene"] = translation.model_dump()
    record["bulge"] = bulge.model_dump() if bulge else None
    return MeshPerturbation(
        scale=(sx, sy, sz),
        translation_scene=translation,
        bulge=bulge,
    )
```

`_weighted_choice` interprets a missing label in `weights` as weight 0; an absent `weights` dict means uniform.

`default.yaml`:

* `scale_x/y/z`: `Range(low=0.92, high=1.08)` (independent per axis ⇒ anisotropic warp).
* `translation_scene`: `Vec3Range(x=Range(-0.005, 0.005), y=..., z=...)` — millimetre-scale jitter, in DejaVu scene units (which are scaled — see §10).
* `bulge_probability`: `0.5`.
* `bulge_offset_within_volume_frac`: `Vec3Range(x=Range(-0.8, 0.8), y=Range(-0.8, 0.8), z=Range(-0.8, 0.8))` — fractions of `half_extents_scene` (so the bulge centre stays near the volume's interior, not on its boundary).
* `bulge_radius_m`: `Range(0.003, 0.010)`.
* `bulge_amplitude_m`: `Range(-0.003, 0.005)` — signed; negative = inward indentation, positive = outward bump.

### 6.5 Camera

The camera sampler has the most geometry. The flow:

1. **Pick a look-at point.** Sample a target volume with weights from `lookat_target_volume_weights` (defaulting to uniform). Within that volume, sample a per-axis fraction from `lookat_offset_within_volume_frac` (clamped to `[-1, +1]`), multiply by `volume.half_extents_scene`, and add to `volume.center_scene`. The result `lookat_scene` is the camera's `lookAt` point.
2. **Compute the base offset.** Let `base_offset = rand.base_offset_scene if rand.base_offset_scene is not None else (base_scene.camera_extrinsics_scene.position - lookat_scene)`. The interpretation is "where was the camera, relative to the new look-at point". This is unambiguous because the look-at is what changed; the camera should move with it.
3. **Compute the camera position.** Apply, in order, to `base_offset`:
   1. Azimuth rotation: `R_world_up(azimuth_deg)` around the scene's world-up axis.
   2. Elevation rotation: `R_right(elevation_deg)` around the camera-right axis after step 1. The camera-right axis is `normalize(cross(world_up, -offset_after_azimuth))` (pointing right when looking from camera at look-at).
   3. Distance scaling: multiply the resulting vector's norm by `distance_jitter_scale`.

   `camera_position_scene = lookat_scene + final_offset`.

4. **Compute the orientation.** Build the look-at orientation from `camera_position_scene` toward `lookat_scene` with the scene's world-up axis as the up reference. Then apply a roll perturbation around the resulting `+z_camera` (camera forward, OpenCV convention).
5. **Sample intrinsics.** Multiplicative jitter on `fx`, `fy`; additive jitter on `cx`, `cy`; optional `Choice` over `(width, height)`. If a new resolution is sampled, scale `(cx, cy)` proportionally so the principal-point ratio is preserved before the additive jitter is applied.

OpenCV camera convention matches piece 3 §6.2: `+x_camera` right, `+y_camera` down, `+z_camera` into the scene. The look-at orientation derivation uses the scene's world-up axis (see implementation note below) as the up reference; the camera frame is constructed so that `+z_camera` points from the camera at the look-at point.

Implementation note on world-up: DejaVu scene convention is verified at implementation time (the existing `brain_dejavu_forceps_poc.scn` has `InteractiveCamera position="0 30 90" lookAt="0 0 0"`, which is consistent with `+y` up). The sampler reads world-up from a module-level constant `WORLD_UP_SCENE: Vec3 = Vec3(x=0, y=1, z=0)` updated alongside any future scene-convention change; it is not a YAML knob in v1.

The sampler enforces:

* `distance_jitter_scale` is multiplicatively bounded: even with extreme range values, the final offset norm cannot shrink below 25 % or grow above 400 % of `||base_offset||` (hard clamp inside the sampler, not just a per-knob `Range.low/high`).
* If `principal_point_offset_px` is sampled, the resulting `(cx, cy)` is clamped to `[0, width-1]` × `[0, height-1]`.
* `||camera_position_scene - lookat_scene|| >= 1e-2` (1 cm). If the combined jitter would produce a degenerate camera, the sampler raises (caught and re-attempted with a fresh draw from the same sub-RNG; max 8 retries before raising to the caller).

`default.yaml`:

* `lookat_offset_within_volume_frac`: `Vec3Range(x=Range(-0.5, 0.5), y=Range(-0.5, 0.5), z=Range(-0.5, 0.5))` — fractions of `half_extents_scene` (so the look-at lands roughly in the middle 50 % of the volume).
* `base_offset_scene`: `Vec3(x=0, y=0.08, z=0.15)` — DejaVu-frame "above and behind the surgical field".
* `distance_jitter_scale`: `Range(low=0.8, high=1.2)` (dimensionless multiplier).
* `azimuth_deg`: `Range(low=-30, high=30)`.
* `elevation_deg`: `Range(low=-20, high=20)`.
* `roll_deg`: `Range(low=-10, high=10)`.
* `fx_jitter_pct`: `Range(low=-10, high=10)` (so sampled `fx = base_fx * uniform(0.9, 1.1)`).
* `fy_jitter_pct`: same.
* `principal_point_offset_px`: `Vec3Range(x=Range(-20, 20), y=Range(-20, 20), z=Range(0, 0))`.
* `resolution_choices`: omitted in `default.yaml` (fixed resolution); aggressive ships `Choice(options=[(640, 480), (720, 540), (800, 600)])`.

### 6.6 Lighting

The base scene has **no explicit `LightManager` / lights**; the SOFA viewer falls back to its default. Piece 4's `.scn` template introduces a `LightManager` plus optionally a `DirectionalLight` and a `SpotLight`. If `SceneConfig.lighting.directional is None` and `.spot is None`, the rendered `.scn` omits the entire `LightManager` block, falling back to the viewer default (so the v1 default exactly reproduces today's look).

`default.yaml` always populates both lights:

* Directional: `direction_scene` sampled from `Vec3Range(x=Range(-0.5, 0.5), y=Range(-1.0, -0.3), z=Range(-0.5, 0.5))` — a downward-ish vector (in scene `+y_up` convention) with horizontal variation. The renderer normalises before writing.
* Directional intensity: `Range(low=0.5, high=1.5)`.
* Directional colour: `Vec3Range(x=Range(0.9, 1.0), y=Range(0.9, 1.0), z=Range(0.9, 1.0))` — slight warm/cool tint.
* Spot position: `Vec3Range` over a box above the surgical field.
* Spot direction: `Vec3Range` pointing roughly toward the brain centre, normalised at render time.
* Spot cone: `Range(low=20.0, high=45.0)` degrees.
* Spot intensity / colour: similar to directional.

Spot lights are surgically realistic (overhead surgical lamp pointing at the field).

### 6.7 Visual tints

```python
def _sample_visual_tint(
    base_overrides: VisualOverrides | None,
    rand: VisualTintRandomization | None,
    rng: np.random.Generator,
    record: dict[str, Any],
) -> VisualOverrides:
    base = base_overrides or VisualOverrides()
    if rand is None:
        return base
    body_color = base.body_color
    if rand.forceps_body_color_rgb is not None:
        rgb = rand.forceps_body_color_rgb.sample(rng)
        body_color = (rgb.x, rgb.y, rgb.z, 1.0)
    # (analogous for clasper_color and tissue_texture_tint_rgb)
    record["forceps_body_color_rgba"] = body_color
    record["forceps_clasper_color_rgba"] = ...
    record["tissue_texture_tint_rgb"] = ...
    return base.model_copy(update={
        "body_color": body_color,
        "clasper_color": ...,
        "tissue_texture_tint_rgb": ...,
    })
```

`default.yaml` ships tints close to greyscale-neutral (multiplier per channel `[0.85, 1.15]`) so tissue retains plausible colour. Aggressive opens this wider.

Each sampled `VisualOverrides` is attached to `EpisodeSpec.scene.tool.visual_overrides`. The forceps factory's `_apply_visual_overrides` is amended to:

1. Apply `body_color` to the shaft, both claspers (current behaviour).
2. Apply `clasper_color` to the left and right clasper `OglModel`s only (fixing the piece-2 gap).
3. If `body_color` and `clasper_color` are both present, `clasper_color` wins on claspers (more specific).

Tissue tint flows separately via the `.scn` template's tissue `OglModel` material string (§7.4).

### 6.8 Motion

```python
def _override_motion_seed(base: MotionGeneratorConfig, rng: np.random.Generator) -> MotionGeneratorConfig:
    new_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    return base.model_copy(update={"seed": new_seed})
```

The motion sub-RNG draws one `uint64` and writes it into `MotionGeneratorConfig.seed`. The rest of the motion config is passed through unchanged. This means the recorder's CLI `--seed` is the master `episode_seed`; the user no longer manages two seeds.

`SampleRecord` does not record the motion sub-seed separately because it is fully determined by `episode_seed`; the realised `EpisodeSpec.motion.seed` is the record.

---

## 7. Per-axis SOFA application

The env stays oblivious to who sampled the values. The application path for each axis lives in either `randomization/scn_template.py` (most axes) or `env/sofa_scenes/forceps.py` (`VisualOverrides`, already mostly wired). All paths reach SOFA at `_NativeSofaBackend.reset(config)` time.

### 7.1 `.scn` template structure

A new file `src/auto_surgery/env/sofa_scenes/brain_dejavu_episodic.scn.template` replaces the role of `brain_dejavu_forceps_poc.scn`. It is **valid SOFA XML** when every `{{ varname }}` is left in place; we keep it loadable as-is for syntax tooling, but SOFA will of course reject unrendered placeholders.

Key placeholders (full list lives next to the template):

| Placeholder | Source on `SceneConfig` |
|---|---|
| `{{ dt }}` | derived from env `control_rate_hz` (constant; not randomised) |
| `{{ gravity }}` | constant `"0 0 0"` |
| `{{ background_rgb }}` | `lighting.background_rgb` |
| `{{ light_manager_block }}` | conditional `LightManager + DirectionalLight + SpotLight` |
| `{{ tissue_volume_mesh_path }}` | `tissue_mesh_perturbation` warped path, or canonical |
| `{{ tissue_surface_mesh_path }}` | ditto |
| `{{ tissue_texture_path }}` | canonical (tint applied to material, not texture) |
| `{{ tissue_material_string }}` | `MAT Diffuse Color R G B 1.0` with `(R,G,B)` = `1 * tint` |
| `{{ young_modulus }}` | `tissue_material.young_modulus_pa` |
| `{{ poisson_ratio }}` | `tissue_material.poisson_ratio` |
| `{{ rayleigh_stiffness }}` | `tissue_material.rayleigh_stiffness` |
| `{{ total_mass }}` | `tissue_material.total_mass_kg` |
| `{{ sparse_grid_n }}` | `tissue_topology.sparse_grid_n` rendered as `"nx ny nz"` |
| `{{ camera_block }}` | `OffscreenCamera` configured from `camera_extrinsics_scene` + `camera_intrinsics` |
| `{{ forceps_block }}` | unchanged from today; tool factory still owns it |

`render_scene_template` does (in order):

1. Read the template file.
2. Render the `light_manager_block` from `scene.lighting`. If both lights are `None`, render an empty string (no lights).
3. Render the `camera_block` from `scene.camera_extrinsics_scene` and `scene.camera_intrinsics`. Use a SOFA `OffscreenCamera` so the recorder's offscreen capture sees the configured intrinsics. The `InteractiveCamera` used by the SOFA GUI is not emitted; piece 4 is recording-first.
4. Render the `tissue_material_string` by clamping `1 * tint` per channel into `[0, 1]`.
5. If `scene.tissue_mesh_perturbation.is_identity()`, set `tissue_volume_mesh_path` and `tissue_surface_mesh_path` to the canonical DejaVu paths (resolved via `resolve_dejavu_asset_path`). Otherwise call `warp_tissue_meshes(...)` and use the warped paths.
6. Substitute `${DEJAVU_ROOT}` with the resolved root (same as `render_dejavu_scene_template` today).
7. Regex-substitute all `{{ varname }}` placeholders against the rendered context dict. Any leftover `{{ ... }}` is a bug; the renderer raises a `ValueError("Unrendered placeholders: ...")`.
8. Write the rendered XML to a `NamedTemporaryFile` with prefix `auto-surgery-brain-dejavu-episode-` and return the path.

### 7.2 Mesh warp

`warp_tissue_meshes` uses `trimesh` (already in `pyproject.toml`):

```python
def warp_tissue_meshes(
    volume_obj_path: Path,
    surface_obj_path: Path,
    perturbation: MeshPerturbation,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    if perturbation.is_identity():
        return (volume_obj_path, surface_obj_path)
    out_volume = _warp_one(volume_obj_path, perturbation, output_dir, suffix="-volume.obj")
    out_surface = _warp_one(surface_obj_path, perturbation, output_dir, suffix="-surface.obj")
    return (out_volume, out_surface)


def _warp_one(src: Path, p: MeshPerturbation, output_dir: Path | None, *, suffix: str) -> Path:
    mesh = trimesh.load(src, process=False, force="mesh")
    verts = mesh.vertices.astype(np.float64).copy()
    verts *= np.asarray(p.scale, dtype=np.float64)
    verts += np.asarray([p.translation_scene.x, p.translation_scene.y, p.translation_scene.z])
    if p.bulge is not None:
        c = np.asarray([p.bulge.center_scene.x, p.bulge.center_scene.y, p.bulge.center_scene.z])
        diff = verts - c
        d = np.linalg.norm(diff, axis=1, keepdims=True)
        # radial-direction displacement; sign of amplitude controls outward vs inward
        gauss = p.bulge.amplitude_m * np.exp(-0.5 * (d / p.bulge.radius_m) ** 2)
        direction = diff / np.maximum(d, 1e-9)
        verts += gauss * direction
    mesh.vertices = verts
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix="auto-surgery-tissue-warp-",
        delete=False,
        dir=output_dir,
    ) as handle:
        out_path = Path(handle.name)
    mesh.export(out_path, file_type="obj")
    return out_path
```

The radial-direction warp (rather than surface-normal warp) is intentional: it works identically on the volumetric `volume_simplified.obj` and the surface `surface_full.obj`, keeping the two registered after warping. Surface-normal warping would require computing normals on the volume mesh, which is ambiguous for tet meshes.

The Gaussian falloff is symmetric in scene-space; for the FEM `SparseGridTopology` (which voxelises the volume mesh) the warp produces a smoothly displaced voxel grid in the bulge region.

### 7.3 FEM constants

The rendered `.scn`'s `TetrahedronFEMForceField` line is:

```
<TetrahedronFEMForceField youngModulus="{{ young_modulus }}" poissonRatio="{{ poisson_ratio }}" rayleighStiffness="{{ rayleigh_stiffness }}" />
```

`UniformMass` becomes:

```
<UniformMass totalMass="{{ total_mass }}" />
```

`SparseGridTopology` becomes:

```
<SparseGridTopology n="{{ sparse_grid_n }}" position="@VolumeLoader.position" />
```

(Format spelt out so reviewers see the exact substitution surface; the actual template will include the surrounding `Brain` node and any required attributes per the existing `brain_dejavu_forceps_poc.scn` layout.)

### 7.4 Tissue tint

The brain `OglModel` material string today is `material="MAT Diffuse Color 1.0 0.7 0.7 1.0..."`. The template uses:

```
material="MAT Diffuse Color {{ tissue_diffuse_r }} {{ tissue_diffuse_g }} {{ tissue_diffuse_b }} 1.0 Ambient Color ... Specular ... Shininess ..."
```

`tissue_diffuse_{r,g,b}` is computed from `scene.tool.visual_overrides.tissue_texture_tint_rgb` (default `(1.0, 0.7, 0.7)` — the canonical scene colour) multiplied by the canonical base, then clamped to `[0, 1]`.

The texture itself (`texture_outpaint.png`) is unchanged; tint is applied at the diffuse-colour level so the texture's UV-mapped detail still shows through. This matches how OpenGL combines texture and diffuse colour by default.

### 7.5 Lighting block

```
<LightManager />
<DirectionalLight direction="{{ dir_x }} {{ dir_y }} {{ dir_z }}" color="{{ dir_r }} {{ dir_g }} {{ dir_b }}" />
<SpotLight position="{{ spot_x }} {{ spot_y }} {{ spot_z }}" direction="{{ spot_dx }} {{ spot_dy }} {{ spot_dz }}" cutoff="{{ spot_cutoff }}" color="{{ spot_r }} {{ spot_g }} {{ spot_b }}" />
```

The renderer normalises `direction_scene` before writing. SOFA's `SpotLight` `cutoff` attribute is the full cone angle, so we write `cone_half_angle_deg * 2`. (The exact SOFA attribute names — `cutoff` vs `coneAngle` — are verified against the SOFA version pinned in `pyproject.toml` during implementation; if the attribute is named differently, the template uses the SOFA-version-correct name.)

If `scene.lighting.directional is None` and `.spot is None`, the entire block is rendered as the empty string and the viewer default applies — preserving today's visual exactly when randomisation is disabled.

`intensity` multiplies the colour at render time (SOFA's `DirectionalLight` does not expose an independent intensity; the convention `color = base_color * intensity` is the standard workaround).

### 7.6 Camera block

```
<OffscreenCamera position="{{ cam_x }} {{ cam_y }} {{ cam_z }}" lookAt="{{ look_x }} {{ look_y }} {{ look_z }}" up="{{ up_x }} {{ up_y }} {{ up_z }}" widthViewport="{{ width }}" heightViewport="{{ height }}" fieldOfView="{{ fov_deg }}" />
```

`fieldOfView` is derived from intrinsics: `fov_deg = 2 * degrees(atan(height / (2 * fy)))`. The `OffscreenCamera` is what the recorder's `attach_capture_camera` already binds to (after the §9 plumbing change), so the rendered `.scn` carries the camera the offscreen capture sees.

`up` is derived from the rendered quaternion (in scene frame, the world `+y` is up unless rolled). Roll is baked into the quaternion before the up vector is extracted.

Principal-point offsets (`cx`, `cy` not at image centre) are passed via the offscreen pipeline post-render (cropping / shifting the captured frame); SOFA's `OffscreenCamera` does not directly support non-centred principal points, so piece 4 emulates them at the capture-side (`attach_capture_camera` crops or shifts the buffer). The CameraIntrinsics roundtrip through the recorder records the *requested* `(cx, cy)`; what hits the buffer matches.

### 7.7 Visual overrides on the forceps subtree

`_apply_visual_overrides` (in `env/sofa_scenes/forceps.py`) is amended:

* If `body_color` is set, apply to the shaft `OglModel` *and* the two clasper `OglModel`s (current behaviour).
* If `clasper_color` is set, override the two clasper `OglModel`s with `clasper_color` (more specific than `body_color`). The shaft keeps `body_color` (or its default).
* If `tissue_texture_tint_rgb` is set, **ignore it here** — it is applied via the `.scn` template (§7.4), not by the forceps factory. The forceps factory only owns forceps visuals.

---

## 8. Presets

Three preset YAMLs in `configs/randomization/`. Each is a full `EpisodeRandomizationConfig` serialisation; we ship them in the same PR as the sampler.

### 8.1 `minimal.yaml`

Tight ranges around the canonical scene. Every axis covered, but the variance is small enough that v1 unit tests can assert frame-to-frame similarity. Used as the smoke-test calibration anchor:

```yaml
tissue_material:
  young_modulus_pa: { low: 2500.0, high: 3500.0 }
  poisson_ratio: { low: 0.43, high: 0.47 }
tissue_mesh:
  scale_x: { low: 0.98, high: 1.02 }
  scale_y: { low: 0.98, high: 1.02 }
  scale_z: { low: 0.98, high: 1.02 }
  bulge_probability: 0.0
camera:
  azimuth_deg: { low: -5.0, high: 5.0 }
  elevation_deg: { low: -5.0, high: 5.0 }
  fx_jitter_pct: { low: -2.0, high: 2.0 }
  fy_jitter_pct: { low: -2.0, high: 2.0 }
lighting:
  directional_intensity: { low: 0.9, high: 1.1 }
visual_tint:
  tissue_texture_tint_rgb:
    x: { low: 0.97, high: 1.03 }
    y: { low: 0.97, high: 1.03 }
    z: { low: 0.97, high: 1.03 }
```

### 8.2 `default.yaml`

The numbers in §6 are the default ranges. This preset is the recorder's CLI default.

### 8.3 `aggressive.yaml`

Wide ranges; opens topology to `(24,24,24)`, intrinsic jitter to ±30 %, mesh scale to `[0.85, 1.15]`, bulge probability to 0.85, lighting intensity to `[0.3, 2.0]`. Must pass the §11.4 calibration gate. Opt-in only.

The preset values inside `default.yaml` (and the others) are not authoritative until the §11.4 acceptance gate passes; the YAMLs in the implementation PR are calibrated post-smoke-test.

---

## 9. Env integration

### 9.1 `SofaEnvironment` lifecycle changes

Two changes, both in the same PR:

1. **`tissue_scene_path` resolution moves into `reset`.** Today `SofaEnvironment.__init__` resolves the scene path once. Piece 4 moves the resolution into `reset(config)`: each `reset` call renders a fresh `.scn` from the template if `config.scene` requires it (any non-default tissue / lighting / mesh-perturbation field). If `config.scene` is exactly the canonical default, the renderer is bypassed and the canonical path is used directly (so today's behaviour is preserved when randomisation is disabled).

2. **Temp file lifecycle.** Per-episode temp paths (rendered `.scn`, warped `.obj`s) are tracked on `_NativeSofaBackend` and deleted when the *next* `reset` lands (or when `close()` is called). We do not delete inside the reset that created them because SOFA may re-open the file during its async init.

### 9.2 Camera plumbing (wiring the dead fields)

Responsibilities split cleanly between the template and the offscreen capture helper:

1. **Template** (§7.6): the rendered `.scn` contains the `OffscreenCamera` node with all attributes (`position`, `lookAt`, `up`, `widthViewport`, `heightViewport`, `fieldOfView`) derived from `scene.camera_extrinsics_scene` and `scene.camera_intrinsics`. The SOFA scene graph loaded at `reset()` time already has the right camera.

2. **`attach_capture_camera`** (in `env/sofa_scenes/sofa_rgb_native.py`): learns to take the same `CameraIntrinsics` so it can:
   * Allocate the offscreen RGB buffer at `(intrinsics.width, intrinsics.height)`.
   * Bind the buffer to the `OffscreenCamera` node the template wrote (look up by node name).
   * If `(intrinsics.cx, intrinsics.cy)` is not at image centre, perform principal-point post-processing: a translated crop / shift of the captured buffer so the recorder receives a frame whose principal point matches the requested `(cx, cy)`. The output frame stays exactly `(width, height)` in size; values outside the original buffer are filled with the rendered background colour (`scene.lighting.background_rgb`).

This change retires the hard-coded `position=(0, 30, 90)` defaults inside `attach_capture_camera` — the camera lives in the template, the capture helper only binds and post-processes.

### 9.3 Mesh path substitution

`render_scene_template` calls `warp_tissue_meshes` only when `scene.tissue_mesh_perturbation.is_identity()` is False. The canonical (no-warp) path resolves to `resolve_dejavu_asset_path("brain/volume_simplified.obj", root=dejavu_root)` and the equivalent for the surface mesh. Both paths are surfaced into the template context as `tissue_volume_mesh_path` / `tissue_surface_mesh_path`.

If the user has a custom DejaVu layout (different filenames), they can override the canonical paths via a new optional field `SceneConfig.tissue_assets: TissueAssetSpec | None`, with:

```python
class TissueAssetSpec(BaseModel):
    model_config = {"extra": "forbid"}
    volume_mesh_relative_path: str = "brain/volume_simplified.obj"
    surface_mesh_relative_path: str = "brain/surface_full.obj"
    texture_relative_path: str = "brain/texture_outpaint.png"
```

`SceneConfig.tissue_assets` defaults to `None`, in which case the canonical paths are used.

---

## 10. Recorder integration

`src/auto_surgery/recording/brain_forceps.py` gains a per-episode outer loop. New CLI flags:

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--randomization-preset` | path | `configs/randomization/default.yaml` | preset YAML to load |
| `--num-episodes` | int | `1` | how many episodes to record |
| `--master-seed` | int | required | seeds the per-episode `episode_seed` stream |

Existing flags `--scene-config`, `--motion-config` continue to work; they describe the **base** `SceneConfig` / `MotionGeneratorConfig` that the sampler perturbs.

Per-episode artefacts:

```
artifacts/<run_id>/episode_{idx:04d}/
  frames/                    # captured PNGs
  episode_spec.json          # full EpisodeSpec (scene, motion, sample_record, seed)
  motion_plan.json           # from SurgicalMotionGenerator.realised_sequence (piece 3)
  preset.yaml                # exact randomization preset YAML used (for reproducibility)
  control_commands.parquet   # per-tick RobotCommand stream (already exists from piece 3 work)
```

`episode_spec.json` is the canonical reproducibility artefact: with it and the same SOFA version, a future caller can re-run *exactly* this episode by passing `EpisodeSpec.scene` to `env.reset` and `EpisodeSpec.motion` to `SurgicalMotionGenerator`.

Per-episode rendering happens in-process; we do not parallelise across episodes in v1 (parallelisation is piece 5's concern).

---

## 11. Testing strategy

Four layers, mirroring piece 3.

### 11.1 Unit tests

* `tests/randomization/test_distributions.py` — `Range`, `LogRange`, `Choice`, `Vec3Range`:
  - Sample inside bounds.
  - Sampling is deterministic given a `np.random.Generator`.
  - Edge cases (`low == high` ⇒ constant).
  - `LogRange` is uniform in log10.
  - `Choice` honours `weights`; missing weights = uniform.

* `tests/randomization/test_mesh_warp.py`:
  - Vertex count preserved.
  - Identity perturbation ⇒ returned paths equal inputs (no I/O).
  - Pure scale ⇒ verts multiplied component-wise.
  - Bulge centred outside the mesh's AABB has no measurable effect (Gaussian falloff).
  - Bulge centred inside the AABB moves nearby verts radially.

* `tests/randomization/test_scn_template.py`:
  - All `{{ ... }}` placeholders are substituted (no leftovers).
  - Identity `MeshPerturbation` ⇒ canonical DejaVu paths in output.
  - Non-identity ⇒ warped temp paths in output.
  - Empty `LightingSpec` ⇒ no `LightManager` in output.

### 11.2 Component tests

* `tests/randomization/test_sampler.py`:
  - Determinism: same `episode_seed` ⇒ same `EpisodeSpec` (compare via `model_dump`).
  - Independence across axes: changing only `tissue_material` randomisation does not affect the sampled `camera` (named substreams).
  - Default `EpisodeRandomizationConfig()` ⇒ `EpisodeSpec.scene == base_scene` (modulo a re-emitted, structurally-equal `SceneConfig`).
  - Sub-stream stability: snapshot a sample for a fixed seed and assert byte-equality across runs.

* `tests/randomization/test_camera_lookat.py`:
  - For 100 sampled cameras with a single target volume, the look-at point is always inside the volume's bounding box (after offset).
  - For all samples, the camera position is at least 1 cm away from the look-at point (degenerate camera prevention).
  - Roll is bounded by `roll_deg.high`.

### 11.3 Integration tests

* `tests/integration/test_episode_sampling.py` (synthetic env, no SOFA):
  - Drive `sample_episode → SyntheticSimEnvironment.reset` for 5 episodes; assert each reset succeeds and the env sees the realised `SceneConfig`.

* `tests/integration/test_sofa_dr_smoke.py` (SOFA-backed; matches piece 2/3 smoke-test convention):
  - Use the `minimal.yaml` preset.
  - Run 3 episodes, 200 ticks each, with a fixed master seed.
  - Assert all three episodes complete (no FEM blowup, no contact-pipeline crash).
  - Assert the three captured frames at tick 100 differ (pixel-wise MSE > a small threshold) — randomisation actually moves the picture.

### 11.4 Calibration acceptance gate

Same shape as piece 3 §11.4. For each of `minimal.yaml`, `default.yaml`, `aggressive.yaml`:

* Sample 50 episodes (fixed master seed), record one frame per episode.
* Assert: zero FEM blowups, zero contact-pipeline crashes, all frames non-empty.
* Assert: max pairwise MSE between frames is below an upper bound (catches "the random tint is fully decorrelated noise") and the median pairwise MSE is above a lower bound (catches "nothing actually varies").

The gate is the prerequisite for committing the YAML values; the YAMLs in the implementation PR are the post-calibration values, not arbitrary hand-edits.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **FEM blowup under aggressive Young's-modulus / topology / mesh-warp combos.** | Three-tier preset library; aggressive must pass the §11.4 calibration gate; per-axis ranges are calibrated against contact-stiffness from piece 2. |
| **Mesh warp produces invalid tet mesh.** Bulge amplitude too high relative to local mesh resolution could fold tets. | Default `bulge_amplitude_m` bounded at 5 mm; SparseGridTopology re-voxelises the warped mesh, which absorbs small inversions. Smoke test asserts no episode crashes. |
| **Camera intrinsics changes break offscreen capture.** SOFA's `OffscreenCamera` does not natively support non-centred principal points; we emulate via post-process crop/shift. | The post-process crop preserves the requested resolution; if the requested `(cx, cy)` is more than 25 % off image centre, the renderer raises (we do not silently truncate). |
| **Lighting randomisation produces unrealistic frames.** Default lighting ranges are wide enough to include "no light hitting anything" or "everything washed out". | Default preset ranges are narrow ([0.5, 1.5] intensity); aggressive opens this and is opt-in only. |
| **DejaVu `${DEJAVU_ROOT}` interaction with `{{ }}` placeholders.** Two substitution mechanisms in one file. | Renderer applies `${DEJAVU_ROOT}` substitution **first**, then `{{ }}` placeholders. The placeholder regex `\{\{\s*(\w+)\s*\}\}` does not match `${...}`. Unit test asserts both pass through cleanly when both appear in a template. |
| **Per-episode .scn rendering performance.** Three I/O ops per episode (template read, warped `.obj` write × 2 + rendered `.scn` write). | All three are small files (<10 MB combined); SOFA reload dominates per-episode time at >100 ms. Profile in the smoke test. |
| **`MotionGeneratorConfig.seed` collision with `--seed` CLI flag.** Piece 3's recorder passes `args.seed` to motion; piece 4 derives the motion seed from `episode_seed`. | Piece 4's recorder retires `--seed` in favour of `--master-seed`; motion seed is no longer user-facing. Cross-piece amendment §13. |
| **Replay determinism across SOFA versions.** SOFA upstream may change contact dynamics. | Out of scope — recorded `EpisodeSpec.scene` reproduces the *inputs* deterministically; we do not claim bit-identical SOFA frames across SOFA releases. The §11.4 gate runs at the pinned SOFA version. |
| **Temp file accumulation.** Each episode writes 2–3 temp files; over 1000 episodes that's up to 3000 files in `/tmp`. | `_NativeSofaBackend` tracks temp paths on `self._episode_temp_paths` and deletes the previous episode's paths at the start of the next `reset`. `close()` deletes the final batch. |
| **`SceneConfig` schema growth (the "fat scene") makes the type harder to read.** | Mitigated by sub-models (`TissueMaterial`, `LightingSpec`, etc.); the top-level `SceneConfig` stays around 12 fields with each being a small typed sub-model rather than a flat list of scalars. |

---

## 13. Cross-piece amendments

### 13.1 Piece-1 amendment: drop `EnvConfig.domain_randomization`

Piece 1's `DomainRandomizationConfig` placeholder on `EnvConfig` is removed. The recorder samples upstream and passes a fully-realised `EnvConfig.scene`; the env has no use for a separate `domain_randomization` field. The `EnvConfig.seed` field is repurposed as the master `episode_seed` (semantic, not structural change).

Affected files:
* `src/auto_surgery/schemas/manifests.py`: remove `DomainRandomizationConfig` class and `EnvConfig.domain_randomization` field.
* `tests/env/test_contract.py`: remove the synthetic env's `domain_randomization` handling.
* `tests/integration/test_sim_to_training.py`, `tests/integration/test_idm_stage0.py`: drop references.
* `src/auto_surgery/training/sofa_smoke.py`: drop the `dict → DomainRandomizationConfig` normalisation (`logging/storage.py` references).

Piece 1's spec gains an amendment note at §A pointing to this section.

### 13.2 Piece-2 amendment: `VisualOverrides` and `clasper_color` wiring

* `VisualOverrides` gains `tissue_texture_tint_rgb: tuple[float, float, float] | None`.
* `env/sofa_scenes/forceps.py:_apply_visual_overrides` is amended to actually apply `clasper_color` to the two clasper `OglModel`s (fixing the piece-2 gap).
* No change to `ForcepsAssemblyParams` (mass / hinge / scale stay frozen-dataclass defaults). Forceps geometry randomisation is out of scope for v1.

Piece 2's spec §5.4 picks up the `tissue_texture_tint_rgb` field and the `clasper_color` wiring fix in a one-section diff.

### 13.3 Piece-3 amendment: `SceneConfig` fat fields

* `SceneConfig` gains `tissue_material`, `tissue_topology`, `tissue_mesh_perturbation`, `lighting` sub-models. All default to today's behaviour, so piece-3 YAMLs (`configs/scenes/dejavu_brain.yaml`) validate unchanged.
* `MotionGeneratorConfig.seed`: no schema change; piece 4 overrides it post-load.
* Recorder CLI: `--seed` → `--master-seed`; `--scene-config` / `--motion-config` retained; `--randomization-preset` and `--num-episodes` added.

Piece 3's spec §10 (recorder integration) is updated to reference the new outer loop.

### 13.4 Schemas re-export

`src/auto_surgery/schemas/__init__.py` gains:

```python
from .randomization import (
    EpisodeRandomizationConfig,
    EpisodeSpec,
    SampleRecord,
    Range,
    LogRange,
    Choice,
    Vec3Range,
    TissueMaterialRandomization,
    TissueTopologyRandomization,
    MeshPerturbationRandomization,
    CameraRandomization,
    LightingRandomization,
    VisualTintRandomization,
)
```

### 13.5 Sequencer's duplicate `MotionGeneratorConfig`

The exploration found a `@dataclass MotionGeneratorConfig` shadow in `motion/sequencer.py` alongside the canonical pydantic one in `schemas/motion.py`. Piece 4 does **not** fix this — it would expand the blast radius outside DR. A separate (one-line) follow-up consolidates them after piece 4 ships; piece 4's tests stay compatible with both definitions by only constructing the pydantic one.

---

## 14. Acceptance criteria

Piece 4 lands when:

1. `sample_episode` returns a valid `EpisodeSpec` for any well-formed `(base_scene, base_motion, randomization, seed)` triple.
2. All three presets (`minimal`, `default`, `aggressive`) pass the §11.4 calibration acceptance gate at the pinned SOFA version.
3. The recorder CLI runs `--num-episodes 3 --randomization-preset configs/randomization/default.yaml --master-seed 42` end-to-end and produces three episode directories with full `episode_spec.json` artefacts, each frame visibly different from the others.
4. The wired-up camera path produces frames whose visible content matches the `SceneConfig.camera_extrinsics_scene` look-at (assert: a contact-event occurs in the central 50 % of the frame for a `default.yaml` episode with a single target volume).
5. The `clasper_color` field, when set, visibly tints the claspers in a captured frame; the shaft retains `body_color`.
6. Removing `EnvConfig.domain_randomization` does not break any piece-1, piece-2, or piece-3 test.
7. `tests/randomization/test_sampler.py` snapshots reproduce byte-identical `EpisodeSpec`s across runs of the same Python / NumPy version.

---

## 15. Out of scope / deferred

| Item | Trigger for revisiting |
|---|---|
| Free-form deformation (FFD) lattice for tissue. | IDM training shows insufficient tissue-shape diversity with scale + bulge alone. |
| Per-tick lighting flicker / camera shake. | IDM struggles with real footage's temporal noise; structural addition (a new sub-config), not a refactor. |
| Pre-curated texture asset library (multiple `brain_texture_*.png`). | We acquire a curated library; trivial extension (`Choice[Path]` on `VisualOverrides.tissue_texture_path`). |
| Multi-scene support (kidney / liver / uterus). | A second `SceneConfig` YAML committed; the randomisation framework is scene-agnostic in shape. |
| Anatomy-aware target-volume weighting. | A scene ships with both `vessel` and `tumor` labels and per-label weighting actually changes IDM outcomes. |
| Forceps mesh warping. | Tool catalog expands or we have evidence the IDM overfits to forceps geometry. |
| Stochastic distribution shapes (truncated normal, mixture-of-gaussians). | A specific knob's distribution is meaningfully non-uniform in real surgical data. |
| Randomisation curriculum / annealing. | An RL or curriculum-IDM training loop wants this; can wrap `sample_episode`. |
| Replay determinism across SOFA versions. | Cross-version replay becomes load-bearing; would need SOFA-version pinning per artefact. |
| Per-knob override layering atop a preset. | Power users hit the "fork the preset YAML" friction; trivial extension. |
| Parallel per-episode rendering. | Wall-clock pressure on dataset generation; piece 5's concern, not piece 4's. |

---

## 16. Pointer to piece 5

Piece 4 produces a deterministic, fully-typed `EpisodeSpec` per call and writes the corresponding `episode_spec.json` alongside the recorded frames and motion plan. The remaining work to make this output a *dataset* lives in piece 5:

**Piece 5 — Episodic recorder + paired (motion-label, video) artefacts** — wraps piece 4's sampler in a batched, optionally parallelised recorder; consolidates the per-episode artefacts under a single dataset manifest (split / shard layout, content addressing of warped mesh files for replay, action-label decimation from control-rate twist commands to frame-rate aligned annotations, optional camera-frame transform of the action stream onto recorded frames, train / val / test splits). The IDM-training stack consumes piece 5's output directly. Piece 5's design doc opens after piece 4's implementation ships and we have empirical numbers on per-episode wall time, FEM-blowup rates under each preset, and visual diversity statistics on the recorded frames.
