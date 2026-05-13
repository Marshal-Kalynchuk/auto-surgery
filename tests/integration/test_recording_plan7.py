from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq

from auto_surgery.config import load_motion_config, load_scene_config
from auto_surgery.recording import brain_forceps
from auto_surgery.logging import storage
from auto_surgery.schemas.commands import Pose, Quaternion, RobotCommand, Twist, Vec3
from auto_surgery.schemas.motion import MotionGeneratorConfig
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.sensors import (
    CameraIntrinsics,
    CameraView,
    SafetyStatus,
    SensorBundle,
    ToolState,
)
from auto_surgery.schemas.scene import SceneConfig


def _build_fake_camera_bundle(*, timestamp_ns: int, width: int, height: int) -> SensorBundle:
    pose = Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    return SensorBundle(
        timestamp_ns=timestamp_ns,
        sim_time_s=timestamp_ns / 1e9,
        tool=ToolState(
            pose=pose,
            twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
            jaw=0.0,
            wrench=Vec3(x=0.0, y=0.0, z=0.0),
            in_contact=False,
        ),
        cameras=[
            CameraView(
                camera_id="capture",
                timestamp_ns=timestamp_ns,
                extrinsics=pose,
                intrinsics=CameraIntrinsics(
                    fx=1.0,
                    fy=1.0,
                    cx=0.0,
                    cy=0.0,
                    width=width,
                    height=height,
                ),
                frame_rgb=b"",
            )
        ],
        safety=SafetyStatus(
            motion_enabled=True,
            command_blocked=False,
            block_reason=None,
            cycle_id_echo=0,
        ),
    )


class _FakeCapture:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def pre_init_hook(self, root_node: object, config: Any) -> None:
        _ = root_node
        _ = config

    def capture(self, root_node, step_index: int) -> dict[str, object]:
        _ = root_node
        _ = self._height
        _ = self._width
        return {"encoding": "image/png", "bytes": f"{step_index}".encode("utf-8")}


class _FakeEnv:
    scene_configs: list[SceneConfig] = []

    def __init__(
        self,
        *,
        scene_config: SceneConfig,
        step_dt: float,
        pre_init_hooks: list,
        action_applier: str | None = None,
        **_: object,
    ) -> None:
        self.scene_config = scene_config
        _ = step_dt
        _ = action_applier
        _FakeEnv.scene_configs.append(scene_config)
        self._frame_index = 0
        self.sofa_scene_root = object()
        for hook in pre_init_hooks:
            hook(
                self.sofa_scene_root,
                SimpleNamespace(scene=scene_config),
            )
        self._width = 95
        self._height = 70

    def reset(self, config) -> StepResult:
        self._frame_index = 0
        self._seed = int(config.seed)
        return StepResult(
            sensors=_build_fake_camera_bundle(
                timestamp_ns=0,
                width=self._width,
                height=self._height,
            ),
            dt=0.01,
            sim_step_index=0,
            is_capture_tick=True,
        )

    def step(self, action: RobotCommand) -> StepResult:
        self._frame_index += 1
        return StepResult(
            sensors=_build_fake_camera_bundle(
                timestamp_ns=action.timestamp_ns,
                width=self._width,
                height=self._height,
            ),
            dt=0.01,
            sim_step_index=self._frame_index,
            is_capture_tick=True,
        )

    def close(self) -> None:
        return None


class _FakeMotionGenerator:
    def __init__(self, motion_config: MotionGeneratorConfig, scene_config: SceneConfig) -> None:
        _ = scene_config
        _ = motion_config
        self._counter = -1

    def reset(self, _initial_step: StepResult) -> RobotCommand:
        self._counter = 0
        return self.next_command(_initial_step)

    def next_command(self, _last_step: StepResult) -> RobotCommand:
        self._counter += 1
        return RobotCommand(
            timestamp_ns=1_000_000_000 + self._counter,
            cycle_id=self._counter,
            cartesian_twist=Twist(
                linear=Vec3(x=0.0, y=0.0, z=0.0),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            tool_jaw_target=0.0,
            enable=True,
        )

    def finalize(self, _last_step: StepResult) -> None:
        return None

    @property
    def realised_sequence(self) -> tuple:
        return ()


def _make_args(
    *,
    output_dir: Path,
    num_episodes: int,
    master_seed: int,
    use_legacy_seed: bool = False,
) -> list[str]:
    seed_arg = "--seed" if use_legacy_seed else "--master-seed"
    return [
        "--output-dir",
        str(output_dir),
        "--num-episodes",
        str(num_episodes),
        seed_arg,
        str(master_seed),
        "--ticks",
        "3",
        "--width",
        "95",
        "--height",
        "70",
        "--overwrite",
    ]


def _run_capture_with_fakes(
    *,
    output_dir: Path,
    num_episodes: int,
    master_seed: int,
    sample_episode: Callable[..., brain_forceps._EpisodeSpec] | None = None,
    render_scene_template: Callable[..., Path] | None = None,
    capture_cls: type[_FakeCapture] = _FakeCapture,
    use_legacy_seed: bool = False,
    monkeypatch,
) -> list[Path]:
    parser = brain_forceps.build_capture_parser()
    monkeypatch.setenv("AUTO_SURGERY_DEJAVU_ROOT", str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args(
        _make_args(
            output_dir=output_dir,
            num_episodes=num_episodes,
            master_seed=master_seed,
            use_legacy_seed=use_legacy_seed,
        )
    )
    monkeypatch.setattr(brain_forceps, "SofaEnvironment", _FakeEnv)
    monkeypatch.setattr(brain_forceps, "SurgicalMotionGenerator", _FakeMotionGenerator)
    monkeypatch.setattr(brain_forceps, "SofaNativeRgbCapture", capture_cls)
    if sample_episode is not None:
        monkeypatch.setattr(brain_forceps, "_sample_episode", sample_episode)
    if render_scene_template is not None:
        monkeypatch.setattr(brain_forceps, "render_scene_template", render_scene_template)
    else:
        monkeypatch.setattr(
            brain_forceps,
            "render_scene_template",
            lambda scene, **kwargs: Path(scene.tissue_scene_path),
        )
    return brain_forceps.run_capture_brain_forceps_pngs(args)


def test_capture_plan7_multi_episode_generates_artifacts(tmp_path, monkeypatch) -> None:
    written = _run_capture_with_fakes(
        output_dir=tmp_path,
        num_episodes=2,
        master_seed=11,
        monkeypatch=monkeypatch,
    )
    run_dir = tmp_path / "run_seed_11"
    assert run_dir.is_dir()

    assert (run_dir / "brain_forceps_sample_motion_plan.json").exists()
    assert (run_dir / "brain_forceps_sample_control_commands.parquet").exists()

    for episode_index in range(2):
        episode_dir = run_dir / f"episode_{episode_index:04d}"
        assert (episode_dir / "episode_spec.json").is_file()
        assert (episode_dir / "preset.yaml").is_file()
        assert (episode_dir / "motion_plan.json").is_file()
        assert (episode_dir / "control_commands.parquet").is_file()
        frames = sorted((episode_dir / "frames").glob("*.png"))
        assert len(frames) == 3

        episode_spec = json.loads((episode_dir / "episode_spec.json").read_text(encoding="utf-8"))
        assert isinstance(episode_spec["seed"], int)
        assert "motion_seed" in episode_spec["sample_record"]
        assert "randomization_preset" in episode_spec["sample_record"]

    assert len(written) == 6
    table = pq.read_table(run_dir / "brain_forceps_sample_control_commands.parquet")
    assert table.column("cycle_id").to_pylist()[:3] == [1, 2, 3]

    run_manifest_path = run_dir / "run_manifest.json"
    assert run_manifest_path.is_file()
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert len(run_manifest["session_manifest_paths"]) == 2
    assert run_manifest["dataset_id"] == "brain_forceps_11"

    for episode_index in range(2):
        session_id = f"episode_{episode_index:04d}"
        session_manifest_path = run_dir / storage.session_manifest_path("brain_forceps", session_id)
        run_metadata_path = run_dir / storage.run_metadata_path("brain_forceps", session_id)
        forceps_trace_path = run_dir / storage.forceps_trace_path("brain_forceps", session_id)
        assert session_manifest_path.is_file()
        assert run_metadata_path.is_file()
        assert forceps_trace_path.is_file()
        session_manifest = json.loads(session_manifest_path.read_text(encoding="utf-8"))
        relative_session_manifest = storage.session_manifest_path(
            "brain_forceps", session_id
        )
        assert run_manifest["session_manifest_paths"][episode_index].endswith(relative_session_manifest)
        assert "artifact_paths" in session_manifest
        assert (
            session_manifest["artifact_paths"]["forceps_contract_trace"]
            == storage.forceps_trace_path("brain_forceps", session_id)
        )
        assert (
            session_manifest["artifact_paths"]["run_metadata"]
            == storage.run_metadata_path("brain_forceps", session_id)
        )
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        assert run_metadata["run_manifest_path"] == "run_manifest.json"
        assert run_metadata["session_manifest_path"] == storage.session_manifest_path(
            "brain_forceps", session_id
        )
        assert (
            run_metadata["trace_artifact_paths"]["forceps_contract_trace"]
            == storage.forceps_trace_path("brain_forceps", session_id)
        )
        assert run_metadata["run_id"] == "run_seed_11"

        trace_table = pq.read_table(forceps_trace_path)
        assert trace_table.num_rows == 3
        trace_payloads = [
            json.loads(raw) for raw in trace_table.column("payload_json").to_pylist()
        ]
        assert trace_payloads
        first = trace_payloads[0]
        assert {"frame_index", "timestamp_ns", "command_cycle_id", "tool_pose", "tool_twist"} <= set(first)
        assert "contacts" in first


def test_capture_plan7_deterministic_seed_path(tmp_path, monkeypatch) -> None:
    first = _run_capture_with_fakes(
        output_dir=tmp_path / "first",
        num_episodes=2,
        master_seed=77,
        monkeypatch=monkeypatch,
    )
    _ = first
    second = _run_capture_with_fakes(
        output_dir=tmp_path / "second",
        num_episodes=2,
        master_seed=77,
        monkeypatch=monkeypatch,
    )
    _ = second

    first_run = tmp_path / "first" / "run_seed_77"
    second_run = tmp_path / "second" / "run_seed_77"
    first_specs = [
        json.loads((first_run / "episode_0000" / "episode_spec.json").read_text(encoding="utf-8")),
        json.loads((first_run / "episode_0001" / "episode_spec.json").read_text(encoding="utf-8")),
    ]
    second_specs = [
        json.loads((second_run / "episode_0000" / "episode_spec.json").read_text(encoding="utf-8")),
        json.loads((second_run / "episode_0001" / "episode_spec.json").read_text(encoding="utf-8")),
    ]
    assert [spec["seed"] for spec in first_specs] == [spec["seed"] for spec in second_specs]
    assert [
        spec["sample_record"]["motion_seed"] for spec in first_specs
    ] == [
        spec["sample_record"]["motion_seed"] for spec in second_specs
    ]


def test_capture_plan7_seed_compat_alias(tmp_path, monkeypatch) -> None:
    written = _run_capture_with_fakes(
        output_dir=tmp_path,
        num_episodes=1,
        master_seed=99,
        use_legacy_seed=True,
        monkeypatch=monkeypatch,
    )
    assert written
    assert (tmp_path / "run_seed_99").is_dir()


def _base_episode(seed: int, sample_record: dict[str, Any]) -> brain_forceps._EpisodeSpec:
    scene = load_scene_config("configs/scenes/dejavu_brain.yaml")
    motion = load_motion_config("configs/motion/default.yaml")
    return brain_forceps._EpisodeSpec(
        seed=seed,
        scene=scene,
        motion=motion,
        sample_record=sample_record,
    )


def test_capture_plan7_does_not_render_template_without_randomized_axes(
    tmp_path, monkeypatch
) -> None:
    render_calls: list[str] = []

    def _fake_render_scene_template(
        scene: SceneConfig,
        *,
        dejavu_root: Path,
        template_path: Path | None = None,
    ) -> Path:
        _ = scene
        _ = template_path
        render_calls.append(f"{dejavu_root}")
        rendered = tmp_path / "rendered.scn"
        rendered.write_text("<Node/>", encoding="utf-8")
        return rendered

    _FakeEnv.scene_configs.clear()
    _run_capture_with_fakes(
        output_dir=tmp_path,
        num_episodes=1,
        master_seed=123,
        monkeypatch=monkeypatch,
        sample_episode=lambda *args, **kwargs: _base_episode(
            seed=1,
            sample_record={"motion_seed": 1, "randomization_preset": {}},
        ),
        render_scene_template=_fake_render_scene_template,
    )
    assert render_calls == []
    assert len(_FakeEnv.scene_configs) == 1
    assert (
        "auto-surgery-brain-dejavu-" not in str(_FakeEnv.scene_configs[0].tissue_scene_path)
    )


def test_capture_plan7_renders_template_for_randomized_axes(
    tmp_path, monkeypatch
) -> None:
    render_calls: list[tuple[Path, Path | None]] = []

    def _fake_render_scene_template(
        scene: SceneConfig,
        *,
        dejavu_root: Path,
        template_path: Path | None = None,
        use_canonical_tissue_mesh_files: bool = False,
    ) -> Path:
        _ = scene
        output_path = tmp_path / "auto-surgery-brain-dejavu-randomized.scn"
        output_path.write_text("<Node/>", encoding="utf-8")
        render_calls.append((output_path, template_path))
        return output_path

    _FakeEnv.scene_configs.clear()
    _run_capture_with_fakes(
        output_dir=tmp_path,
        num_episodes=1,
        master_seed=124,
        monkeypatch=monkeypatch,
        sample_episode=lambda *args, **kwargs: _base_episode(
            seed=2,
            sample_record={"camera": {"intrinsics": {"fx": 900.0}}},
        ),
        render_scene_template=_fake_render_scene_template,
    )
    assert len(render_calls) == 1
    assert len(_FakeEnv.scene_configs) == 1
    assert str(_FakeEnv.scene_configs[0].tissue_scene_path).startswith(
        str(tmp_path / "auto-surgery-brain-dejavu-")
    )


def test_capture_plan7_pre_init_hook_uses_sampled_camera_from_episode(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, Any] = {}

    class _SceneAwareCapture(_FakeCapture):
        def pre_init_hook(self, root_node: object, config: Any) -> None:
            _ = root_node
            captured["pose"] = config.scene.camera_extrinsics_scene
            captured["intrinsics"] = config.scene.camera_intrinsics

    expected_pose = Pose(
        position=Vec3(x=1.0, y=2.0, z=3.0),
        rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    expected_intrinsics = CameraIntrinsics(
        fx=950.0,
        fy=700.0,
        cx=12.0,
        cy=34.0,
        width=640,
        height=480,
    )
    scene = load_scene_config("configs/scenes/dejavu_brain.yaml").model_copy(
        update={
            "camera_extrinsics_scene": expected_pose,
            "camera_intrinsics": expected_intrinsics,
        }
    )
    _FakeEnv.scene_configs.clear()

    monkeypatch.setattr(brain_forceps, "SofaNativeRgbCapture", _SceneAwareCapture)
    _run_capture_with_fakes(
        output_dir=tmp_path,
        num_episodes=1,
        master_seed=125,
        monkeypatch=monkeypatch,
        capture_cls=_SceneAwareCapture,
        sample_episode=lambda *args, **kwargs: brain_forceps._EpisodeSpec(
            seed=3,
            scene=scene,
            motion=MotionGeneratorConfig(seed=5),
            sample_record={"camera": {"intrinsics": {"fx": 950.0}}},
        ),
    )
    assert captured["pose"] == expected_pose
    assert captured["intrinsics"] == expected_intrinsics


