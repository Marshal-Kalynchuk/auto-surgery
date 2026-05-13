from __future__ import annotations

from pathlib import Path
from typing import Any

from auto_surgery.logging.case_log import CaseCatalog
from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import SessionWriter
from auto_surgery.schemas.commands import (
    ControlMode,
    Pose,
    Quaternion,
    RobotCommand,
    Twist,
    Vec3,
)
from auto_surgery.schemas.logging import LoggedFrame, SafetyDecision
from auto_surgery.schemas.manifests import (
    CheckpointManifest,
    DataClassification,
    DatasetManifest,
    EnvConfig,
    SceneConfig,
)
from auto_surgery.schemas.results import StepResult
from auto_surgery.schemas.scene import SceneGraph, SlotRecord
from auto_surgery.schemas.sensors import (
    CameraIntrinsics,
    CameraView,
    Contact,
    JointState,
    SensorBundle,
    SafetyStatus,
    ToolState,
)
from auto_surgery.training.checkpoints import load_torch_checkpoint, save_torch_checkpoint_atomic
from auto_surgery.training.datasets import frame_count_estimate, iter_logged_frames
from auto_surgery.training.sofa_smoke import run_sofa_rollout_dataset


CANONICAL_SCENE_PATH = Path("src/auto_surgery/env/sofa_scenes/brain_dejavu_forceps_poc.scn").resolve()


def _canonical_scene_config() -> SceneConfig:
    return SceneConfig(tissue_scene_path=CANONICAL_SCENE_PATH)


def _build_tool_pose() -> Pose:
    return Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))


def _build_twist() -> Twist:
    return Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0))


def _build_camera(*, timestamp_ns: int, tool_pose: Pose, frame_rgb: bytes | None = None) -> CameraView:
    return CameraView(
        camera_id="fake",
        timestamp_ns=timestamp_ns,
        extrinsics=tool_pose,
        intrinsics=CameraIntrinsics(
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            width=640,
            height=480,
        ),
        frame_rgb=frame_rgb,
    )


class _SyntheticEpisodeEnvironment:
    """Minimal SOFA-free environment used by Stage-0 pipeline tests."""

    def __init__(
        self,
        *,
        tool_wrench: Vec3 | None = None,
        in_contact: bool = False,
    ) -> None:
        self._jaw_target = 0.0
        self._last_accepted_cycle_id = -1
        self._frame_index = 0
        self._sim_time_s = 0.0
        self._control_rate_hz = 250.0
        self._frame_rate_hz = 30.0
        self._tool_wrench = Vec3(
            x=0.0 if tool_wrench is None else float(tool_wrench.x),
            y=0.0 if tool_wrench is None else float(tool_wrench.y),
            z=0.0 if tool_wrench is None else float(tool_wrench.z),
        )
        self._in_contact = bool(in_contact)
        self._sensors = self._build_sensor_bundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            safety=SafetyStatus(
                motion_enabled=False,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=-1,
            ),
            frame_rgb=b"reset",
        )

    def reset(self, config: EnvConfig) -> StepResult:
        self._jaw_target = float(config.scene.tool.initial_jaw)
        self._last_accepted_cycle_id = -1
        self._frame_index = 0
        self._sim_time_s = 0.0
        self._control_rate_hz = config.control_rate_hz
        self._frame_rate_hz = config.frame_rate_hz
        self._sensors = self._build_sensor_bundle(
            timestamp_ns=0,
            sim_time_s=0.0,
            safety=SafetyStatus(
                motion_enabled=False,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=-1,
            ),
            frame_rgb=b"reset",
        )
        return StepResult(
            sensors=self._sensors,
            dt=0.0,
            sim_step_index=0,
            is_capture_tick=True,
        )

    def step(self, action: RobotCommand) -> StepResult:
        blocked, reason = self._command_block_reason(action)
        if not blocked:
            self._last_accepted_cycle_id = action.cycle_id
        self._frame_index += 1
        dt = 1.0 / self._control_rate_hz
        self._sim_time_s += dt
        if not blocked and action.tool_jaw_target is not None:
            self._jaw_target = action.tool_jaw_target
        self._sensors = self._build_sensor_bundle(
            timestamp_ns=action.timestamp_ns,
            sim_time_s=self._sim_time_s,
            safety=SafetyStatus(
                motion_enabled=action.enable and not blocked,
                command_blocked=blocked,
                block_reason=reason,
                cycle_id_echo=action.cycle_id,
            ),
        )
        return StepResult(
            sensors=self._sensors,
            dt=dt,
            sim_step_index=self._frame_index,
            is_capture_tick=(self._frame_index % max(1, round(self._control_rate_hz / self._frame_rate_hz)) == 0),
        )

    def gate_command(self, cmd: RobotCommand) -> SafetyDecision:
        blocked, reason = self._command_block_reason(cmd)
        return SafetyDecision(
            ok=not blocked,
            gate_action="pass" if not blocked else "veto",
            reason_codes=[reason] if reason else [],
        )

    def get_joint_state(self) -> JointState:
        return JointState(positions={}, velocities={})

    def get_sensors(self) -> SensorBundle:
        return self._sensors.model_copy(deep=True)

    def get_contacts(self) -> list[Contact]:
        return []

    def _command_block_reason(self, command: RobotCommand) -> tuple[bool, str | None]:
        if command.cycle_id <= self._last_accepted_cycle_id:
            return True, "stale_cycle_id"
        if not command.enable:
            return True, "disabled"
        return False, None

    def _build_sensor_bundle(
        self,
        *,
        timestamp_ns: int,
        sim_time_s: float,
        safety: SafetyStatus,
        frame_rgb: bytes | None = None,
    ) -> SensorBundle:
        pose = _build_tool_pose()
        tool_state = ToolState(
            pose=pose,
            twist=_build_twist(),
            jaw=self._jaw_target,
            wrench=self._tool_wrench,
            in_contact=self._in_contact,
        )
        camera = _build_camera(timestamp_ns=timestamp_ns, tool_pose=pose, frame_rgb=frame_rgb)
        return SensorBundle(
            timestamp_ns=timestamp_ns,
            sim_time_s=sim_time_s,
            tool=tool_state,
            cameras=[camera],
            safety=safety,
        )


class _SyntheticRuntimeBackend:
    """Protocol-compatible SOFA backend used by dataset harness tests."""

    def __init__(
        self,
        scene_path: str | None = None,
        *,
        tool_wrench: Vec3 | None = None,
        in_contact: bool = False,
    ) -> None:
        del scene_path
        self._frame_index = 0
        self._jaw = 0.0
        self._control_rate_hz = 250.0
        self._frame_decimation = 1
        self._tool_wrench = Vec3(
            x=0.0 if tool_wrench is None else float(tool_wrench.x),
            y=0.0 if tool_wrench is None else float(tool_wrench.y),
            z=0.0 if tool_wrench is None else float(tool_wrench.z),
        )
        self._in_contact = bool(in_contact)

    def reset(self, config: EnvConfig) -> StepResult:
        self._frame_index = 0
        self._control_rate_hz = config.control_rate_hz
        self._frame_decimation = max(1, round(self._control_rate_hz / config.frame_rate_hz))
        self._jaw = float(config.scene.tool.initial_jaw)
        return StepResult(
            sensors=self._build_sensors(
                sim_step_index=0,
                reset=True,
                timestamp_ns=0,
                cycle_id=-1,
            ),
            dt=0.0,
            sim_step_index=0,
            is_capture_tick=True,
        )

    def step(self, action: RobotCommand) -> StepResult:
        self._frame_index += 1
        dt = 1.0 / self._control_rate_hz
        self._jaw = float(action.tool_jaw_target) if action.tool_jaw_target is not None else self._jaw
        return StepResult(
            sensors=self._build_sensors(
                sim_step_index=self._frame_index,
                timestamp_ns=action.timestamp_ns,
                cycle_id=action.cycle_id,
            ),
            dt=dt,
            sim_step_index=self._frame_index,
            is_capture_tick=(self._frame_index % self._frame_decimation == 0),
        )

    def get_joint_state(self) -> JointState:
        return JointState(positions={}, velocities={})

    def get_sensors(self) -> SensorBundle:
        return self._build_sensors(
            sim_step_index=self._frame_index,
            timestamp_ns=self._frame_index,
            cycle_id=self._frame_index,
        )

    def get_scene(self) -> SceneGraph:
        return SceneGraph(
            frame_index=self._frame_index,
            slots=[SlotRecord(slot_id="tool_0", pose={"x": 0.0, "y": 0.0, "z": 0.0})],
            events=[{"frame": self._frame_index}],
        )

    def get_contacts(self) -> list[Contact]:
        return []

    def set_control_rate_hz(self, control_rate_hz: float) -> None:
        self._control_rate_hz = control_rate_hz

    def set_initial_jaw(self, jaw: float) -> None:
        self._jaw = jaw

    def set_tool_jaw_target(self, jaw: float) -> None:
        self._jaw = jaw

    def _build_sensors(
        self,
        *,
        sim_step_index: int,
        timestamp_ns: int,
        cycle_id: int,
        reset: bool = False,
    ) -> SensorBundle:
        pose = _build_tool_pose()
        return SensorBundle(
            timestamp_ns=timestamp_ns,
            sim_time_s=sim_step_index / self._control_rate_hz,
            tool=ToolState(
                pose=pose,
                twist=_build_twist(),
                jaw=self._jaw,
                wrench=self._tool_wrench,
                in_contact=self._in_contact,
            ),
            cameras=[
                _build_camera(
                    timestamp_ns=timestamp_ns,
                    tool_pose=pose,
                    frame_rgb=(b"reset" if reset else f"frame_{sim_step_index}".encode("utf-8")),
                )
            ],
            safety=SafetyStatus(
                motion_enabled=True,
                command_blocked=False,
                block_reason=None,
                cycle_id_echo=cycle_id,
            ),
        )


def test_sim_rollout_to_dataset_loader_and_checkpoint(tmp_path: Path) -> None:
    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    sim = _SyntheticEpisodeEnvironment()
    sim.reset(
        EnvConfig(seed=3, scene=_canonical_scene_config())
    )
    writer = SessionWriter(
        root_uri,
        "case_x",
        "sess_y",
        capture_rig_id="rig",
        clock_source="monotonic",
        software_git_sha="abc123",
        data_classification=DataClassification.SIMULATION,
        sensor_list=["kinematics"],
        segment_max_frames=4,
    )
    for i in range(10):
        cmd = RobotCommand(
            timestamp_ns=1_000 + i,
            cycle_id=i,
            control_mode=ControlMode.CARTESIAN_TWIST,
            cartesian_twist=Twist(
                linear=Vec3(x=float(i) * 0.01, y=0.0, z=0.0),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            enable=True,
            source="test",
        )
        gate = sim.gate_command(cmd)
        step = sim.step(cmd)
        lf = LoggedFrame(
            frame_index=i,
            timestamp_ns=cmd.timestamp_ns,
            sensor_payload=step.sensors,
            scene_snapshot=None,
            commanded_action=cmd,
            executed_action=cmd,
            safety_decision=gate,
        )
        writer.write_frame(lf)
    manifest = writer.finalize()
    catalog = CaseCatalog(root_uri)
    catalog.append(manifest, manifest_relative_path=session_manifest_path("case_x", "sess_y"))

    manifest_uri = f"{root_uri}{session_manifest_path('case_x', 'sess_y')}"
    ds = DatasetManifest(
        dataset_id="ds1",
        session_manifest_paths=[manifest_uri],
        data_classification=DataClassification.SIMULATION,
    )
    assert frame_count_estimate(ds) == 10
    frames = list(iter_logged_frames(ds))
    assert len(frames) == 10
    assert frames[0].commanded_action is not None

    ckpt_uri = (tmp_path / "checkpoint.pt").as_uri()
    save_torch_checkpoint_atomic(
        ckpt_uri,
        {"metrics": {"loss": 0.01}},
        CheckpointManifest(
            checkpoint_id="ckpt1",
            dataset_manifest_path=manifest_uri,
            training_config_path="n/a",
            git_sha="abc123",
        ),
    )
    loaded = load_torch_checkpoint(ckpt_uri)
    assert loaded["metrics"]["loss"] == 0.01


def test_synthetic_episode_environment_emits_contact_and_wrench() -> None:
    env = _SyntheticEpisodeEnvironment(
        tool_wrench=Vec3(x=0.5, y=-0.25, z=0.1),
        in_contact=True,
    )
    env.reset(EnvConfig(seed=9, scene=_canonical_scene_config()))
    command = RobotCommand(
        timestamp_ns=123,
        cycle_id=0,
        control_mode=ControlMode.CARTESIAN_TWIST,
        cartesian_twist=Twist(
            linear=Vec3(x=0.0, y=0.0, z=0.0),
            angular=Vec3(x=0.0, y=0.0, z=0.0),
        ),
        enable=True,
        source="test",
    )
    step = env.step(command)
    assert step.sensors.tool.wrench == Vec3(x=0.5, y=-0.25, z=0.1)
    assert step.sensors.tool.in_contact is True


def test_synthetic_runtime_backend_emits_contact_and_wrench() -> None:
    backend = _SyntheticRuntimeBackend(
        "test://scene",
        tool_wrench=Vec3(x=0.4, y=0.25, z=-0.15),
        in_contact=True,
    )
    first = backend.reset(EnvConfig(seed=11, scene=_canonical_scene_config()))
    assert first.sensors.tool.wrench == Vec3(x=0.4, y=0.25, z=-0.15)
    assert first.sensors.tool.in_contact is True

    step = backend.step(
        RobotCommand(
            timestamp_ns=222,
            cycle_id=1,
            control_mode=ControlMode.CARTESIAN_TWIST,
            cartesian_twist=Twist(
                linear=Vec3(x=0.0, y=0.0, z=0.0),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            ),
            enable=True,
            source="test",
        )
    )
    assert step.sensors.tool.wrench == Vec3(x=0.4, y=0.25, z=-0.15)
    assert step.sensors.tool.in_contact is True


def test_sofa_smoke_rollout_harness(tmp_path: Path) -> None:
    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    ds = run_sofa_rollout_dataset(
        storage_root_uri=root_uri,
        case_id="sofa_case",
        session_id="sofa_session",
        steps=12,
        segment_max_frames=4,
        sofa_scene_path="test://scene.json",
        sofa_backend_factory=lambda scene_path, _extra: _SyntheticRuntimeBackend(scene_path),
        scene_config=_canonical_scene_config(),
    )
    assert frame_count_estimate(ds) == 12
    frames = list(iter_logged_frames(ds))
    assert len(frames) == 12
    assert frames[0].commanded_action is not None


def test_sofa_rollout_dataset_prefers_factory_when_scene_path_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: dict[str, Any] = {}

    class _FakeSofaEnvironment(_SyntheticEpisodeEnvironment):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            recorded["kwargs"] = kwargs

        def get_scene(self):
            return None

    monkeypatch.setattr("auto_surgery.training.sofa_smoke.SofaEnvironment", _FakeSofaEnvironment)

    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    run_sofa_rollout_dataset(
        storage_root_uri=root_uri,
        case_id="sofa_case",
        session_id="sofa_session",
        steps=2,
        segment_max_frames=4,
        sofa_scene_path=None,
        sofa_scene_factory=lambda scene_handle, _config: None,
        scene_config=SceneConfig(tissue_scene_path=CANONICAL_SCENE_PATH),
    )

    assert "sofa_scene_factory" in recorded["kwargs"]
    assert "sofa_scene_path" not in recorded["kwargs"]


def test_sofa_rollout_dataset_prefers_scene_config_tissue_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: dict[str, Any] = {}

    class _FakeSofaEnvironment(_SyntheticEpisodeEnvironment):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            recorded["kwargs"] = kwargs

        def get_scene(self):
            return None

    monkeypatch.setattr("auto_surgery.training.sofa_smoke.SofaEnvironment", _FakeSofaEnvironment)

    scene_config = SceneConfig(tissue_scene_path=CANONICAL_SCENE_PATH)
    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    run_sofa_rollout_dataset(
        storage_root_uri=root_uri,
        case_id="sofa_case",
        session_id="sofa_session",
        steps=2,
        segment_max_frames=4,
        scene_config=scene_config,
    )

    assert recorded["kwargs"]["sofa_scene_path"] == str(CANONICAL_SCENE_PATH)


def test_sofa_rollout_dataset_prefers_explicit_scene_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: dict[str, Any] = {}

    class _FakeSofaEnvironment(_SyntheticEpisodeEnvironment):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            recorded["kwargs"] = kwargs

        def get_scene(self):
            return None

    monkeypatch.setattr("auto_surgery.training.sofa_smoke.SofaEnvironment", _FakeSofaEnvironment)

    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    run_sofa_rollout_dataset(
        storage_root_uri=root_uri,
        case_id="sofa_case",
        session_id="sofa_session",
        steps=2,
        segment_max_frames=4,
        sofa_scene_path="explicit://scene.scn",
        sofa_scene_factory=lambda scene_handle, _config: None,
        scene_config=SceneConfig(tissue_scene_path=CANONICAL_SCENE_PATH),
    )

    assert "sofa_scene_path" in recorded["kwargs"]
    assert recorded["kwargs"]["sofa_scene_path"] == "explicit://scene.scn"
    assert "sofa_scene_factory" not in recorded["kwargs"]
