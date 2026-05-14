from __future__ import annotations

from pathlib import Path

import pytest

from auto_surgery.logging.case_log import CaseCatalog
from auto_surgery.logging.storage import session_manifest_path
from auto_surgery.logging.writer import SessionWriter
from auto_surgery.config import load_scene_config
from auto_surgery.schemas.commands import ControlMode, Pose, Quaternion, RobotCommand, Twist, Vec3
from auto_surgery.schemas.logging import LoggedFrame, SafetyDecision
from auto_surgery.schemas.manifests import (
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
    JointState,
    Contact,
    SensorBundle,
    SafetyStatus,
    ToolState,
)
from auto_surgery.training.datasets import frame_count_estimate, iter_logged_frames
from auto_surgery.training.extract_pseudo_actions import extract_pseudo_actions
from auto_surgery.training.idm_train import train_idm
from auto_surgery.training.sofa_smoke import run_sofa_smoke_pipeline


def _test_scene_config() -> SceneConfig:
    return load_scene_config("configs/scenes/dejavu_brain.yaml")


class _SyntheticEpisodeEnvironment:
    """SOFA-free environment used by Stage-0 IDM tests."""

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
        self._scene = SceneGraph(slots=[])
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
        self._scene = SceneGraph(
            frame_index=0,
            slots=[SlotRecord(slot_id="tool_0", pose={"x": 0.0, "y": 0.0, "z": 0.0})],
            events=[{"seed": config.seed}],
        )
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
        tool_pose = Pose(
            position=Vec3(x=0.0, y=0.0, z=0.0),
            rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        )
        return SensorBundle(
            timestamp_ns=timestamp_ns,
            sim_time_s=sim_time_s,
            tool=ToolState(
                pose=tool_pose,
                twist=Twist(
                    linear=Vec3(x=0.0, y=0.0, z=0.0),
                    angular=Vec3(x=0.0, y=0.0, z=0.0),
                ),
                jaw=self._jaw_target,
                wrench=self._tool_wrench,
                in_contact=self._in_contact,
            ),
            cameras=[
                CameraView(
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
            ],
            safety=safety,
        )

    def get_contacts(self) -> list[Contact]:
        return []

class _SyntheticRuntimeBackend:
    """Protocol-compatible SOFA backend used by smoke pipeline tests."""

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
        self._tool_pose = Pose(position=Vec3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
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
        return SensorBundle(
            timestamp_ns=timestamp_ns,
            sim_time_s=sim_step_index / self._control_rate_hz,
            tool=ToolState(
                pose=self._tool_pose,
                twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
                jaw=self._jaw,
                wrench=self._tool_wrench,
                in_contact=self._in_contact,
            ),
            cameras=[
                CameraView(
                    camera_id="fake",
                    timestamp_ns=timestamp_ns,
                    extrinsics=self._tool_pose,
                    intrinsics=CameraIntrinsics(
                        fx=1.0,
                        fy=1.0,
                        cx=0.0,
                        cy=0.0,
                        width=640,
                        height=480,
                    ),
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

def _ensure_torch_available() -> None:
    pytest.importorskip("torch", reason="torch extra is required for Stage-0 IDM training")


def test_idm_stage0_stub_pipeline(tmp_path: Path) -> None:
    _ensure_torch_available()

    root_uri = tmp_path.as_uri().rstrip("/") + "/"
    derived_root_uri = (tmp_path / "derived").as_uri().rstrip("/") + "/"

    sim = _SyntheticEpisodeEnvironment()
    sim.reset(
        EnvConfig(
            seed=3,
            scene=_test_scene_config(),
        )
    )

    writer = SessionWriter(
        root_uri,
        "case_id_stage0",
        "session_id_stage0",
        capture_rig_id="rig",
        clock_source="monotonic",
        software_git_sha="abc123",
        data_classification=DataClassification.SIMULATION,
        sensor_list=["kinematics"],
        segment_max_frames=8,
    )

    n_frames = 64
    for i in range(n_frames):
        cmd = RobotCommand(
            timestamp_ns=1_000_000 + i,
            cycle_id=i,
            control_mode=ControlMode.CARTESIAN_TWIST,
            cartesian_twist=Twist(
                linear=Vec3(x=0.01 * float(i), y=0.0, z=0.0),
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
    CaseCatalog(root_uri).append(
        manifest,
        manifest_relative_path=session_manifest_path("case_id_stage0", "session_id_stage0"),
    )
    manifest_uri = f"{root_uri}{session_manifest_path('case_id_stage0', 'session_id_stage0')}"

    ds = DatasetManifest(
        dataset_id="ds_stage0",
        session_manifest_paths=[manifest_uri],
        data_classification=DataClassification.SIMULATION,
    )
    assert frame_count_estimate(ds) == n_frames

    ckpt_uri = (tmp_path / "idm_stage0.pt").as_uri()
    train_idm(
        ds,
        out_ckpt_uri=ckpt_uri,
        dataset_manifest_path=manifest_uri,
        steps=300,
        lr=5e-3,
        hidden_dim=32,
        device=None,
        git_sha="stage0-test",
    )

    out_ds = extract_pseudo_actions(
        ds,
        idm_ckpt_uri=ckpt_uri,
        out_root_uri=derived_root_uri,
        out_case_id="case_id_stage0_derived",
        out_session_id="session_id_stage0_derived",
        capture_rig_id="rig",
        clock_source="monotonic",
        software_git_sha="stage0-test",
    )

    orig_frames = list(iter_logged_frames(ds))
    pred_frames = list(iter_logged_frames(out_ds))
    assert len(pred_frames) == len(orig_frames) == n_frames

    mse_sum = 0.0
    n = 0
    for orig, pred in zip(orig_frames, pred_frames, strict=True):
        assert orig.frame_index == pred.frame_index
        assert orig.executed_action is not None
        assert pred.executed_action is not None
        assert orig.executed_action.cartesian_twist is not None
        assert pred.executed_action.cartesian_twist is not None
        diff = (
            orig.executed_action.cartesian_twist.linear.x
            - pred.executed_action.cartesian_twist.linear.x
        )
        mse_sum += diff * diff
        n += 1

    mse = mse_sum / max(n, 1)
    assert mse < 1e-4


def test_synthetic_episode_environment_emits_tool_contact_fields() -> None:
    sim = _SyntheticEpisodeEnvironment(tool_wrench=Vec3(x=0.2, y=0.0, z=-0.5), in_contact=True)
    sim.reset(EnvConfig(seed=5, scene=_test_scene_config()))
    step = sim.step(
        RobotCommand(
            timestamp_ns=123,
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
    assert step.sensors.tool.wrench == Vec3(x=0.2, y=0.0, z=-0.5)
    assert step.sensors.tool.in_contact is True


def test_synthetic_runtime_backend_emits_tool_contact_fields() -> None:
    backend = _SyntheticRuntimeBackend(
        "test://scene",
        tool_wrench=Vec3(x=-0.3, y=0.1, z=0.0),
        in_contact=False,
    )
    reset = backend.reset(EnvConfig(seed=8, scene=_test_scene_config()))
    assert reset.sensors.tool.wrench == Vec3(x=-0.3, y=0.1, z=0.0)
    assert reset.sensors.tool.in_contact is False

    step = backend.step(
        RobotCommand(
            timestamp_ns=300,
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
    assert step.sensors.tool.wrench == Vec3(x=-0.3, y=0.1, z=0.0)
    assert step.sensors.tool.in_contact is False


def test_sofa_smoke_pipeline(tmp_path: Path) -> None:
    _ensure_torch_available()

    root_uri = (tmp_path / "sofa_smoke").as_uri().rstrip("/") + "/"
    source, stats, derived = run_sofa_smoke_pipeline(
        out_root_uri=root_uri,
        case_id="sofa_case_stage0",
        session_id="sofa_session_stage0",
        derived_case_id="sofa_case_stage0_derived",
        derived_session_id="sofa_session_stage0_derived",
        sofa_scene_path="test://sofa_case_stage0",
        sofa_backend_factory=lambda scene_path, _extra: _SyntheticRuntimeBackend(scene_path),
        scene_config=_test_scene_config(),
        steps=24,
        train_steps=64,
        train_lr=5e-3,
        hidden_dim=16,
    )

    assert "train_mse" in stats
    assert frame_count_estimate(source) == 24
    src_frames = list(iter_logged_frames(source))
    pred_frames = list(iter_logged_frames(derived))
    assert len(src_frames) == len(pred_frames) == 24
