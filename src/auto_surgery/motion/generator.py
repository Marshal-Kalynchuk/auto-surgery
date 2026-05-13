"""Public motion generator runtime for surgical trajectory playback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

from auto_surgery.motion.frames import TwistSceneTip
from auto_surgery.motion.primitives import (
    Reach,
    Hold,
    ContactReach,
    Grip,
    Drag,
    Brush,
    Primitive,
    _CONTACT_TOLERANCE_M,
    _SMALL_STANDOFF_M,
    _camera_basis,
    _camera_tool_to_scene,
    _evaluate as _evaluate_primitive,
    _jaw_target,
    _pose_rotation_delta_vec,
    _time_to_fraction,
    _vec_to_array,
    _vector_to_vec3,
)
from auto_surgery.motion.profile import min_jerk_velocity_scalar
from auto_surgery.motion.sequencer import _Sequencer
from auto_surgery.schemas.commands import (
    ControlFrame,
    ControlMode,
    Pose,
    Quaternion,
    RobotCommand,
    SafetyMetadata,
    Twist,
    Vec3,
)
from auto_surgery.schemas.motion import MotionGeneratorConfig
from auto_surgery.schemas.results import StepResult


@dataclass(frozen=True)
class RealisedPrimitive:
    primitive: Primitive
    started_at_tick: int
    ended_at_tick: int
    early_terminated: bool


def _as_vec(value: Vec3 | list[float] | tuple[float, float, float]) -> list[float]:
    if isinstance(value, Vec3):
        return [float(value.x), float(value.y), float(value.z)]
    return [float(value[0]), float(value[1]), float(value[2])]


def _norm(vector: list[float] | tuple[float, float, float]) -> float:
    return math.sqrt(float(vector[0]) ** 2 + float(vector[1]) ** 2 + float(vector[2]) ** 2)


def _normalize(vector: list[float] | tuple[float, float, float]) -> list[float]:
    magnitude = _norm(vector)
    if magnitude <= 0.0:
        return [0.0, 0.0, 1.0]
    return [float(vector[0]) / magnitude, float(vector[1]) / magnitude, float(vector[2]) / magnitude]


def _cross(a: list[float] | tuple[float, float, float], b: list[float] | tuple[float, float, float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _matmul(matrix: list[list[float]], vector: list[float] | tuple[float, float, float]) -> list[float]:
    v = _as_vec(vector)
    return [
        float(matrix[0][0]) * v[0] + float(matrix[0][1]) * v[1] + float(matrix[0][2]) * v[2],
        float(matrix[1][0]) * v[0] + float(matrix[1][1]) * v[1] + float(matrix[1][2]) * v[2],
        float(matrix[2][0]) * v[0] + float(matrix[2][1]) * v[1] + float(matrix[2][2]) * v[2],
    ]


def _mat_transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]


def _quat_to_tuple(quaternion: object) -> tuple[float, float, float, float]:
    return (
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    )


def _quat_to_matrix(quaternion: object) -> list[list[float]]:
    x, y, z, w = _quat_to_tuple(quaternion)
    magnitude = math.sqrt(max(1.0e-12, x * x + y * y + z * z + w * w))
    x /= magnitude
    y /= magnitude
    z /= magnitude
    w /= magnitude
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _slerp_direction(start: list[float], end: list[float], blend: float) -> list[float]:
    start_n = _normalize(start)
    end_n = _normalize(end)
    blend = max(0.0, min(1.0, float(blend)))
    if blend <= 0.0:
        return start_n
    if blend >= 1.0:
        return end_n

    dot = float(start_n[0] * end_n[0] + start_n[1] * end_n[1] + start_n[2] * end_n[2])
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.999999:
        return _normalize([start_n[0] * (1.0 - blend) + end_n[0] * blend, start_n[1] * (1.0 - blend) + end_n[1] * blend, start_n[2] * (1.0 - blend) + end_n[2] * blend])
    if dot < -0.999999:
        return [0.0, 0.0, 1.0]

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if abs(sin_theta) <= 1.0e-12:
        return end_n
    weight_start = math.sin((1.0 - blend) * theta) / sin_theta
    weight_end = math.sin(blend * theta) / sin_theta
    return _normalize([
        weight_start * start_n[0] + weight_end * end_n[0],
        weight_start * start_n[1] + weight_end * end_n[1],
        weight_start * start_n[2] + weight_end * end_n[2],
    ])


def _axis_angle_between(start: list[float], end: list[float]) -> list[float]:
    start_n = _normalize(start)
    end_n = _normalize(end)
    dot = float(start_n[0] * end_n[0] + start_n[1] * end_n[1] + start_n[2] * end_n[2])
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.999999:
        return [0.0, 0.0, 0.0]

    cross = _cross(start_n, end_n)
    cross_norm = _norm(cross)
    if cross_norm <= 1.0e-12:
        if abs(start_n[0]) < 0.999999:
            axis = _normalize(_cross(start_n, [1.0, 0.0, 0.0]))
        else:
            axis = _normalize(_cross(start_n, [0.0, 1.0, 0.0]))
        axis = _normalize(axis)
        return [axis[0] * math.pi, axis[1] * math.pi, axis[2] * math.pi]

    angle = math.atan2(cross_norm, dot)
    return [cross[0] / cross_norm * angle, cross[1] / cross_norm * angle, cross[2] / cross_norm * angle]


def _twist_camera_to_scene(last_step: StepResult, twist_camera: Twist) -> Twist:
    camera_matrix, camera_position = _camera_basis(last_step)
    matrix = camera_matrix.tolist()
    angular_scene = _matmul(matrix, _as_vec(twist_camera.angular))
    linear_camera = _as_vec(twist_camera.linear)
    correction = _cross(_as_vec(camera_position), angular_scene)
    linear_scene = [
        linear_camera[0] + correction[0],
        linear_camera[1] + correction[1],
        linear_camera[2] + correction[2],
    ]
    linear_scene = _matmul(matrix, linear_scene)
    return Twist(
        linear=Vec3(x=float(linear_scene[0]), y=float(linear_scene[1]), z=float(linear_scene[2])),
        angular=Vec3(x=float(angular_scene[0]), y=float(angular_scene[1]), z=float(angular_scene[2])),
    )


def _twist_scene_to_camera(last_step: StepResult, twist_scene: Twist) -> Twist:
    camera_matrix, camera_position = _camera_basis(last_step)
    matrix = camera_matrix.tolist()
    rotation_inverse = _mat_transpose(matrix)
    angular = _as_vec(twist_scene.angular)
    linear = _as_vec(twist_scene.linear)
    correction = _cross(_as_vec(camera_position), angular)
    adjusted_linear = [linear[0] - correction[0], linear[1] - correction[1], linear[2] - correction[2]]
    angular_camera = _matmul(rotation_inverse, angular)
    linear_camera = _matmul(rotation_inverse, adjusted_linear)
    return Twist(
        linear=Vec3(x=float(linear_camera[0]), y=float(linear_camera[1]), z=float(linear_camera[2])),
        angular=Vec3(x=float(angular_camera[0]), y=float(angular_camera[1]), z=float(angular_camera[2])),
    )


def _tool_workspace_envelope(scene_config: object) -> object | None:
    tool = getattr(scene_config, "tool", None)
    if tool is None:
        return None
    return getattr(tool, "workspace_envelope", None)


def _signed_distance_to_envelope(*, envelope: object, point: Vec3) -> float | None:
    if envelope is None:
        return None
    signed_distance_fn = getattr(envelope, "signed_distance", None)
    if callable(signed_distance_fn):
        return float(signed_distance_fn(point))
    signed_distance_to_envelope_fn = getattr(envelope, "signed_distance_to_envelope", None)
    if callable(signed_distance_to_envelope_fn):
        return float(signed_distance_to_envelope_fn(point))
    return None


def _outward_normal(*, envelope: object, point: Vec3) -> list[float]:
    center = getattr(envelope, "center_scene", None)
    if isinstance(center, Vec3):
        delta = [
            float(point.x) - float(center.x),
            float(point.y) - float(center.y),
            float(point.z) - float(center.z),
        ]
        return _normalize(delta)
    return _normalize(_as_vec(point))


def _apply_linear_bias(
    *,
    raw_linear: list[float],
    signed_distance: float,
    point: Vec3,
    envelope: object,
    bias_gain_max: float,
    bias_ramp_distance_m: float,
    outer_margin_m: float,
) -> tuple[list[float], bool]:
    if bias_gain_max <= 0.0 or outer_margin_m <= 0.0 or bias_ramp_distance_m <= 0.0:
        return raw_linear, False

    beta = (float(outer_margin_m) - float(signed_distance)) / float(bias_ramp_distance_m)
    beta = max(0.0, min(float(beta), 1.0)) * float(bias_gain_max)
    if beta <= 0.0:
        return raw_linear, False

    raw_speed = _norm(raw_linear)
    outward = _outward_normal(envelope=envelope, point=point)
    if raw_speed <= 1.0e-12:
        gain = float(bias_gain_max)
        v_bias = [
            -gain * float(signed_distance) * outward[0],
            -gain * float(signed_distance) * outward[1],
            -gain * float(signed_distance) * outward[2],
        ]
    else:
        sign = 1.0 if float(signed_distance) > 0.0 else (-1.0 if float(signed_distance) < 0.0 else 0.0)
        v_bias = [
            -sign * raw_speed * outward[0],
            -sign * raw_speed * outward[1],
            -sign * raw_speed * outward[2],
        ]

    return [
        (1.0 - beta) * raw_linear[0] + beta * v_bias[0],
        (1.0 - beta) * raw_linear[1] + beta * v_bias[1],
        (1.0 - beta) * raw_linear[2] + beta * v_bias[2],
    ], True


def _evaluate_reach(active: object, last_step: StepResult) -> TwistSceneTip:
    """Evaluate active primitive as a single SE(3) min-jerk motion."""

    primitive = getattr(active, "primitive", None)
    duration_s = float(getattr(active, "duration_s", 0.0))
    elapsed_s = float(getattr(active, "elapsed_s", 0.0))
    tau = _time_to_fraction(elapsed_s, duration_s)

    if isinstance(primitive, Dwell):
        return TwistSceneTip(
            Twist(
                linear=Vec3(x=0.0, y=0.0, z=0.0),
                angular=Vec3(x=0.0, y=0.0, z=0.0),
            )
        )

    if isinstance(primitive, Approach):
        camera_matrix, camera_position = _camera_basis(last_step)
        start_pose_scene = _camera_tool_to_scene(
            tool_pose=getattr(active, "started_at_pose_scene"),
            camera_matrix=camera_matrix,
            camera_position=camera_position,
        )
        if duration_s <= 0.0:
            return TwistSceneTip(
                Twist(
                    linear=Vec3(x=0.0, y=0.0, z=0.0),
                    angular=Vec3(x=0.0, y=0.0, z=0.0),
                )
            )

        speed = min_jerk_velocity_scalar(tau=tau, duration_s=duration_s)
        linear_delta = _vec_to_array(primitive.target_pose_scene.position) - _vec_to_array(start_pose_scene.position)
        angular_delta = _pose_rotation_delta_vec(start_pose_scene, primitive.target_pose_scene)
        return TwistSceneTip(
            Twist(
                linear=_vector_to_vec3(linear_delta * speed),
                angular=_vector_to_vec3(angular_delta * speed),
            )
        )

    legacy_output = _evaluate_primitive(active, last_step)
    return TwistSceneTip(_twist_camera_to_scene(last_step, legacy_output.twist_camera))


class SurgicalMotionGenerator:
    def __init__(self, motion_config: MotionGeneratorConfig, scene_config: object) -> None:
        self._motion_config = motion_config
        self._scene_config = scene_config
        from auto_surgery.motion.fsm import _Fsm  # local import to avoid circular import at module load

        self._sequencer = _Sequencer(motion_config, scene_config)
        self._fsm = _Fsm(self._sequencer)
        self._cycle_id = -1
        self._reset_called = False
        self._last_jaw_commanded = self._initial_jaw()
        self._finalized = False
        self._tool_orientation_bias = getattr(
            getattr(self._scene_config, "tool", None),
            "orientation_bias",
            None,
        )
        self._scene_center: list[float] | None = None

    def _scene_center_for_geometry(self, *, fallback: list[float]) -> list[float]:
        if self._scene_center is None:
            scene_geometry = getattr(self._sequencer, "_scene_geometry", None)
            if scene_geometry is not None:
                try:
                    geometry_min, geometry_max = scene_geometry.bounds()
                    self._scene_center = [
                        0.5 * (float(geometry_min.x) + float(geometry_max.x)),
                        0.5 * (float(geometry_min.y) + float(geometry_max.y)),
                        0.5 * (float(geometry_min.z) + float(geometry_max.z)),
                    ]
                except Exception:
                    self._scene_center = None
            if self._scene_center is None:
                if fallback:
                    self._scene_center = [float(fallback[0]), float(fallback[1]), float(fallback[2])]
                else:
                    self._scene_center = [0.0, 0.0, 0.0]
        return list(self._scene_center)

    def _compute_orientation_bias(
        self,
        *,
        raw_angular: list[float],
        tip_pose_scene,
        primitive: Primitive,
        in_contact: bool,
    ) -> tuple[list[float], bool]:
        if not bool(getattr(self._motion_config, "motion_shaping_enabled", False)):
            return raw_angular, False

        motion_shaping = getattr(self._motion_config, "motion_shaping", None)
        if motion_shaping is None:
            return raw_angular, False

        orientation_bias_gain = float(
            getattr(motion_shaping, "orientation_bias_gain", 0.0)
            if motion_shaping is not None
            else 0.0
        )
        if orientation_bias_gain <= 0.0 or in_contact:
            return raw_angular, False

        primitive_bias_enabled = bool(getattr(primitive, "allow_orientation_bias", True))
        if not primitive_bias_enabled:
            return raw_angular, False

        tool_bias = self._tool_orientation_bias
        beta_ori = float(getattr(tool_bias, "gain", 0.0)) if tool_bias is not None else 0.0
        if beta_ori <= 0.0:
            return raw_angular, False

        surface_normal_blend = float(getattr(tool_bias, "surface_normal_blend", 0.0)) if tool_bias is not None else 0.0
        forward_axis_local = _as_vec(getattr(tool_bias, "forward_axis_local", Vec3(x=0.0, y=0.0, z=1.0)))

        scene_geometry = getattr(self._sequencer, "_scene_geometry", None)
        if scene_geometry is None:
            return raw_angular, False

        try:
            cp = scene_geometry.closest_surface_point(tip_pose_scene.position)
        except Exception:
            return raw_angular, False

        tip_position = _as_vec(tip_pose_scene.position)
        scene_center = self._scene_center_for_geometry(fallback=tip_position)
        d_surface = _normalize([
            float(cp.position.x) - tip_position[0],
            float(cp.position.y) - tip_position[1],
            float(cp.position.z) - tip_position[2],
        ])
        d_center = _normalize([
            scene_center[0] - tip_position[0],
            scene_center[1] - tip_position[1],
            scene_center[2] - tip_position[2],
        ])

        d_desired = _normalize(
            _slerp_direction(
                start=d_center,
                end=d_surface,
                blend=surface_normal_blend,
            )
        )
        forward_scene = _matmul(_quat_to_matrix(tip_pose_scene.rotation), forward_axis_local)
        e_rot = _axis_angle_between(forward_scene, d_desired)
        if _norm(e_rot) <= float(
            getattr(
                self._motion_config.motion_shaping,
                "orientation_deadband_rad",
                float(getattr(tool_bias, "deadband_rad", 0.0)) if tool_bias is not None else 0.0,
            )
        ):
            return raw_angular, False

        omega_bias = [float(e_rot[0]) * orientation_bias_gain, float(e_rot[1]) * orientation_bias_gain, float(e_rot[2]) * orientation_bias_gain]
        omega_final = [
            (1.0 - beta_ori) * raw_angular[0] + beta_ori * omega_bias[0],
            (1.0 - beta_ori) * raw_angular[1] + beta_ori * omega_bias[1],
            (1.0 - beta_ori) * raw_angular[2] + beta_ori * omega_bias[2],
        ]
        return omega_final, True

    def _evaluate_contact_reach(
        self,
        *,
        active: object,
        last_step: StepResult,
        tip_now: Vec3,
        tip_pose_scene: Pose | None = None,
    ) -> TwistSceneTip:
        primitive = getattr(active, "primitive", None)
        if not isinstance(primitive, ContactReach):
            return _evaluate_reach(active=active, last_step=last_step)

        scene_geometry = getattr(self._sequencer, "_scene_geometry", None)
        if scene_geometry is None:
            return _evaluate_reach(active=active, last_step=last_step)

        if tip_pose_scene is None:
            tip_pose_scene = Pose(position=tip_now, rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))

        try:
            cp = scene_geometry.closest_surface_point(tip_now)
        except Exception:
            return _evaluate_reach(active=active, last_step=last_step)

        cp_normal = _as_vec(getattr(cp, "normal", Vec3(x=0.0, y=0.0, z=1.0)))
        cp_position = _as_vec(getattr(cp, "position", tip_now))
        tip_position = _as_vec(tip_now)
        target_position = [
            cp_position[0] + cp_normal[0] * _SMALL_STANDOFF_M,
            cp_position[1] + cp_normal[1] * _SMALL_STANDOFF_M,
            cp_position[2] + cp_normal[2] * _SMALL_STANDOFF_M,
        ]
        remaining_search_m = max(
            0.0,
            _norm([target_position[0] - tip_position[0], target_position[1] - tip_position[1], target_position[2] - tip_position[2]]),
        )
        peak_speed_m_per_s = float(getattr(primitive, "peak_speed_m_per_s", 0.0))
        local_dt = max(0.0, float(getattr(last_step, "dt", 0.0)))
        if peak_speed_m_per_s > 0.0:
            local_dt = max(remaining_search_m / peak_speed_m_per_s, local_dt)

        in_contact = bool(getattr(last_step.sensors.tool, "in_contact", False))
        signed_distance = float(getattr(cp, "signed_distance", remaining_search_m))
        if in_contact or signed_distance < _CONTACT_TOLERANCE_M:
            return TwistSceneTip(
                Twist(
                    linear=Vec3(x=0.0, y=0.0, z=0.0),
                    angular=Vec3(x=0.0, y=0.0, z=0.0),
                )
            )

        local_primitive = Reach(
            target_pose_scene=Pose(position=_vector_to_vec3(target_position), rotation=tip_pose_scene.rotation),
            duration_s=local_dt,
            jaw_target_start=primitive.jaw_target_start,
            jaw_target_end=primitive.jaw_target_end,
            end_on_contact=False,
        )
        local_active = SimpleNamespace(
            primitive=local_primitive,
            started_at_pose_scene=tip_pose_scene,
            duration_s=local_dt,
            elapsed_s=0.5 * local_dt,
            started_at_jaw=getattr(active, "started_at_jaw", 0.0),
        )
        return _evaluate_reach(active=local_active, last_step=last_step)

    def _initial_jaw(self) -> float:
        tool = getattr(self._scene_config, "tool", None)
        if tool is None:
            return 0.0
        return float(getattr(tool, "initial_jaw", 0.0))

    def reset(self, initial_step: StepResult) -> RobotCommand:
        self._sequencer.reset(initial_step)
        self._fsm.reset()
        self._cycle_id = -1
        self._last_jaw_commanded = self._initial_jaw()
        self._scene_center = None
        self._reset_called = True
        self._finalized = False
        return self.next_command(initial_step)

    def next_command(self, last_step: StepResult) -> RobotCommand:
        if not self._reset_called:
            raise RuntimeError("reset() must be called before next_command()")

        active = self._fsm.step(last_step, self._last_jaw_commanded)
        camera_matrix, camera_position = _camera_basis(last_step)
        tip_pose_scene = _camera_tool_to_scene(
            tool_pose=last_step.sensors.tool.pose,
            camera_matrix=camera_matrix,
            camera_position=camera_position,
        )
        primitive = getattr(active, "primitive")

        if isinstance(primitive, ContactReach):
            raw_scene = self._evaluate_contact_reach(
                active=active,
                last_step=last_step,
                tip_now=tip_pose_scene.position,
                tip_pose_scene=tip_pose_scene,
            )
        else:
            raw_scene = _evaluate_reach(active, last_step)
        raw_linear = _as_vec(raw_scene.data().linear)
        raw_angular = _as_vec(raw_scene.data().angular)

        duration_s = float(getattr(active, "duration_s", 0.0))
        elapsed_s = float(getattr(active, "elapsed_s", 0.0))
        tau = _time_to_fraction(elapsed_s, duration_s)
        jaw_target = _jaw_target(primitive=primitive, active=active, tau=tau)
        motion_shaping_enabled = bool(getattr(self._motion_config, "motion_shaping_enabled", False))
        motion_shaping = (
            getattr(self._motion_config, "motion_shaping", None)
            if motion_shaping_enabled
            else None
        )

        tip_now_scene = tip_pose_scene.position

        envelope = _tool_workspace_envelope(self._scene_config)
        signed_distance = _signed_distance_to_envelope(envelope=envelope, point=tip_now_scene) if envelope is not None else None

        if envelope is None or signed_distance is None or not motion_shaping_enabled:
            bias_linear = False
            final_linear = raw_linear
        else:
            bias_gain_max = float(getattr(motion_shaping, "bias_gain_max", 0.0)) if motion_shaping is not None else 0.0
            bias_ramp_distance_m = (
                float(getattr(motion_shaping, "bias_ramp_distance_m", 0.0))
                if motion_shaping is not None
                else 0.0
            )
            outer_margin_m = float(getattr(envelope, "outer_margin_m", 0.0))
            final_linear, bias_linear = _apply_linear_bias(
                raw_linear=raw_linear,
                signed_distance=float(signed_distance),
                point=tip_now_scene,
                envelope=envelope,
                bias_gain_max=bias_gain_max,
                bias_ramp_distance_m=bias_ramp_distance_m,
                outer_margin_m=outer_margin_m,
            )

        final_angular, bias_angular = (
            self._compute_orientation_bias(
                raw_angular=[float(raw_angular[0]), float(raw_angular[1]), float(raw_angular[2])],
                tip_pose_scene=tip_pose_scene,
                primitive=primitive,
                in_contact=bool(last_step.sensors.tool.in_contact),
            )
            if motion_shaping_enabled
            else ([float(raw_angular[0]), float(raw_angular[1]), float(raw_angular[2])], False)
        )
        final_scene = Twist(
            linear=Vec3(x=float(final_linear[0]), y=float(final_linear[1]), z=float(final_linear[2])),
            angular=Vec3(
                x=float(final_angular[0]),
                y=float(final_angular[1]),
                z=float(final_angular[2]),
            ),
        )
        command_twist = _twist_scene_to_camera(last_step, final_scene)
        self._last_jaw_commanded = jaw_target
        self._cycle_id += 1

        return RobotCommand(
            timestamp_ns=last_step.sensors.timestamp_ns + int(last_step.dt * 1_000_000_000),
            cycle_id=self._cycle_id,
            control_mode=ControlMode.CARTESIAN_TWIST,
            cartesian_twist=command_twist,
            frame=ControlFrame.CAMERA,
            tool_jaw_target=jaw_target,
            enable=True,
            source="scripted",
            motion_shaping=motion_shaping,
            motion_shaping_enabled=motion_shaping_enabled,
            safety=SafetyMetadata(
                clamped_linear=False,
                clamped_angular=False,
                biased_linear=bool(locals().get("bias_linear", False)),
                biased_angular=bias_angular,
                scaled_by=None,
                signed_distance_to_envelope_m=signed_distance,
                signed_distance_to_surface_m=None,
            ),
        )

    def finalize(self, last_step: StepResult) -> None:
        if self._finalized:
            return
        self._fsm.finalize(last_step)
        self._finalized = True

    @property
    def realised_sequence(self) -> tuple[RealisedPrimitive, ...]:
        return tuple(
            RealisedPrimitive(
                primitive=primitive,
                started_at_tick=started_at,
                ended_at_tick=ended_at,
                early_terminated=early_terminated,
            )
            for primitive, started_at, ended_at, early_terminated in self._fsm.completed
        )