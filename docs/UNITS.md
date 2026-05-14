# Physical units (DejaVu / SOFA path)

This repository’s **SOFA-backed DejaVu brain** pipeline uses a single **scene-space** convention. Do not assume SI metres in planner or command code unless an explicit adapter says so.

## Scene frame

- **Length:** millimetres (`mm`)
- **Time:** seconds (`s`)
- **Angles:** radians (`rad`)

Scene assets, tool poses, target volumes, and workspace envelopes are expressed in this frame.

## Cartesian pose commands (SOFA forceps integration)

The motion generator and SOFA forceps applier use **`RobotCommand` with `control_mode=CARTESIAN_POSE` and `frame=ControlFrame.SCENE`**:

- **`cartesian_pose_target`:** rigid pose of the **tool tip** in **scene millimetres** (position `mm`, rotation unit quaternion).
- The applier servoes the shaft `Rigid3d` DOF using **pose error** (`pose_log` on the incremental rigid error), then applies **linear mm/s** and **angular rad/s** velocity limits and **linear mm/s²** / **angular rad/s²** acceleration limits from `MotionShaping`.

`CARTESIAN_TWIST` remains a schema mode for logging and non-SOFA bridges; **`enable=True` + `CARTESIAN_TWIST` is rejected by the forceps applier** — use scene-frame pose targets instead.

## Safety and envelopes

- Workspace **signed distances** and envelope margins (`radius_mm`, `outer_margin_mm`, `inner_margin_mm`) are in **mm**.
- `SafetyMetadata.signed_distance_to_envelope_mm` / `signed_distance_to_surface_mm` match those scene-mm values.
- `SafetyMetadata.pose_error_norm_mm` / `pose_error_norm_rad` report the magnitude of the pose error vector used by the applier (diagnostics).

## Motion generator

- Primitive distances (`max_search_mm`, `lift_distance_mm`, `drag.distance_mm`, etc.) are **mm**.
- Speeds such as `peak_speed_mm_per_s` and `MotionShaping.max_linear_mm_s` are **mm/s**; linear accelerations use **mm/s²**.

## ROS / SI (non-goal here)

If a future ROS bridge needs SI, add a **single** `scene_mm ↔ si_m` adapter module. Do not reintroduce ambiguous `_m` suffixes on scene-adjacent fields.
