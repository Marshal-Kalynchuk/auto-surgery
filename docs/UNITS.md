# Physical units (DejaVu / SOFA path)

This repository’s **SOFA-backed DejaVu brain** pipeline uses a single **scene-space** convention. Do not assume SI metres in planner or command code unless an explicit adapter says so.

## Scene frame

- **Length:** millimetres (`mm`)
- **Time:** seconds (`s`)
- **Angles:** radians (`rad`)

Scene assets, tool poses, target volumes, and workspace envelopes are expressed in this frame.

## Cartesian twist commands (SOFA integration)

For `RobotCommand` with `control_mode=CARTESIAN_TWIST` and `frame=ControlFrame.CAMERA` on the SOFA forceps path:

- **Linear:** millimetres per second (`mm/s`) in **camera axes** at the tool tip (before rigid mappings).
- **Angular:** radians per second (`rad/s`) in camera axes.

The applier maps camera-frame twist to scene frame (including rigid adjoint / offsets) and tip-to-shaft transforms; SOFA `Rigid3d` `velocity` uses the same **linear mm/s** convention for the integrated degrees of freedom after those mappings, with **angular rad/s** unchanged in meaning.

## Safety and envelopes

- Workspace **signed distances** and envelope margins (`radius_mm`, `outer_margin_mm`, `inner_margin_mm`) are in **mm**.
- `SafetyMetadata.signed_distance_to_envelope_mm` / `signed_distance_to_surface_mm` match those scene-mm values.

## Motion generator

- Primitive distances (`max_search_mm`, `lift_distance_mm`, `drag.distance_mm`, etc.) are **mm**.
- Speeds such as `peak_speed_mm_per_s` and `MotionShaping.max_linear_mm_s` are **mm/s**; linear accelerations use **mm/s²**.

## ROS / SI (non-goal here)

If a future ROS bridge needs SI, add a **single** `scene_mm ↔ si_m` adapter module. Do not reintroduce ambiguous `_m` suffixes on scene-adjacent fields.
