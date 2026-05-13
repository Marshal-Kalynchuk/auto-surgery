# Motion shaping rollout sequence

## 8-step rollout order

1. **Scene geometry first (Step 1 / Section 1)**  
   Standalone geometry updates with no behavior change; verify using scene-level geometry checks.

2. **Schemas + observability + conservative limits (Section 3, 4, 9)**  
   Introduce schema updates (`SceneGraph`, `SafetyMetadata`, `MotionShaping`) and applier soft-clamp/acceleration limits.
   Keep new envelopes permissive so existing configs retain behavior.

3. **Primitive collapse to Reach + Hold (Section 2)**  
   Collapse existing evaluator paths to Reach/Hold behavior for compatibility, then delete dead code after parity checks.

4. **Sequencer envelope clamping + orientation-seeded targets (Section 5)**  
   Enable sequencer-level envelope constraints and deterministic target gating logic.

5. **Generator envelope + orientation bias (Section 6, 7)**  
   Turn on generator-side biasing and workspace-aware shaping when envelopes are available.

6. **Orchestrator predictive scaling (Section 8)**  
   Enable predictive scaling in orchestration where needed and verify no unintended limit hits on open-loop trajectories.

7. **Interaction primitives with zero-weight rollout (Section 7)**  
   Ship ContactReach/Grip/Drag/Brush primitives with weights starting at 0, then promote after
   `test_real_interaction.py` and visual replay reviews are stable.

8. **Hard enablement + visual diff gate**  
   Each step requires:
   - green integration/unit tests relevant to that area
   - deterministic replay diff check with fixed seed
   - visual smoke output review before promoting the next feature.
