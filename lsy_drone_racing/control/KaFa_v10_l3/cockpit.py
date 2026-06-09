"""Tunable constants (the "cockpit") for the KaFa_1500_v10_l3 SEARCH phase.

Only the search phase is new; navigation reuses v10's MPCC and its cockpit unchanged. The
search sweep is an expanding circle (Archimedean spiral): one plain circle cannot pass within
the ~0.7 m horizontal sensor range of gates scattered across the 5x3 m arena, so the radius
grows each revolution until the whole arena (out to where gates can sit) has been swept. The
sweep is flown by the same v10 MPCC, so these knobs only shape the spiral and how fast/where it
is flown.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Search altitude (detection is HORIZONTAL-only, so altitude is free; pick a safe height) ---
# Gate/obstacle detection uses the XY distance to the object, NOT 3-D, so the sweep height does
# not affect what is found. 1.8 m clears the obstacle poles (1.55 m) and the tops of the tall
# gates (~1.66 m), so the drone can fly straight over everything while sweeping.
SEARCH_ALT = 1.8        # m, constant altitude of the search sweep

# --- Spiral geometry (the expanding circle) ---
# Waypoints r(theta)=a*theta from R0 outward to R_MAX, clipped to the arena. RADIAL_STEP is how
# much the radius grows per full revolution; it must be < the ~1.4 m sensor swath (2*0.7) so
# successive loops overlap and leave no gap a gate could hide in. R0 starts the spiral off-centre
# to avoid the high-curvature singularity at r=0. R_MAX reaches the far gates (seen out to ~2.2 m).
SPIRAL_R0 = 0.5         # m, innermost radius
SPIRAL_R_MAX = 2.3      # m, outermost radius
SPIRAL_RADIAL_STEP = 1.0  # m, radius growth per revolution (< 1.4 m swath for overlap)
SPIRAL_ANGLE_STEP = 0.32  # rad, angular spacing of spiral waypoints (~pi/10)
ARENA_X_LIM = 2.3       # m, clip spiral |x| (inside the 2.5 m safety limit)
ARENA_Y_LIM = 1.3       # m, clip spiral |y| (inside the 1.5 m safety limit)

# --- Search speed (passed to the v10 MPCC as its progress-rate cap during SEARCH) ---
# Slower than the race so the sweep is steady and the sensor dwells near each object. The MPCC's
# friction-circle cap brakes the tight inner loops automatically; SEARCH_A_LAT/V_MIN set that cap
# for the search path (separate from the race values so the inner loops don't crawl).
SEARCH_SPEED = 2.0      # m/s, top sweep speed (MPCC progress-rate cap during SEARCH)
SEARCH_A_LAT = 6.0      # m/s^2, lateral budget setting the sweep's corner speed
SEARCH_V_MIN = 0.8      # m/s, floor so the tight inner loops don't crawl to a stop

# --- Ease into the sweep after the takeoff hand-off (same idea as v10's nav ramp) ---
RAMP_START = 0.1        # fraction of SEARCH_SPEED right after the takeoff hand-off
RAMP_S = 1.2            # s, time to ramp the sweep speed up to full

# --- STAGE: reposition behind gate 0 before racing (search -> navigate hand-off) ---
# The sweep ends at an arbitrary pose; gate yaw is randomised and the env counts gate 0 only when
# crossed in its canonical +x direction. When the sweep ends on the EXIT (+x) side of gate 0 the
# race planner must reverse around the gate -> a cusp the MPCC crashes on (the dominant transition
# failure). STAGE detects that and loops the drone (at the sweep altitude, above the obstacles) to
# behind gate 0's entry first. STAGE_MIN_ALONG is how far onto the exit side (m, measured along the
# gate's +x axis) the drone must be before staging triggers; entry-side hand-offs (along<=0) skip it.
# along = (drone - gate0)·(gate +x). A drone only slightly past the gate plane (along < ~0.7 m)
# the planner still threads with its own run-in/nudge and finishes; staging it there only perturbs
# a working approach. The cusp turns fatal once the drone is well onto the exit side, so trigger
# staging only beyond that band.
STAGE_MIN_ALONG = 0.9   # m, trigger staging only when clearly on the wrong (exit) side of gate 0
STAGE_D_ENTRY = 0.6     # m, hand-off point behind gate 0 (just behind the planner's run-in)
STAGE_SIDE_W = 0.85     # m, lateral swing clearing the 0.72 m gate frame as the drone loops around
STAGE_SPEED = 1.8       # m/s, top staging speed (MPCC progress-rate cap during STAGE)
STAGE_REACH = 0.30      # m, XY distance to the entry point at which STAGE hands off to NAVIGATE
STAGE_T_MAX = 4.0       # s, stage at most this long, then race regardless (safety)

# --- NAVIGATE budget for level3 (gentler than v10's level2 race) ---
# Level3 is online planning, not a speed run: the drone enters NAVIGATE high (~1.8 m) and moving
# (~1.7 m/s) from the sweep, then must dive and thread gates scattered across the arena. v10's
# level2 budget (v_theta 3.0 / a_lat 8.0) overshoots that transient and the scattered geometry,
# so level3 races at a lower budget for reliability. These override v10's nav caps in __init__
# (the MPCC solver itself is unchanged -- these are the ArcPath/progress-rate caps it is fed).
NAV_V_THETA_MAX = 2.3   # m/s, navigate progress-rate cap on level3 (vs v10's 3.0)
NAV_A_LAT_MAX = 6.0     # m/s^2, navigate lateral budget on level3 (vs v10's 8.0)

# Descent authority: the drone enters NAVIGATE high (the safe ~1.8 m sweep height) and must dive
# ~0.9 m to the first gate while moving. v10's a_z_min=7.0 caps descent accel at ~2.8 m/s^2, so it
# overshoots the first gate horizontally before it can drop. Lowering a_z_min lets the MPCC cut
# vertical thrust harder and descend in time. This REBUILDS the MPCC solver (a_z_min is baked into
# its thrust constraint), so it is a level3-only MPCC; search (level flight) is unaffected.
NAV_A_Z_MIN = 4.5       # m/s^2, min vertical thrust accel (vs v10's 7.0) -> faster descent


@dataclass(frozen=True)
class SearchSettings:
    """Search-phase knobs (from the cockpit constants above)."""

    alt: float = SEARCH_ALT
    r0: float = SPIRAL_R0
    r_max: float = SPIRAL_R_MAX
    radial_step: float = SPIRAL_RADIAL_STEP
    angle_step: float = SPIRAL_ANGLE_STEP
    arena_x: float = ARENA_X_LIM
    arena_y: float = ARENA_Y_LIM
    speed: float = SEARCH_SPEED
    a_lat: float = SEARCH_A_LAT
    v_min: float = SEARCH_V_MIN
    ramp_start: float = RAMP_START
    ramp_s: float = RAMP_S
    stage_min_along: float = STAGE_MIN_ALONG
    stage_d_entry: float = STAGE_D_ENTRY
    stage_side_w: float = STAGE_SIDE_W
    stage_speed: float = STAGE_SPEED
    stage_reach: float = STAGE_REACH
    stage_t_max: float = STAGE_T_MAX
