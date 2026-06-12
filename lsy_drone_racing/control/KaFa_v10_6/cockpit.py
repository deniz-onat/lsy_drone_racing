"""Tunable constants (the "cockpit") for the KaFa_1500_v10_6 controller.

v10.6 = v10.5 + GUARDED SPLINE SMOOTHING (see KaFa_v10_6.trajectory). Every racing/launch/anchor
knob -- speed budget, contouring weights, ramp, takeoff, reactive caps, anchor band -- is
inherited UNCHANGED from v10.5. The new knobs only shape the smoother and its acceptance guard.

VERDICT: CONDEMNED AS FLAGSHIP -- v10.5 REMAINS THE LINE'S SHIP. v10.6 is the NINTH mechanism
measured onto the v10.x speed/robustness frontier, and the first ROUTE-SHAPE one. Keep it for
study; do not race it on level2. The full measured chain, paired-seed protocol
(scripts/compare_v10_5.py, same track draws per seed):

  OFFLINE the smoothed route is unambiguously better (scripts/analyze_spline.py, level2
  nominal): arc 11.94 -> 9.93 m, predicted NAVIGATE 6.29 -> 5.34 s, with gate reveal-window
  and obstacle-passage profile speeds pinned at or below the v10.4 plan's everywhere.

  IN FLIGHT the gain does not materialise. A per-tick probe (|vel| vs profile v_curv at the
  anchor) shows the MPCC flies 0.5-0.9 m/s BELOW the curvature profile on every leg, both
  controllers: the solver's contouring pace, not the path profile, is the binding constraint
  on level2. A shorter route therefore converts to ~-0.1 s typical, not the predicted -0.9 s.

  ITERATION LEDGER (level2 seed 42 x 20 vs v10.5's 19/20 @ 8.076 on the same draws):
    per-rebuild accept/reject (first cut) ......... 15/20 @ 8.04  (accept/reject flips teleport
                                                    the reference ~0.5 m mid-flight; 3 fails at
                                                    gate 3 + one 14.9 s near-miss recovery)
    + episode-sticky mode + run-matched caps ...... 17/20 @ 8.23  (flips gone; gate-3 fails
                                                    remain -- the smoothed approach lets the
                                                    solver carry extra REALIZED speed into the
                                                    g2->g3 U-turn fold)
    + reversal legs protected (final form) ........ 17/20 @ 8.18  (still -2 finishes, and 2/20
                                                    runs lose ~1.7 s to wide-turn recoveries)
  FINAL FORM, 3 SEEDS (42/7/123 x 20): v10.6 50/60 @ 8.33 vs v10.5 54/60 @ 8.13 on the same
  draws. Per-seed: 17/20 vs 19/20, 14/20 vs 18/20 (four gate-3 U-turn fails), 19/20 vs 17/20
  -- high seed variance, net -4 finishes for +0.20 s. CONDEMNED.

  WHY PARITY CAPS CANNOT FIX THE TAIL: the caps pin the PROFILE to the unsmoothed plan's, but
  realized speed is profile MINUS a geometry-dependent solver gap -- straighter approaches
  shrink the gap, so the drone genuinely arrives hotter at downstream hazards even under
  byte-equal profile caps. Pinning REALIZED speed would mean capping below the base profile,
  i.e. paying back the route gain. That is the frontier, reasserting itself through route
  shape exactly as it did through the eight v10.3/v10.4 speed knobs.

  WHAT SURVIVES: (1) scripts/analyze_spline.py and the knot-timing study (v8's heuristic knot
  spacing is shape-optimal already; chord/centripetal re-timing buys nothing); (2) the sticky
  plan-mode lesson -- NEVER re-decide a binary plan-shape choice per rebuild; (3) the
  reversal-leg sanctity result; (4) this mechanism may pay on a future stack whose solver
  pace tracks the profile (a v11 with a tighter realized-speed gap), which is why the
  implementation stays.
"""

from __future__ import annotations

# Re-export every v10.5 knob unchanged (budget 3.2/8.5, ramp 0.25/2.4, mini-takeoff, reactive
# gate caps, anchor band, contouring, solver...) -- and the whole v10.x ledger beneath it.
from lsy_drone_racing.control.KaFa_v10_5.cockpit import *  # noqa: F401,F403

# --- Waypoint smoothing (KaFa_v10_6.trajectory.smooth_waypoints) ---
# Free waypoints move pull * (neighbour midpoint - self) per sweep; gate triplets, chain ends,
# the replan-continuity arc, and whole reversal legs never move.
SMOOTH_PULL = 0.5
SMOOTH_ITERS = 3

# --- Parity caps (KaFa_v10_6.arc_path) ---
# Window/tube geometry for the caps copied from the unsmoothed profile. The reveal window is
# the 0.7 m sensor range upstream of each gate plane (where the reveal correction is absorbed);
# the obstacle tube is the passage region where a +/-0.15 m obstacle shift revealed at 0.7 m
# still demands a swerve. Obstacle caps are run-matched arc intervals (see trajectory).
REVEAL_WINDOW_M = 0.7
OBS_CAP_RADIUS = 0.45

# --- Acceptance guard (KaFa_v10_6.trajectory.ReferenceManager.build, build 1 per episode) ---
# A smoothed candidate ships only if every gate-plane crossing inside the frame square threads
# the opening within CROSS_TOL_M on both axes (audited v10.4 crossings sit at 0.002-0.088 m;
# the opening half-width is 0.20 m), real-obstacle clearance keeps r_obs, and the capped
# profile beats the base profile by MIN_GAIN_S. The decision is episode-sticky (see trajectory).
CROSS_TOL_M = 0.15
MIN_GAIN_S = 0.05
