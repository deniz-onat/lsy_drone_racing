"""Tunable constants (the "cockpit") for the KaFa_1500_v10 controller.

v10 keeps v9's gate-aware planner and vertical takeoff but replaces v9/v9.1's
fixed-recede-rate contouring MPCC with a *time-optimal* MPCC: the path progress (arc
length) and its rate are promoted to decision variables, and the cost pays the optimiser to
make progress (``-MU * v_theta``). Speed is therefore an OUTPUT of the optimisation at the
true dynamic limit -- the solver rides V_MAX on straights and auto-brakes into tight gate
clusters against the SAME thrust/tilt/velocity limits -- instead of being bought through a
hand-capped V_REF and the lag weight (which the v9 cockpit documents overshoots gates when
pushed). This also removes the need for v9.1's external projection and 5-knob stall
governor: progress-as-a-state cannot freeze.

Only the MPCC knobs live here; the planner/takeoff tuning is inherited from v8 via v9.
"""

from __future__ import annotations

# --- Horizon (look-ahead) ---
HORIZON = 18        # number of prediction steps
STEP_DT = 0.05      # s, prediction step (HORIZON * STEP_DT = look-ahead, ~0.9 s)

# --- Actuator limits (these set the achievable speed; the corner speed is now emergent) ---
# V_MAX can be opened up relative to v9.1's 2.4: because the optimiser explicitly brakes for
# corners (pushing harder there would blow the contour/tilt limits), V_MAX no longer doubles
# as the only corner-safety mechanism, so it can be the straight-line top speed.
V_MAX = 3.0         # m/s, hard velocity cap (now ~the straight-line top speed)
TILT_RATIO = 0.65   # tan(max tilt), about 33 deg
A_Z_MIN = 7.0       # m/s^2, minimum vertical thrust acceleration

# --- Progress (the time-optimality driver) ---
# MU is the single lap-time knob: the linear reward -MU*v_theta pays the solver to advance
# along the path. Raise it until straights reach V_MAX and the success rate still holds; too
# large and the drone cuts corners (gate misses). V_THETA_MAX caps the reference rate (kept
# equal to V_MAX). A_THETA_MAX caps how fast the reference may accelerate; ~tilt_ratio*g is
# the physical forward-accel scale, set a bit above. R_DV smooths the progress command.
MU = 0.15           # progress reward weight (the lap-time lever)
V_THETA_MAX = 3.0   # m/s, max progress (reference) rate
A_THETA_MAX = 8.0   # m/s^2, max progress acceleration
R_DV = 0.01         # progress-acceleration regulariser

# --- Contouring weights ---
# W_LAG keeps the progress state synced to the drone's foot-point (so the reference can't run
# ahead of a thrust/tilt-limited drone); it is no longer the speed lever (MU is), so it drops
# from v9.1's 1.5 back to 1.0. W_CONTOUR (stay on the path) dominates and enforces gates.
W_CONTOUR = 14.0
W_LAG = 1.0
W_ACCEL = 0.02

# --- Navigation-start ramp (settle onto the path before accelerating) ---
# After the takeoff handoff the drone is slightly off the path; ramp the progress-rate cap
# from V_THETA_MAX*RAMP_START up to full over RAMP_S so it settles before charging the gate.
RAMP_S = 1.1
RAMP_START = 0.2

# --- Path grid (the fixed arc-length grid the spline reference is sampled onto) ---
# The MPCC reference p_d(theta) is a CasADi b-spline over this FIXED grid; the actual plan's
# node positions are passed as parameters (so the NLP is built once and never rebuilt on a
# replan). S_MAX must exceed the longest remaining-path length; N_NODES sets the resolution
# (~7 cm here) so gate curvature is resolved.
S_MAX = 14.0        # m, arc-length extent of the reference grid
N_NODES = 200       # b-spline nodes over [0, S_MAX]

# --- Solver ---
MAX_ITER = 60       # ipopt iteration cap per control step
GRAVITY = 9.81
