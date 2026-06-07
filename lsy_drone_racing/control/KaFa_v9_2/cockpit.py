"""Tunable constants (the "cockpit") for the KaFa_1500_v9.2 controller.

v9.2 = v9.1 with the MPCC reference's CONSTANT recede rate (v9.1's V_REF=1.8) replaced by a
curvature-aware speed profile (KaFa_v9_1.speed_profile): the reference slows into sharp turns
and speeds up on the straights, so the straights run toward V_CAP -- above v9.1's fixed
V_REF -- while the tight gate-2/3 turns self-brake instead of being flown at one blunt safe
rate. Only the look-ahead spacing changes; v9.1's MPCC (still the safety net via its v_max /
tilt / thrust limits), stall governor, planner, and takeoff are reused unchanged.

These are the only new knobs; everything else is inherited from the v9.1 cockpit.
"""

from __future__ import annotations

# --- Straight-line speed cap (the fast end of the profile) ---
# How fast the reference may recede on a straight; the main speed-up lever (v9.1's constant
# V_REF was 1.8). 2.4 holds 100% finish on level2 while cutting lap time vs v9.1. It only
# stays safe because the ramp below eases the takeoff->gate-0 hand-off. Raising to 2.8 is ~3%
# quicker again but the gate-0 transient starts to bite (finish ~92%), so 2.4 is the
# reliability-first default; bump to 2.8 to trade a little finish rate for speed.
V_CAP = 2.4         # m/s, max reference recede speed on straights
V_MAX = 3.0         # m/s, MPCC hard velocity cap -- headroom above V_CAP so the tracker can
                    # catch the faster reference instead of clipping it at the profile cap

# --- Corner speed (friction-circle cap) ---
# Corner speed = sqrt(A_LAT_MAX / curvature), so a bigger budget corners faster. tilt_ratio*g
# (~8 at tilt 0.85) is the physical ceiling, and the sweep landed there: 8.0 brakes the sharp
# gate-2/3 reversal just enough to stay on the path (the win), while 12+ corners too fast and
# overshoots those gates (finish drops to 40-50%, even though the laps that survive are quicker).
A_LAT_MAX = 8.0     # m/s^2, lateral-acceleration budget that sets corner speed

# --- Floor ---
# Keep the reference always creeping forward so it can't stall to a crawl at a spline cusp.
V_MIN = 1.3         # m/s, minimum reference recede speed

# --- Navigation-start ramp (gentler than v9.1's, to absorb the higher V_CAP) ---
# v9.2 recedes faster than v9.1, so the takeoff->gate-0 hand-off needs more easing or the drone
# lunges at the first gate and crashes (~2 s, gate 0). Ramping the look-ahead in from
# RAMP_START over RAMP_S (vs v9.1's 0.2 / 1.1 s) removes those transient failures: it took the
# finish rate from ~75% back to 100% on the hard seed sets. 1.4 s is the knee -- v9.1's 1.1 s
# still lunges, while 2.0 s over-eases and slows the first two gates for no reliability gain.
RAMP_START = 0.12   # fraction of full look-ahead right after the takeoff hand-off
RAMP_S = 1.4        # s, time to ramp the look-ahead up to full
