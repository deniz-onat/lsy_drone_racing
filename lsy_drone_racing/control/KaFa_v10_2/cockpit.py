"""Tunable constants (the "cockpit") for the KaFa_1500_v10_2 controller.

v10.2 = v10.1 with a DYNAMICS-AWARE progress anchor. Every speed/contouring/solver knob is inherited
unchanged from v10.1 (see KaFa_v10_1.cockpit); the only new knob is PROJ_BAND_M.

Why this exists: v10.1 anchors the MPCC's progress state at the GLOBAL nearest path point in a
2.0 m forward window. On a sharp slalom the path folds back on itself near a gate, so a far leg
of the fold is spatially closer than the gate apex and the anchor snaps ~1-2 m forward across the
fold in one step -- the drone skips the gate. v10.2 instead anchors progress to the SOLVER'S OWN
predicted progress (dynamics-feasible, advances at the bounded rate vth, cannot teleport) and lets a
geometric search correct it only within +/- PROJ_BAND_M. The far fold leg lies outside the band and
can never be selected; the gate-apex motion lies inside it, so tracking is unaffected.

Tuning PROJ_BAND_M: set it just ABOVE the legitimate per-step fold advance (~0.6-0.7 m on this
track, measured) so the anchor tracks the drone rounding the doubled-over path, and BELOW the
fold's self-approach gap (~1 m) so the far leg stays out of reach. 0.6 m is the measured sweet spot
(skip eliminated -- max anchor jump drops from ~2.0 m to ~0.7 m -- at finish within run-to-run noise
of v10.1). Smaller starts to clamp the legitimate fold motion (finish falls); larger lets the skip
creep back (>=1.0 m the teleport returns).
"""

from __future__ import annotations

# Re-export every v10.1 knob unchanged (HORIZON, speed budget, gate-aware contouring, solver...).
from lsy_drone_racing.control.KaFa_v10_1.cockpit import *  # noqa: F401,F403

# --- Dynamics-aware progress anchor (the v10.2 mechanism) ---
# Half-width (m) of the geometric correction applied around the solver's predicted progress. Just
# above the legitimate per-step fold advance (~0.6-0.7 m) and below the fold self-approach gap
# (~1 m), so the anchor tracks the drone through the fold but can never teleport across it.
PROJ_BAND_M = 0.6
