"""Gate-aware time-optimal MPCC package for KaFa_1500_v10_6.

VERDICT: CONDEMNED AS FLAGSHIP -- v10.5 remains the ship. Measured: the smoothed route is
~0.9 s better offline but the MPCC flies 0.5-0.9 m/s below the curvature profile (solver
pace, not the path, binds level2 lap time), so only ~0.1 s materialises -- and the straighter
approaches carry extra REALIZED speed into downstream turn folds, costing ~2/20 finishes that
profile-parity caps cannot prevent. Full ledger: KaFa_v10_6.cockpit. Kept for study and for a
future stack with a tighter realized-speed gap.

v10.6 = v10.5 (fast launch + dynamics-aware anchor) + GUARDED SPLINE SMOOTHING. The v10.x
lap-time levers split into two regimes: speed INSIDE the gate reveal windows (proven to trade
finish 1-for-1; the v10.3/v10.4 ledgers condemned every knob that raises it) and time spent
OUTSIDE them (path length + curvature, ~2 s of slack on level2 that the condemned static trims
tried and failed to harvest safely). v10.6 attacks only the second regime:

- KaFa_v10_6.trajectory builds BOTH the shipped v10.4 waypoint chain and a smoothed copy
  (free waypoints relaxed toward straightness, gate triplets pinned), prices the smoothed plan
  with PARITY CAPS copied from the unsmoothed profile (per-gate reveal windows, per-obstacle
  passage tubes -- the two measured failure surfaces), and ships it only if a geometry guard
  passes; otherwise the plan is byte-identical to v10.4's.
- KaFa_v10_6.arc_path applies those caps to the speed profile with the usual feasibility
  passes, on top of v10.5's reactive caps and bounded-projection anchor.

The OCP, horizon, launch, and anchor are v10.5's, so the compiled acados solver is shared
(same ``kafa_v10_4`` codegen namespace). Implemented as a thin subclass of KaFa1500V105.
REQUIRES the acados environment -- run under ``pixi run``.
"""
