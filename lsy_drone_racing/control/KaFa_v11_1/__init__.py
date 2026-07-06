"""KaFa_1500_v11_1: the v11 tunnel MPCC over the v10.6 guarded-smoothed reference.

VERDICT: CONDEMNED -- the composition hypothesis is measured negative on level2 (seed-42
paired smoke: 5/6 @ 9.15 s with a 12.1 s recovery lap, vs v10.5's 6/6 @ 8.07 on the same
draws). The de-paced tunnel does not convert the shorter route into time here either; the
reveal ceiling binds the composition just as it binds each half (full ladder in
KaFa_v11/cockpit.py). Kept as the ready-made platform for exact-pose tracks.

IMPROVEMENT_PLAN.md Phase 2 (first step). The verified literature says the solver-pace gap
has two co-equal causes -- the cost formulation (Phase 1, v11's tunnel) and the reference
itself (a smoothness-optimal spline caps realized speed). v10.6's guarded waypoint smoothing
shortened the route 11.94 -> 9.93 m but was CONDEMNED under v10.5 because the cost-paced
solver converted none of it into time (its full ledger: KaFa_v10_6/cockpit.py). v11 removes
that pacing between gates -- so v11.1 re-tests the composition the research predicted to be
complementary: the de-paced tunnel MPCC flying the shorter, parity-capped smoothed reference.

Everything is reuse: the planner is v10.6's ReferenceManager (episode-sticky smoothing,
geometry guard, parity caps), the path view is v11's TunnelArcPath with the plan's parity
caps applied on top, the OCP is v11's (same compiled ``kafa_v11`` solver). REQUIRES the
acados environment -- run under ``pixi run``.
"""
