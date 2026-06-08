"""Gate-aware time-optimal MPCC drone racing controller for known tracks (KaFa_1500_v10_1).

v10.1 is v10 made *precise where it matters*. v10 already lets the optimiser pick traversal speed
(progress is a state, the cost rewards progress, a friction-circle cap brakes corners), but it
uses ONE constant contouring weight, so raising the speed budget trades contour error for
progress everywhere -- including at the gates -- and the drone overshoots. On this track the lap
time is capped by gate-passing precision under +/-0.15 m gate randomisation, not by actuator
authority, so that overshoot is the ceiling. v10.1 makes the contouring weight a per-stage value
that spikes in a window around each gate's arc-position (see KaFa_v10_1.arc_path / KaFa_v10_1.mpcc):
the drone hugs the line exactly where a gate must be threaded and stays free on the straights,
which lets v10.1 carry a higher speed budget (V_MAX, A_LAT_MAX, V_THETA_MAX) than v10 without
losing gates.

Only the MPCC and how the path is built differ from v10, so this is a thin subclass of
KaFa1500V10: it reuses v10's takeoff, the foot-point progress anchor, and the NAVIGATE flow, and
overrides _track_action only to build the gate-aware path (GateArcPath) from the plan's gate
snapshot. Acceleration stays the zero-order-hold control, matching the plant -- the v10 solver
structure is unchanged apart from one extra per-stage parameter (the gate weight).

REQUIRES the acados environment -- run under ``pixi run`` (sets ACADOS_SOURCE_DIR + the tera
renderer). The C solver is code-generated/compiled once per process into c_generated_code/.

Measured on level2 (scripts/evaluate.py, 20 runs, vs v10 at the same call): v10.1 at the shipped
config (V_MAX = V_THETA_MAX = 3.2, A_LAT_MAX = 8.5, gate weight base 20 + peak 20) ran
19/20 (95%) at avg 8.27 s / best 7.66 s, against v10's 16/20 (80%) at avg 8.45 s -- faster and at
least as reliable, with a tighter worst case. The gain is modest because the binding limit is
gate-passing precision under randomisation, not actuator authority; the gate-aware weight buys a
little speed at equal-or-better finish, not a large jump.

Tuning recipe to push further (re-confirm over 20 runs each; the run-to-run finish noise is ~+/-2):
  1. Keep W_CONTOUR_BASE = 20 (v10's value). Dropping it below 20 tracks gate APPROACHES looser
     than v10 and HURTS finish -- the sweep showed this clearly. Stack the gate weight ON TOP.
  2. Speed-leaning: raise V_MAX = V_THETA_MAX (3.2 -> 3.3) and A_LAT_MAX (8.5 -> 9.0). This is a
     little faster but drops finish toward ~70-75%; only worth it if the score rewards speed over
     finish. 3.6 / 10.0 collapses finish -- do not ship it.
  3. If gate-0 crashes appear at a higher budget, lengthen RAMP_S (2.0 -> 2.4) before backing off.

Note: an earlier v10.1 tried a thrust-vector slew limit (acceleration as a state, jerk as the
control). It was dropped -- the plant takes acceleration as a zero-order-hold command, so a
jerk-as-state model mismatched it and tracked worse than v10, and v10's plans were already well
tracked in sim (attitude lag was not the limiter). Gate precision is, hence this design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.kafa1500_v6.attitude import _vector_to_attitude
from lsy_drone_racing.control.KaFa_1500_v10 import KaFa1500V10
from lsy_drone_racing.control.KaFa_v10_1.arc_path import GateArcPath
from lsy_drone_racing.control.KaFa_v10_1.mpcc import MPCC
from lsy_drone_racing.control.KaFa_v10_1.settings import ControllerSettings

if TYPE_CHECKING:
    from lsy_drone_racing.control.kafa1500_v6.state import DroneObservation


class KaFa1500V101(KaFa1500V10):
    """v10's time-optimal MPCC with a gate-aware contouring weight, flown at a raised budget."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10, then swap in the v10.1 settings and the gate-aware time-optimal MPCC."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(mpcc, a_max)
        # Refresh the cached limits v10's flow reads, from the v10.1 settings.
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        # Gate-aware contouring knobs (read when (re)building the GateArcPath).
        self._w_base = mpcc.w_contour_base
        self._w_gate = mpcc.w_contour_gate
        self._gate_sigma = mpcc.gate_sigma
        self._path: GateArcPath | None = None
        self._s = 0.0

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """Fly the plan with the gate-aware MPCC; the contouring weight spikes at each gate."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt or self._path is None:  # new plan -> rebuild the gate-aware arc view and reload
            gates_ahead = plan.gate_pos_snapshot[max(frame.target_gate, 0):]
            self._path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
            )
            self._mpcc.set_path(self._path)
            self._s = self._path.project(frame.pos, 0.0)
        self._s = self._path.project(frame.pos, self._s)  # advance the foot-point anchor
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
