"""Gate-aware time-optimal MPCC with guarded spline smoothing (KaFa_1500_v10_6).

VERDICT: CONDEMNED AS FLAGSHIP -- race v10.5 instead. Paired-seed measured (level2 seed 42):
17/20 @ 8.18 s vs v10.5's 19/20 @ 8.08 s on the same draws. The route is shorter but the
solver's pace, not the path profile, binds level2 lap time, and the straighter approaches
arrive hotter at downstream turn folds. Mechanism and full ledger: KaFa_v10_6.cockpit.

v10.6 is v10.5 with the SPLINE GENERATION upgraded: every plan build also builds a smoothed
copy of the waypoint chain (the ~5 m of clearance detour the v10.4 audit measured but could not
trim statically), prices it with parity speed caps copied from the unsmoothed profile at the
two measured failure surfaces (gate reveal windows, obstacle passage tubes), and ships it only
if a geometry guard passes -- otherwise the plan, profile, and behaviour are byte-identical to
v10.5. Rationale, guard conditions, and offline numbers: KaFa_v10_6.trajectory / cockpit.

Everything else -- the OCP, the launch, the dynamics-aware anchor, the reactive gate caps, and
every cockpit value -- is v10.5's, sharing its compiled acados solver. Implemented as a thin
subclass of KaFa1500V105 with the NAVIGATE flow copied and the v10.6 path view woven in (the
house pattern). REQUIRES the acados environment -- run under ``pixi run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.kafa1500_v6.attitude import _vector_to_attitude
from lsy_drone_racing.control.KaFa_1500_v10_5 import KaFa1500V105
from lsy_drone_racing.control.KaFa_v10_6.arc_path import GateArcPath
from lsy_drone_racing.control.KaFa_v10_6.settings import ControllerSettings
from lsy_drone_racing.control.KaFa_v10_6.trajectory import ReferenceManager

if TYPE_CHECKING:
    from lsy_drone_racing.control.kafa1500_v6.state import DroneObservation


class KaFa1500V106(KaFa1500V105):
    """v10.5's racing MPCC planning over smoothed, parity-capped reference splines."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.5 (its MPCC/solver is reused as-is), then swap in the v10.6 planner."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        self._references = ReferenceManager(
            self._settings.planner,
            self._settings.runtime.replan_gate_delta_m,
            self._settings.runtime.replan_obstacle_delta_m,
            self._v_theta_max,
            self._a_lat_max,
            self._v_min,
        )

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """v10.5's flow verbatim, building the path view with the plan's parity caps."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:  # first NAVIGATE tick: nothing is revealed yet
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        new_plan = rebuilt or self._path is None
        if new_plan:
            first = max(frame.target_gate, 0)
            gates_ahead = plan.gate_pos_snapshot[first:]
            deltas = np.linalg.norm(
                plan.gate_pos_snapshot[first:] - self._gate_nominal[first:], axis=1
            )
            caps = np.where(deltas > self._react_delta, self._v_gate_react, np.inf)
            if first == 0:  # gate 0 is always capped: launch-window protection (v10.4 cockpit)
                caps[0] = self._v_gate_react
            path = GateArcPath(
                plan.curve,
                plan.t_total,
                self._v_theta_max,
                self._a_lat_max,
                self._v_min,
                gates_ahead,
                self._w_base,
                self._w_gate,
                self._gate_sigma,
                caps,
                self._react_v_pre,
                self._react_v_post,
                window_caps=plan.gate_window_caps,
                window_pre=self._settings.planner.reveal_window_m,
                obstacle_caps=plan.obstacle_caps,
            )
            self._s = path.project(frame.pos, 0.0)
            if self._path is None:  # episode start / first plan: honest cold start (v10.4 mpcc)
                self._mpcc.set_path(path)
            else:  # mid-flight replan: keep the warm start, re-anchor onto the new path
                self._mpcc.rebase(path, self._s)
            self._path = path
        # v10.5's dynamics-aware anchor, including its telemetry.
        th_pred = self._mpcc.predicted_progress()
        if th_pred is None:  # first solve of a plan: geometric fallback
            self._s = self._path.project(frame.pos, self._s)
        else:
            self._s = self._path.project_near(frame.pos, th_pred, self._proj_band)
            self._band_calls += 1
            if abs(self._s - th_pred) >= self._proj_band - 1e-9:
                self._band_edge_hits += 1
        if not new_plan and self._anchor_prev_s is not None:
            self._anchor_jumps.append(abs(self._s - self._anchor_prev_s))
        self._anchor_prev_s = self._s
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
