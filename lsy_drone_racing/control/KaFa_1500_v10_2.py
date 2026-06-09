"""Gate-aware time-optimal MPCC for known tracks, robust on sharp slaloms (KaFa_1500_v10_2).

v10.2 is v10.1 with ONE fix: a dynamics-aware progress anchor. v10.1 is fast and precise at the
gates (per-stage gate-aware contouring weight), but it anchors the MPCC's progress state at the
GLOBAL nearest path point over a 2.0 m forward arc window (inherited from v10's ArcPath.project). On
a sharp slalom -- where the drone enters and leaves a gate from the same side -- the planned path
folds back on itself into a cusp at the gate (the steep-angle crossing plus the d_pre/d_post
run-in/out place the approach/exit waypoints so the spline overshoots). When the drone is near that
fold, a far-along-the-arc leg of the cusp is spatially CLOSER than the gate apex, so the nearest-
point search snaps the progress anchor ~1-2 m forward ACROSS the fold in a single step. Progress is
hard-set (MPCC x0 bound) to that post-gate arc, so the drone cuts straight across and SKIPS the
gate. This was observed on config/level2_sharp_slalom.toml.

Constraining the forward search alone does not work: at the fold the path is nearly doubled over, so
the LEGITIMATE foot-point genuinely advances ~0.7 m per step as the drone rounds the apex (measured,
even in runs that finish). Any forward window / jump cap tight enough to block the ~1-2 m skip also
clamps that legitimate motion and the controller stalls or destabilises -- every such variant
regressed finish, one catastrophically (7/20 at a seed v10.1 finished 13/20).

The cure is to anchor progress to the SOLVER'S OWN predicted progress (the th state one step ahead;
see KaFa_v10_2.mpcc.MPCC.predicted_progress). That value is dynamics-feasible -- it advances at the
bounded progress rate vth -- so it cannot teleport across a fold. A geometric search then corrects
it only within +/- PROJ_BAND_M (KaFa_v10_2.arc_path.GateArcPath.project_near): the far fold leg lies
outside the band and can never be selected, while the gate-apex motion lies inside it, so tracking
is unchanged. Measured on the sharp slalom over 3 seeds x 20 runs, this drops the worst single-step
anchor jump from ~2.0 m to ~0.7 m (the skip is gone) at finish within run-to-run noise of v10.1,
with no catastrophic regression at any seed. The solver, the speed budget, the gate-aware contouring
weight, the planner, and the takeoff are all v10.1 -- only the progress anchor changes.

Implemented as a thin subclass of KaFa1500V101: it reuses v10.1's flow and the cached acados solver,
swaps in the v10.2 MPCC (which exposes the predicted progress) and the band setting, and overrides
_track_action to anchor progress to the prediction. REQUIRES the acados environment -- run under
``pixi run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.kafa1500_v6.attitude import _vector_to_attitude
from lsy_drone_racing.control.KaFa_1500_v10_1 import KaFa1500V101
from lsy_drone_racing.control.KaFa_v10_2.arc_path import GateArcPath
from lsy_drone_racing.control.KaFa_v10_2.mpcc import MPCC
from lsy_drone_racing.control.KaFa_v10_2.settings import ControllerSettings

if TYPE_CHECKING:
    from lsy_drone_racing.control.kafa1500_v6.state import DroneObservation


class KaFa1500V102(KaFa1500V101):
    """v10.1's gate-aware time-optimal MPCC with a dynamics-aware (predicted-progress) anchor."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.1, then swap in the v10.2 settings, MPCC (predicted-progress) and anchor band."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(mpcc, a_max)  # v10.1 solver + predicted_progress() accessor
        # Refresh the cached limits v10's flow reads, from the v10.2 settings (== v10.1's values).
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._w_base, self._w_gate, self._gate_sigma = (
            mpcc.w_contour_base, mpcc.w_contour_gate, mpcc.gate_sigma,
        )
        self._proj_band = mpcc.proj_band_m  # half-width of the geometric correction (the fix)
        self._path: GateArcPath | None = None
        self._s = 0.0

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """Fly the plan; anchor progress to the solver's prediction with a bounded geometric fix."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt or self._path is None:  # new plan -> rebuild the gate-aware arc view and reload
            gates_ahead = plan.gate_pos_snapshot[max(frame.target_gate, 0):]
            self._path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
            )
            self._mpcc.set_path(self._path)  # also clears the solver's warm start (no prediction yet)
            self._s = self._path.project(frame.pos, 0.0)
        th_pred = self._mpcc.predicted_progress()
        if th_pred is None:  # first step after a (re)plan: no prediction yet -> geometric anchor
            self._s = self._path.project(frame.pos, self._s)
        else:  # anchor to the dynamics-feasible prediction, correct geometrically within the band
            self._s = self._path.project_near(frame.pos, th_pred, self._proj_band)
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
