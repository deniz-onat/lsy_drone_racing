"""Level-3 search-then-navigate drone racing controller (KaFa_1500_v10_l3).

KaFa 1500 v10 + an initial SEARCH phase. On level3 the gates and obstacles are randomised across
the whole arena and their true positions are only revealed once the drone passes within its
(horizontal) sensor range, so the drone cannot plan a race line from the start. This controller
adds a search phase that sweeps the arena to discover the gates, then flies v10's normal race.

    TAKEOFF -> SEARCH -> NAVIGATE

* TAKEOFF  : v10's vertical climb (inherited).
* SEARCH   : fly an expanding circular sweep (Archimedean spiral) over the arena at a safe
             height, driven by the SAME v10 time-optimal MPCC used for racing -- it is simply
             handed the spiral as its path instead of the race path. Detection is horizontal, so
             the sweep height is free; the spiral radius grows each loop until it has passed
             within sensor range of every gate. SEARCH ends once ALL gates have been detected
             (obstacles need NOT all be found -- they are picked up en route and re-planned around).
* NAVIGATE : v10's time-optimal MPCC through the discovered gates. Only sensor-confirmed
             obstacles are ever fed to the planner; undiscovered ones (reported at the origin
             placeholder) are pushed to a far sentinel so they cannot corrupt the path, and snap
             to their true position -> trigger a re-plan the moment they are sensed.

Detected gates (green crosses) and obstacles (red crosses) are drawn in the sim, along with the
active path. The search knobs live in KaFa_v10_l3.cockpit; the sweep path is KaFa_v10_l3.search.

This is a thin subclass of KaFa1500V10: it reuses v10's takeoff, planner, ArcPath, and MPCC, and
only inserts the SEARCH phase and the obstacle masking. REQUIRES the acados environment (run
under ``pixi run``). Inspired by the level-3 structure of KaFa_1500_v7 (spiral-as-virtual-gates),
re-expressed on v10's MPCC.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line

from lsy_drone_racing.control.kafa1500_v6.attitude import _vector_to_attitude
from lsy_drone_racing.control.kafa1500_v6.state import parse_observation
from lsy_drone_racing.control.KaFa_1500_v10 import KaFa1500V10
from lsy_drone_racing.control.KaFa_v10.arc_path import ArcPath
from lsy_drone_racing.control.KaFa_v10.mpcc import MPCC
from lsy_drone_racing.control.KaFa_v10_l3.cockpit import (
    NAV_A_LAT_MAX,
    NAV_A_Z_MIN,
    NAV_V_THETA_MAX,
    SearchSettings,
)
from lsy_drone_racing.control.KaFa_v10_l3.search import build_search_curve

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v6.state import DroneObservation

_OBS_FAR = np.array([50.0, 50.0, 50.0])  # sentinel for undiscovered obstacles (out of the arena)


class KaFa1500V10L3(KaFa1500V10):
    """v10 with a SEARCH phase that sweeps the arena to discover the gates before racing."""

    _MODE_SEARCH = "SEARCH"
    _MODE_BRAKE = "BRAKE"
    _BRAKE_V = 0.5      # m/s, race starts once the drone has slowed below this
    _BRAKE_T_MAX = 1.2  # s, brake at most this long, then race regardless

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build v10, then add the search-phase state."""
        super().__init__(obs, info, config)
        # Gentler NAVIGATE budget than v10's level2 race (see cockpit): lower the ArcPath /
        # progress-rate caps the MPCC is fed...
        self._v_theta_max = NAV_V_THETA_MAX
        self._a_lat_max = NAV_A_LAT_MAX
        # ...and rebuild the MPCC with more descent authority (lower a_z_min) so it can dive to the
        # first gate from the sweep height. The same MPCC is reused for SEARCH (level flight, so
        # a_z_min never binds there).
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(replace(self._settings.mpcc, a_z_min=NAV_A_Z_MIN), a_max)
        self._search = SearchSettings()
        self._search_path: ArcPath | None = None
        self._search_curve = None
        self._search_t_total = 0.0
        self._search_start_tick = 0
        self._brake_start_tick = 0
        # Discovery + render state (written by compute_control, read by render_callback).
        self._obs_visited = np.asarray(obs["obstacles_visited"], dtype=bool)
        self._dbg_gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        self._dbg_gate_known = np.asarray(obs["gates_visited"], dtype=bool)
        self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)
        self._dbg_target = 0

    def reset(self) -> None:
        """Reset v10 state plus the search phase."""
        super().reset()  # sets mode back to TAKEOFF, resets mpcc/planner/takeoff
        self._search_path = None
        self._search_curve = None
        self._search_t_total = 0.0

    # ── Main loop ─────────────────────────────────────────────────────────────
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """TAKEOFF -> SEARCH (until all gates seen) -> NAVIGATE, all on the v10 MPCC."""
        frame = parse_observation(obs)
        self._last_target = frame.target_gate
        gates_visited = np.asarray(obs["gates_visited"], dtype=bool)
        self._obs_visited = np.asarray(obs["obstacles_visited"], dtype=bool)
        self._record_render_state(frame, gates_visited)

        if self._tick / self._freq >= self._settings.runtime.timeout_s or frame.target_gate == -1:
            self._finished = True
            return self._last_action.copy()

        # ── TAKEOFF: v10 vertical climb ───────────────────────────────────────
        if self._mode == self._MODE_TAKEOFF:
            if not self._takeoff.is_complete(frame, self._tick, self._dt):
                action = self._takeoff.action(
                    frame, self._feedback, self._tick, self._dt, self._mass, self._gravity
                )
                self._last_action = action
                return action.copy()
            self._mpcc.reset()
            if gates_visited.all():  # all gates already known -> skip the sweep
                self._begin_navigate()
            else:
                self._begin_search(frame)

        # ── SEARCH: sweep the arena until every gate has been detected ─────────
        if self._mode == self._MODE_SEARCH:
            if gates_visited.all():
                self._brake_start_tick = self._tick
                self._mode = self._MODE_BRAKE
            else:
                action = self._search_action(frame)
                self._last_action = action
                return action.copy()

        # ── BRAKE: shed the sweep's tangential momentum before racing ──────────
        # The drone finds the last gate mid-sweep moving ~1.7 m/s; racing straight from there
        # overshoots the first gate. Freeze the MPCC reference (vth_cap=0) so the SAME MPCC pulls
        # the drone to a near-stop, then hand off to the race from a clean, slow state.
        if self._mode == self._MODE_BRAKE:
            elapsed = (self._tick - self._brake_start_tick) * self._dt
            if float(np.linalg.norm(frame.vel)) < self._BRAKE_V or elapsed > self._BRAKE_T_MAX:
                self._begin_navigate()
            else:
                self._s = self._search_path.project(frame.pos, self._s)
                accel = self._mpcc.solve(frame.pos, frame.vel, self._s, 0.0)
                thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
                action = _vector_to_attitude(thrust_vector, frame.quat, self._command)
                self._last_action = action
                return action.copy()

        # ── NAVIGATE: v10 race, planner fed only sensor-confirmed obstacles ────
        action = self._track_action(frame)
        self._last_action = action
        return action.copy()

    # ── Phase transitions ─────────────────────────────────────────────────────
    def _begin_search(self, frame: DroneObservation) -> None:
        """Build the spiral sweep path and load it into the (shared) v10 MPCC."""
        self._search_curve, self._search_t_total = build_search_curve(frame.pos[:2], self._search)
        self._search_path = ArcPath(
            self._search_curve, self._search_t_total, self._search.speed, self._search.a_lat,
            self._search.v_min,
        )
        self._mpcc.set_path(self._search_path)
        self._s = self._search_path.project(frame.pos, 0.0)
        self._search_start_tick = self._tick
        self._mode = self._MODE_SEARCH

    def _begin_navigate(self) -> None:
        """Hand the (shared) MPCC back to v10's race flow (it rebuilds the race path)."""
        self._references.reset()
        self._mpcc.reset()
        self._path = None  # force v10._track_action to rebuild the race ArcPath
        self._s = 0.0
        self._nav_start_tick = self._tick
        self._mode = self._MODE_NAVIGATE

    # ── SEARCH action (same MPCC as NAVIGATE, fed the spiral path) ─────────────
    def _search_action(self, frame: DroneObservation) -> NDArray[np.floating]:
        """Fly the spiral with the v10 MPCC at the (ramped) search speed; loop if exhausted."""
        self._s = self._search_path.project(frame.pos, self._s)
        if self._s >= self._search_path.total - 0.05:  # swept once without finding all -> re-sweep
            self._s = 0.0
            self._mpcc.reset()
        elapsed = (self._tick - self._search_start_tick) * self._dt
        ramp = min(1.0, self._search.ramp_start + (1.0 - self._search.ramp_start) * elapsed
                   / self._search.ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._search.speed * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)

    # ── NAVIGATE: mask undiscovered obstacles before v10 plans ─────────────────
    def _track_action(self, frame: DroneObservation) -> NDArray[np.floating]:
        """v10's race tracking, but the planner only ever sees sensor-confirmed obstacles."""
        return super()._track_action(self._mask_obstacles(frame))

    def _mask_obstacles(self, frame: DroneObservation) -> DroneObservation:
        """Push undiscovered obstacles to a far sentinel so they can't corrupt the race path.

        The array shape is preserved (the planner's re-plan delta check compares like-for-like),
        so when an obstacle is finally sensed it jumps from the sentinel to its true position and
        the large delta triggers a re-plan around it.
        """
        masked = np.asarray(frame.obstacles_pos, dtype=np.float64).copy()
        if not self._obs_visited.all():
            masked[~self._obs_visited] = _OBS_FAR
        return replace(frame, obstacles_pos=masked)

    # ── Rendering: detected gates / obstacles + the active path ────────────────
    def _record_render_state(self, frame: DroneObservation, gates_visited: NDArray) -> None:
        """Cache what render_callback draws (it runs in the sim, not this control step)."""
        self._dbg_gate_pos = np.asarray(frame.gate_pos, dtype=np.float64)
        self._dbg_gate_known = np.asarray(gates_visited, dtype=bool)
        self._dbg_target = int(frame.target_gate)
        obs_pos = np.asarray(frame.obstacles_pos, dtype=np.float64)
        self._dbg_obs_pos = obs_pos[self._obs_visited] if self._obs_visited.any() else np.empty(
            (0, 3)
        )

    def render_callback(self, sim: Sim) -> None:
        """Draw the active path (green) plus crosses at detected gates and obstacles."""
        # Active path: the spiral during SEARCH, the race spline during NAVIGATE.
        if self._mode in (self._MODE_SEARCH, self._MODE_BRAKE) and self._search_curve is not None:
            samples = self._search_curve(np.linspace(0.0, self._search_t_total, 120))
            draw_line(sim, np.asarray(samples, dtype=np.float32), rgba=(0.1, 0.6, 1.0, 1.0))
        else:
            plan = self._references.plan
            if plan is not None:
                samples = plan.curve(np.linspace(0.0, plan.t_total, 100))
                draw_line(sim, np.asarray(samples, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))

        def _cross(pos: NDArray, rgba: tuple, arm: float = 0.12) -> None:
            p = np.asarray(pos, dtype=np.float32)
            for axis in range(3):
                seg = np.tile(p, (2, 1))
                seg[0, axis] -= arm
                seg[1, axis] += arm
                draw_line(sim, seg, rgba=rgba)

        # Detected gates: bright green for the current target, dim green for the rest.
        for i, gp in enumerate(self._dbg_gate_pos):
            if i < len(self._dbg_gate_known) and self._dbg_gate_known[i]:
                _cross(gp, (0.0, 1.0, 0.0, 1.0) if i == self._dbg_target else (0.0, 0.55, 0.0, 1.0))
        # Detected obstacles: red crosses at the sensed positions.
        for op in self._dbg_obs_pos:
            _cross(op, (1.0, 0.0, 0.0, 1.0))

    def diagnostic(self) -> dict[str, object]:
        """Status summary for logging."""
        return {
            "controller_phase": "FINISHED" if self._finished else self._mode,
            "active_target_gate": self._last_target,
            "gates_found": int(np.count_nonzero(self._dbg_gate_known)),
            "obstacles_found": int(np.count_nonzero(self._obs_visited)),
            "controller_time": min(self._tick / self._freq, self._settings.runtime.timeout_s),
        }
