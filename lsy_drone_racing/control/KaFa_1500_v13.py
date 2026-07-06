"""KaFa_1500_v13 — v12 tunnel-MPCC racing controller + a level-3 gate-search phase.

A standalone clone of KaFa_1500_v12 (the v11 tunnel-constrained, de-paced time-optimal MPCC flying
v10.6's guarded-smoothed reference) extended for level 3, where the whole gate/obstacle layout is
randomized and the nominal positions the observation reports out of sensor range are useless. v13
inserts a SEARCH phase that flies a lawnmower sweep over the arena to reveal the gates before
navigating. Everything it needs lives in ``KaFa_v13``; it imports nothing from any other version.

Phase machine:

    TAKEOFF  -> mini vertical climb (PID-tracked), holding XY inside the floor-touch carve-out
    SEARCH   -> lawnmower sweep above all obstacles (KaFa_v13.search), PID-tracked, until every
                gate has been sensed (or the sweep path runs out). New in v13; a no-op for levels
                where gates start visible near their nominal spots.
    NAVIGATE -> guarded-smoothed global spline (KaFa_v13.planner), flown by the tunnel MPCC
                (KaFa_v13.mpcc) over the capped tunnel path view (KaFa_v13.arc_path), with a
                fast launch ramp and v10.2's dynamics-aware predicted-progress anchor. Unchanged
                from v12: by the time it starts, the revealed gate positions are already in the
                observation, so the planner builds the correct spline exactly as it does on a
                mid-flight gate reveal.

The NAVIGATE flow is bit-identical to v12; only the takeoff->navigate handoff now routes through
SEARCH. REQUIRES the acados environment -- run under ``pixi run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.KaFa_v13.arc_path import CappedTunnelArcPath
from lsy_drone_racing.control.KaFa_v13.attitude import _vector_to_attitude
from lsy_drone_racing.control.KaFa_v13.feedback import CascadedPid
from lsy_drone_racing.control.KaFa_v13.mpcc import MPCC
from lsy_drone_racing.control.KaFa_v13.observation import parse_observation
from lsy_drone_racing.control.KaFa_v13.planner import ReferenceManager, gate_post_obstacles
from lsy_drone_racing.control.KaFa_v13.search import SearchPhase
from lsy_drone_racing.control.KaFa_v13.settings import ControllerSettings
from lsy_drone_racing.control.KaFa_v13.takeoff import TakeoffPhase

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.KaFa_v13.observation import DroneObservation


class KaFa1500V13(Controller):
    """v12 tunnel MPCC over the guarded-smoothed reference, with a level-3 gate-search sweep."""

    _MODE_TAKEOFF = "TAKEOFF"
    _MODE_SEARCH = "SEARCH"
    _MODE_NAVIGATE = "NAVIGATE"

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the planner, takeoff/search phases, and the v11 tunnel MPCC (the only solver)."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("KaFa_1500_v13 requires env.control_mode = 'attitude'.")
        self._settings = ControllerSettings()
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(params["mass"])
        self._gravity = self._settings.runtime.gravity
        self._command = self._settings.command
        self._feedback = CascadedPid(self._settings.feedback)

        # Cached MPCC-derived limits the NAVIGATE flow reads (from the resolved v11.1 settings).
        # Only the limits v11.1's tunnel flow actually uses are kept; the v10.x gate-weight and
        # reactive-cap caches are never read by the tunnel path view, so they are dropped.
        mpcc = self._settings.mpcc
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._proj_band = mpcc.proj_band_m  # v10.2 anchor band half-width

        # The surviving runtime objects: v10.6 guarded-smoothing planner + v10.4 mini-takeoff.
        self._references = ReferenceManager(
            self._settings.planner,
            self._settings.runtime.replan_gate_delta_m,
            self._settings.runtime.replan_obstacle_delta_m,
            self._v_theta_max,
            self._a_lat_max,
            self._v_min,
        )
        self._takeoff = TakeoffPhase(self._settings, self._settings.takeoff)
        self._search = SearchPhase(self._settings, self._settings.search)

        # The one MPCC actually flown: v11's tunnel-constrained OCP (codegen namespace kafa_v13).
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(self._settings.mpcc, a_max)

        self._tick = 0
        self._nav_start_tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._mode = self._MODE_TAKEOFF
        self._last_target = -1
        self._last_action = self._hover_action()

        # Arc-length path view + foot-point progress anchor (built on the first NAVIGATE tick).
        self._gate_nominal: np.ndarray | None = None  # gate poses before any reveal
        self._path: CappedTunnelArcPath | None = None
        self._s = 0.0

        # In-sim obstacle overlay state (render only).
        self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)
        self._r_obs = float(getattr(self._references._settings, "r_obs", 0.20))
        self._nominal_obs_pos = (
            np.asarray(obs.get("obstacles_pos", []), dtype=np.float64).reshape(-1, 3).copy()
        )

        self._reset_anchor_telemetry()

    def _reset_anchor_telemetry(self) -> None:
        """Clear the per-episode anchor diagnostics (max jump, jump samples, band-edge rate)."""
        self._anchor_prev_s: float | None = None
        self._anchor_jumps: list[float] = []  # |s_t - s_{t-1}| in a plan (rebuild steps excluded)
        self._band_edge_hits = 0   # project_near results pinned at the +/- band edge
        self._band_calls = 0       # project_near calls (i.e. steps with a live prediction)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return a [roll, pitch, yaw, thrust] attitude action for the current step."""
        frame = parse_observation(obs)
        self._last_target = frame.target_gate
        if self._tick / self._freq >= self._settings.runtime.timeout_s or frame.target_gate == -1:
            self._finished = True
            return self._last_action.copy()

        # -- TAKEOFF: vertical climb, PID-tracked --
        if self._mode == self._MODE_TAKEOFF:
            if not self._takeoff.is_complete(frame, self._tick, self._dt):
                action = self._takeoff.action(
                    frame, self._feedback, self._tick, self._dt, self._mass, self._gravity
                )
                self._last_action = action
                return action.copy()
            self._mode = self._MODE_SEARCH

        # -- SEARCH: lawnmower sweep above all obstacles, PID-tracked, until all gates sensed --
        if self._mode == self._MODE_SEARCH:
            if not self._search.is_complete(
                frame.gates_visited, frame.obstacles_visited, self._tick, self._dt
            ):
                action = self._search.action(
                    frame, self._feedback, self._tick, self._dt, self._mass, self._gravity
                )
                self._last_action = action
                return action.copy()
            self._mode = self._MODE_NAVIGATE
            self._references.reset()
            self._mpcc.reset()
            self._progress_t = 0.0
            self._nav_start_tick = self._tick

        # -- NAVIGATE: guarded-smoothed path, tunnel-MPCC-tracked --
        action = self._track_action(frame)
        self._last_action = action
        return action.copy()

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """v11's flow with the smoothed plan's parity caps folded into the path view."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        new_plan = rebuilt or self._path is None
        if new_plan:
            first = max(frame.target_gate, 0)
            posts = gate_post_obstacles(
                frame.gate_pos, frame.gate_quat, 0, self._settings.planner.gate_post_offset
            )
            path = CappedTunnelArcPath(
                plan.curve,
                plan.t_total,
                self._settings.mpcc,
                plan.gate_pos_snapshot[first:],
                frame.obstacles_pos,
                posts,
                gate_is_target_zero=first == 0,
                gate_window_caps=plan.gate_window_caps,
                window_pre=self._settings.planner.reveal_window_m,
                obstacle_caps=plan.obstacle_caps,
            )
            self._s = path.project(frame.pos, 0.0)
            if self._path is None:
                self._mpcc.set_path(path)
            else:
                self._mpcc.rebase(path, self._s)
            self._path = path
        th_pred = self._mpcc.predicted_progress()
        if th_pred is None:
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

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the controller clock and latch sensor-confirmed obstacle positions for render."""
        obs_pos = np.asarray(obs.get("obstacles_pos", []), dtype=np.float64).reshape(-1, 3)
        visited = np.asarray(obs.get("obstacles_visited", []), dtype=bool).reshape(-1)
        if visited.any() and len(visited) <= len(obs_pos):
            self._dbg_obs_pos = obs_pos[: len(visited)][visited].copy()
        self._tick += 1
        return self._finished

    def reset(self) -> None:
        """Reset per-episode controller state."""
        self._tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._mode = self._MODE_TAKEOFF
        self._last_target = -1
        self._feedback.reset()
        self._references.reset()
        self._takeoff.reset()
        self._search.reset()
        self._mpcc.reset()
        self._last_action = self._hover_action()
        self._path = None
        self._s = 0.0
        self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)
        self._gate_nominal = None
        self._reset_anchor_telemetry()

    def episode_callback(self) -> None:
        """Reset state after an episode completes."""
        self.reset()

    def episode_reset(self) -> None:
        """Reset state before the next episode starts."""
        self.reset()

    def render_callback(self, sim: Sim) -> None:
        """Draw the active path (green), planner/obstacle overlays, and the MPCC horizon (cyan)."""
        plan = self._references.plan
        if plan is not None and self._mode == self._MODE_NAVIGATE:
            samples = plan.curve(np.linspace(0.0, plan.t_total, 100))
            draw_line(sim, np.asarray(samples, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))

        def _cross(pos: NDArray[np.floating], rgba: tuple, arm: float) -> None:
            """Draw a 3-axis cross at pos via three two-point segments."""
            p = np.asarray(pos, dtype=np.float32)
            for axis in range(3):
                seg = np.repeat(p[None, :], 2, axis=0)
                seg[0, axis] -= arm
                seg[1, axis] += arm
                draw_line(sim, seg, rgba=rgba)

        # White crosses at the nominal (config) obstacle positions, before/independent of sensing.
        for op in self._nominal_obs_pos:
            _cross(op, rgba=(1.0, 1.0, 1.0, 0.6), arm=0.06)

        # Blue crosses at the generated spline's knot points (the planner's waypoint chain).
        if plan is not None and self._mode == self._MODE_NAVIGATE:
            for wp in np.asarray(plan.waypoints, dtype=np.float64).reshape(-1, 3):
                _cross(wp, rgba=(0.2, 0.4, 1.0, 1.0), arm=0.05)

        # Red crosses + orange keep-out rings at the detected (sensor-confirmed) obstacles.
        angles = np.linspace(0.0, 2.0 * np.pi, 32)
        unit_ring = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
        for op in self._dbg_obs_pos:
            _cross(op, rgba=(1.0, 0.0, 0.0, 1.0), arm=0.08)
            ring = (unit_ring * self._r_obs + np.asarray(op, dtype=np.float32)).astype(np.float32)
            draw_line(sim, ring, rgba=(1.0, 0.5, 0.0, 0.8))

        # The MPCC's predicted horizon (cyan): the solver's own intended trajectory.
        pred = self._mpcc.predicted_positions()
        if pred is not None and self._mode == self._MODE_NAVIGATE:
            draw_line(sim, pred.astype(np.float32), rgba=(0.0, 0.9, 1.0, 1.0))

    def diagnostic(self) -> dict[str, float | int | str | None]:
        """Return a short status summary for debugging and logging."""
        plan = self._references.plan
        return {
            "controller_phase": "FINISHED" if self._finished else self._mode,
            "active_target_gate": self._last_target,
            "controller_time": min(self._tick / self._freq, self._settings.runtime.timeout_s),
            "reference_end_time": None if plan is None else plan.t_total,
        }

    def anchor_telemetry(self) -> dict[str, float | int]:
        """Per-episode progress-anchor diagnostics (the v10.2-style fold-teleport signature)."""
        jumps = np.asarray(self._anchor_jumps, dtype=np.float64)
        edge_rate = (self._band_edge_hits / self._band_calls) if self._band_calls else 0.0
        return {
            "n_steps": int(jumps.size),
            "max_jump_m": float(jumps.max()) if jumps.size else 0.0,
            "p99_jump_m": float(np.percentile(jumps, 99)) if jumps.size else 0.0,
            "n_jumps_gt_1m": int((jumps > 1.0).sum()),
            "band_calls": int(self._band_calls),
            "band_edge_rate": edge_rate,
        }

    def _hover_action(self) -> NDArray[np.float32]:
        """Build a level hover action that holds altitude at startup and on finish."""
        thrust = float(
            np.clip(self._mass * self._gravity, self._command.thrust_min, self._command.thrust_max)
        )
        return np.array([0.0, 0.0, 0.0, thrust], dtype=np.float32)
