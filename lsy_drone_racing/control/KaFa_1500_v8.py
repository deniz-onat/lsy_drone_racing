"""Global-planner drone racing controller for known tracks (KaFa_1500_v8).

v8 is built for the deployment case where we scan the whole course before flying, so the
planner gets the real gate and obstacle layout from t=0. It reuses pieces of v6 and v7:

- v6's global cubic-spline planner through all gates from t=0 (full look-ahead).
- v6's dedicated vertical takeoff phase, so lift-off is decoupled from cruise speed.
- v7's gate-post funnels (route the spline through each opening's centre) and canonical
  +x gate crossing (orient_gates_to_travel=False), which is the direction the env counts.

We drop v7's SEARCH/discovery code on purpose: it's there for unknown (Level-3) tracks
and only hurts when the layout is already known. The phase machine is just
TAKEOFF -> NAVIGATE. All tunables live in KaFa_v8/cockpit.py and reach the modules
through KaFa_v8/settings.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.kafa1500_v6.attitude import attitude_action
from lsy_drone_racing.control.kafa1500_v6.feedback import CascadedPid
from lsy_drone_racing.control.kafa1500_v6.state import parse_observation
from lsy_drone_racing.control.KaFa_v8.settings import ControllerSettings
from lsy_drone_racing.control.KaFa_v8.takeoff import TakeoffPhase
from lsy_drone_racing.control.KaFa_v8.trajectory import ReferenceManager

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v6.state import DroneObservation
    from lsy_drone_racing.control.KaFa_v8.trajectory import ReferencePlan


class KaFa1500V8(Controller):
    """Flies a fully-known gate layout with a global, gate-funnelled spline reference."""

    _MODE_TAKEOFF = "TAKEOFF"
    _MODE_NAVIGATE = "NAVIGATE"

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize timing, drone parameters, feedback, and the flight phases."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("KaFa_1500_v8 requires env.control_mode = 'attitude'.")
        # Every v8 tunable comes from KaFa_v8/cockpit.py via these settings.
        self._settings = ControllerSettings()
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(params["mass"])
        self._feedback = CascadedPid(self._settings.feedback)
        self._references = ReferenceManager(
            self._settings.planner,
            self._settings.runtime.replan_gate_delta_m,
            self._settings.runtime.replan_obstacle_delta_m,
        )
        # Dedicated vertical lift-off, so takeoff aggressiveness is independent of cruise speed.
        self._takeoff = TakeoffPhase(self._settings, self._settings.takeoff)
        self._tick = 0
        self._plan_start_tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._mode = self._MODE_TAKEOFF
        self._last_time = 0.0
        self._last_target = -1
        self._last_action = self._hover_action()
        # Debug viz state: written in compute_control, read back in render_callback.
        self._dbg_gate_pos: NDArray = np.empty((0, 3), dtype=np.float64)
        self._dbg_known_mask: NDArray = np.zeros(0, dtype=bool)
        self._dbg_target_gate: int = -1
        self._dbg_obs_pos: NDArray = np.empty((0, 3), dtype=np.float64)
        self._dbg_wp_pos: NDArray = np.empty((0, 3), dtype=np.float64)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return a [roll, pitch, yaw, thrust] attitude action for the current step."""
        frame = parse_observation(obs)
        now = min(self._tick / self._freq, self._settings.runtime.timeout_s)
        self._last_time = now
        self._last_target = frame.target_gate

        if now >= self._settings.runtime.timeout_s:
            self._finished = True
            return self._last_action.copy()

        # All gates passed: hold the last action and finish.
        if frame.target_gate == -1:
            self._finished = True
            self._capture_debug(obs, frame, plan=None)
            return self._last_action.copy()

        # TAKEOFF: clean vertical climb, then hand off to gate tracking.
        if self._mode == self._MODE_TAKEOFF:
            if not self._takeoff.is_complete(frame, self._tick, self._dt):
                action = self._takeoff.action(
                    frame, self._feedback, self._tick, self._dt,
                    self._mass, self._settings.runtime.gravity,
                )
                self._last_action = action
                self._capture_debug(obs, frame, plan=None)
                return action.copy()
            # Hit target altitude: switch to navigation and build the gate plan from hover.
            self._mode = self._MODE_NAVIGATE
            self._references.reset()
            self._plan_start_tick = self._tick
            self._progress_t = 0.0

        # NAVIGATE: global, funnelled gate-spline reference.
        action, plan = self._track_action(frame)
        self._last_action = action
        self._capture_debug(obs, frame, plan=plan)
        return action.copy()

    def _track_action(
        self, frame: DroneObservation
    ) -> tuple[NDArray[np.floating], ReferencePlan]:
        """Track the global gate spline; (re)plan on target/pose change and return the plan."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt:
            self._plan_start_tick = self._tick
            self._progress_t = 0.0
        clock_t = (self._tick - self._plan_start_tick) * self._dt
        self._progress_t = self._project(plan, frame.pos)
        t_eval = float(
            np.clip(
                min(clock_t, self._progress_t + self._settings.runtime.lookahead_s),
                0.0,
                plan.t_total,
            )
        )
        action, _ = attitude_action(
            plan.curve,
            t_eval,
            frame.pos,
            frame.vel,
            frame.quat,
            self._feedback,
            self._dt,
            self._mass,
            self._settings.runtime.gravity,
            self._settings.command,
        )
        return action, plan

    def _capture_debug(
        self, obs: dict, frame: DroneObservation, plan: ReferencePlan | None
    ) -> None:
        """Capture marker state for render_callback (gates, obstacles, planned waypoints)."""
        self._dbg_gate_pos = np.asarray(frame.gate_pos, dtype=np.float64)
        gates_visited = np.asarray(obs.get("gates_visited", []), dtype=bool).reshape(-1)
        known_mask = np.zeros(len(frame.gate_pos), dtype=bool)
        known_mask[: len(gates_visited)] = gates_visited[: len(known_mask)]
        self._dbg_known_mask = known_mask
        self._dbg_target_gate = int(frame.target_gate)
        obs_visited = np.asarray(obs.get("obstacles_visited", []), dtype=bool).reshape(-1)
        if obs_visited.any() and len(obs_visited) <= len(frame.obstacles_pos):
            self._dbg_obs_pos = np.asarray(
                frame.obstacles_pos[: len(obs_visited)][obs_visited], dtype=np.float64
            )
        else:
            self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)
        # Planned waypoints: points sampled along the active reference spline. There are
        # none during the vertical takeoff climb.
        if plan is not None:
            n_samples = min(10, plan.t_total * 5)
            t_pts = np.linspace(0.0, plan.t_total, max(2, int(n_samples)))
            self._dbg_wp_pos = np.asarray(plan.curve(t_pts), dtype=np.float64)
        else:
            self._dbg_wp_pos = np.empty((0, 3), dtype=np.float64)

    def _project(self, plan: ReferencePlan, pos: NDArray[np.floating]) -> float:
        """Find the spline time of the closest point ahead of the current progress."""
        window = self._settings.runtime.projection_window_s
        upper = min(self._progress_t + window, plan.t_total)
        sample_t = np.linspace(self._progress_t, upper, 40)
        distances = np.linalg.norm(np.asarray(plan.curve(sample_t)) - pos, axis=1)
        return float(sample_t[int(np.argmin(distances))])

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the controller clock after an environment step."""
        self._tick += 1
        return self._finished

    def reset(self) -> None:
        """Reset per-episode controller state."""
        self._tick = 0
        self._plan_start_tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._mode = self._MODE_TAKEOFF
        self._last_time = 0.0
        self._last_target = -1
        self._feedback.reset()
        self._references.reset()
        self._takeoff.reset()
        self._last_action = self._hover_action()
        self._dbg_gate_pos = np.empty((0, 3), dtype=np.float64)
        self._dbg_known_mask = np.zeros(0, dtype=bool)
        self._dbg_target_gate = -1
        self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)
        self._dbg_wp_pos = np.empty((0, 3), dtype=np.float64)

    def episode_callback(self) -> None:
        """Reset state after an episode completes."""
        self.reset()

    def episode_reset(self) -> None:
        """Reset state before the next episode starts."""
        self.reset()

    def render_callback(self, sim: Sim) -> None:
        """Draw the active reference spline and debug markers."""
        plan = self._references.plan
        if plan is not None and self._mode == self._MODE_NAVIGATE:
            samples = plan.curve(np.linspace(0.0, plan.t_total, 100))
            draw_line(sim, np.asarray(samples, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))

        # --- debug markers ---
        arm = 0.08  # half-length of each cross arm, in metres

        def _cross(pos: NDArray, rgba: tuple) -> None:
            """Draw a 3-axis cross at pos using three two-point draw_line calls."""
            p = np.asarray(pos, dtype=np.float32)
            for axis in range(3):
                seg = np.zeros((2, 3), dtype=np.float32)
                seg[0] = p
                seg[1] = p
                seg[0, axis] -= arm
                seg[1, axis] += arm
                draw_line(sim, seg, rgba=rgba)

        # Green crosses at detected gate centres, bright green for the current target.
        for i, gp in enumerate(self._dbg_gate_pos):
            if not self._dbg_known_mask[i]:
                continue
            if i == self._dbg_target_gate:
                _cross(gp, rgba=(0.0, 1.0, 0.0, 1.0))   # bright green: target gate
            else:
                _cross(gp, rgba=(0.0, 0.55, 0.0, 1.0))  # dim green: other known gates

        # Red crosses at sensor-confirmed obstacle positions.
        for op in self._dbg_obs_pos:
            _cross(op, rgba=(1.0, 0.0, 0.0, 1.0))

        # Blue crosses at the current planned waypoints.
        for wp in self._dbg_wp_pos:
            _cross(wp, rgba=(0.2, 0.4, 1.0, 1.0))

    def diagnostic(self) -> dict[str, float | int | str | None]:
        """Return a short status summary for debugging and logging."""
        plan = self._references.plan
        return {
            "controller_phase": "FINISHED" if self._finished else self._mode,
            "active_target_gate": self._last_target,
            "controller_time": self._last_time,
            "reference_end_time": None if plan is None else plan.t_total,
        }

    def _hover_action(self) -> NDArray[np.float32]:
        """Build a level hover action that holds altitude at startup and on finish."""
        thrust = float(
            np.clip(
                self._mass * self._settings.runtime.gravity,
                self._settings.command.thrust_min,
                self._settings.command.thrust_max,
            )
        )
        return np.array([0.0, 0.0, 0.0, thrust], dtype=np.float32)
