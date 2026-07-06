"""Gate-search sweep phase for the KaFa_1500_v13 controller (new in v13).

Level 3 fully randomizes the gate/obstacle layout, so the nominal positions the observation
reports out of sensor range are useless. Before it can navigate, the drone has to *find* the
gates. Detection is by XY-proximity: any gate whose XY the drone comes within ``sensor_range`` of
is revealed permanently (``gates_visited``). This phase flies a boustrophedon ("lawnmower") sweep
that brings the drone within ``sensor_range`` of every point of the gate placement region, at an
altitude above every obstacle and gate so the blind sweep can't hit anything it hasn't sensed yet.

The sweep is one cubic-spline reference tracked with the same ``attitude_action`` + cascaded-PID
machinery the takeoff climb uses -- no new tracking stack. It exits early the moment every gate is
visited; otherwise it runs to the end of the planned path (plus a short grace) and hands off to
NAVIGATE, which reveals any straggler gate en route via its existing replanning.

Run ``python -m lsy_drone_racing.control.KaFa_v13.search`` for the coverage/safety self-check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.KaFa_v13.attitude import attitude_action

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.KaFa_v13.feedback import CascadedPid
    from lsy_drone_racing.control.KaFa_v13.observation import DroneObservation
    from lsy_drone_racing.control.KaFa_v13.settings import ControllerSettings, SearchSettings


def build_sweep_path(
    start: NDArray[np.float64], search: SearchSettings
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(waypoints, knot_times)`` for the eased-climb + lawnmower sweep from ``start``.

    Pure geometry (no controller state) so the ``__main__`` self-check can verify the resulting
    spline covers the placement region and stays inside the safety box. Layout:

    * an eased vertical climb from ``start`` to ``search.alt`` (two knots), then a short hover
      dwell so the spline reaches altitude with ~zero vertical velocity (kills the cubic z-bow
      that would otherwise punch through the ceiling),
    * boustrophedon rows across ``x in [-x_span, x_span]`` at each ``search.rows`` height,
      alternating direction, ``search.dens`` knots per row so the spline hugs the straight line.
    """
    x0, y0, z0 = (float(v) for v in start)
    alt = search.alt
    wp: list[list[float]] = [
        [x0, y0, z0],
        [x0, y0, z0 + 0.6 * (alt - z0)],
        [x0, y0, alt],
        [x0, y0, alt],  # dwell endpoint (same point, dt=dwell_time) -> flat arrival
    ]
    seg_t: list[float] = [search.climb_time * 0.6, search.climb_time * 0.4, search.dwell_time]
    for i, y in enumerate(search.rows):
        xa, xb = (-search.x_span, search.x_span) if i % 2 == 0 else (search.x_span, -search.x_span)
        for x in np.linspace(xa, xb, search.dens):
            wp.append([float(x), float(y), alt])

    waypoints = np.asarray(wp, dtype=np.float64)
    # Times for the sweep segments (everything after the fixed climb/dwell) from distance/speed.
    dist = np.linalg.norm(np.diff(waypoints[3:], axis=0), axis=1)
    sweep_t = np.maximum(dist / max(search.speed, 1e-3), 1e-3)
    knot_times = np.concatenate([[0.0], np.cumsum(np.concatenate([seg_t, sweep_t]))])
    return waypoints, knot_times


class SearchPhase:
    """Fly the lawnmower sweep, tracked by the shared attitude machinery, until all gates found."""

    def __init__(self, settings: ControllerSettings, search: SearchSettings):
        """Store settings; the sweep spline builds lazily on the first action (true start pose)."""
        self._settings = settings
        self._search = search
        self._curve: CubicSpline | None = None
        self._t_total = 0.0
        self._start_tick = 0

    def reset(self) -> None:
        """Forget the cached sweep so the next episode rebuilds from its post-takeoff pose."""
        self._curve = None
        self._t_total = 0.0
        self._start_tick = 0

    def is_complete(
        self,
        gates_visited: NDArray[np.bool_],
        obstacles_visited: NDArray[np.bool_],
        tick: int,
        dt: float,
    ) -> bool:
        """True once every gate *and* obstacle is sensed, or the sweep path (plus grace) ran out.

        Requiring obstacles too matters: exiting the moment the last gate is found can leave an
        unsensed obstacle whose nominal position is wrong, and NAVIGATE would then fly blind into
        it. The sweep covers the obstacle region anyway, so this only costs the tail of the path.
        """
        all_gates = gates_visited.size > 0 and bool(np.all(gates_visited))
        all_obs = bool(np.all(obstacles_visited))  # empty -> True (nothing to find)
        if self._curve is None:
            return all_gates and all_obs
        overran = (tick - self._start_tick) * dt >= self._t_total + self._search.max_extra_time
        return (all_gates and all_obs) or overran

    def action(
        self,
        frame: DroneObservation,
        feedback: CascadedPid,
        tick: int,
        dt: float,
        mass: float,
        gravity: float,
    ) -> NDArray[np.float32]:
        """Return the [roll, pitch, yaw, thrust] action tracking the sweep reference."""
        if self._curve is None:
            self._build(frame, tick)
        t_eval = float(np.clip((tick - self._start_tick) * dt, 0.0, self._t_total))
        action, _ = attitude_action(
            self._curve,
            t_eval,
            frame.pos,
            frame.vel,
            frame.quat,
            feedback,
            dt,
            mass,
            gravity,
            self._settings.command,
        )
        return action

    def _build(self, frame: DroneObservation, tick: int) -> None:
        """Build the sweep spline from the current pose (clamped to a rest-to-rest boundary)."""
        start = np.asarray(frame.pos, dtype=np.float64)
        waypoints, knot_times = build_sweep_path(start, self._search)
        bc = ((1, np.zeros(3)), (1, np.zeros(3)))  # start/end at rest, like the takeoff climb
        self._curve = CubicSpline(knot_times, waypoints, bc_type=bc)
        self._t_total = float(knot_times[-1])
        self._start_tick = tick


def _self_check() -> None:
    """Assert the sweep spline covers the gate placement region and stays inside the safety box."""
    from lsy_drone_racing.control.KaFa_v13.settings import SearchSettings

    search = SearchSettings()
    sensor_range = 0.7  # env sensor_range; gates reveal within this XY distance
    # Placement region (border_margin=0.5 inside the [-3,3]x[-2,2] safety box) gates can occupy.
    gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 51), np.linspace(-1.5, 1.5, 31))
    region = np.column_stack([gx.ravel(), gy.ravel()])

    worst_cover = 0.0
    for start in [(1.0, 1.0, 0.42), (-2.5, -1.5, 0.42), (2.5, 1.5, 0.42), (0.0, 0.0, 0.42)]:
        wp, t = build_sweep_path(np.asarray(start), search)
        curve = CubicSpline(t, wp, bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
        pos = curve(np.linspace(0.0, t[-1], 8000))
        # Safety box: hard limits are +/-3 (x), +/-2 (y), 2.0 (z); require a margin.
        assert np.abs(pos[:, 0]).max() < 2.9, f"x overshoot {np.abs(pos[:,0]).max():.2f}"
        assert np.abs(pos[:, 1]).max() < 1.9, f"y overshoot {np.abs(pos[:,1]).max():.2f}"
        assert pos[:, 2].max() < 1.97, f"z ceiling breach {pos[:,2].max():.2f}"
        # Every placement point must be within sensor_range of the (perfectly-tracked) sweep path.
        gap = np.sqrt(((region[:, None, :] - pos[None, :, :2]) ** 2).sum(-1)).min(1).max()
        worst_cover = max(worst_cover, gap)
    assert worst_cover < sensor_range - 0.15, f"coverage gap {worst_cover:.2f} too tight"
    print(f"search self-check OK: worst coverage gap {worst_cover:.2f} m (< {sensor_range} sensor)")


if __name__ == "__main__":
    _self_check()
