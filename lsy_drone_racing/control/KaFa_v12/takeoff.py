"""Consolidated takeoff-phase chain for the KaFa_v12 controller.

This module merges the single-inheritance takeoff chain into one file, preserving the
original base->derived method resolution order:

* ``_TakeoffPhaseV8`` -- the KaFa_v8 vertical takeoff phase (originally
  ``KaFa_v8.takeoff.TakeoffPhase``): a clean rest-to-altitude vertical climb (holding the
  start x/y), tracked with the shared attitude_action machinery, with the climb duration
  derived from a target climb speed.
* ``TakeoffPhase`` -- the KaFa_v10_4 mini-takeoff (originally
  ``KaFa_v10_4.takeoff.TakeoffPhase``) which subclasses the v8 phase and replaces the
  duration rule with a fixed (short) spline duration. This is the public export.

Both class bodies and the module-level constant are copied verbatim from their source
modules; only the import statements and the intermediate class name/base reference have
been rewritten for the consolidated layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.KaFa_v12.attitude import attitude_action

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.KaFa_v12.feedback import CascadedPid
    from lsy_drone_racing.control.KaFa_v12.observation import DroneObservation
    from lsy_drone_racing.control.KaFa_v12.settings import (
        ControllerSettings,
        LaunchTakeoffSettings,
        TakeoffSettings,
    )

# Floor on the climb duration so a tiny start-to-target height can't give a near-zero
# time (and an enormous endpoint acceleration with it). 0.6 s keeps the worst-case short
# climb gentle.
_MIN_CLIMB_TIME = 0.6


class _TakeoffPhaseV8:
    """A self-contained vertical climb from the ground to a target altitude.

    The climb is a clamped cubic from the current position straight up to [x, y, alt] with
    a zero end-velocity boundary condition, so it settles to a hover at the top. It builds
    its plan lazily on the first action() call so it captures the true post-reset start
    pose, and exposes is_complete() for the controller's mode handoff.
    """

    def __init__(self, settings: ControllerSettings, takeoff: TakeoffSettings):
        """Store the controller/takeoff settings; the plan is built lazily (see action)."""
        self._settings = settings
        self._alt = float(takeoff.alt)
        self._climb_speed = max(float(takeoff.climb_speed), 1e-3)
        self._z_tol = float(takeoff.z_tol)
        self._time_margin = float(takeoff.time_margin)
        self._curve: CubicSpline | None = None
        self._t_total = 0.0
        self._start_tick = 0

    def reset(self) -> None:
        """Forget the cached climb plan so the next episode rebuilds from its start pose."""
        self._curve = None
        self._t_total = 0.0
        self._start_tick = 0

    def is_complete(self, frame: DroneObservation, tick: int, dt: float) -> bool:
        """Return True once the drone has reached the target altitude (or the climb overran).

        Safe to call before the plan exists: it then decides purely from the measured
        altitude, so an already-airborne start skips takeoff entirely.
        """
        reached = float(frame.pos[2]) >= self._alt - self._z_tol
        if self._curve is None:
            return reached
        overran = (tick - self._start_tick) * dt >= self._t_total + self._time_margin
        return reached or overran

    def action(
        self,
        frame: DroneObservation,
        feedback: CascadedPid,
        tick: int,
        dt: float,
        mass: float,
        gravity: float,
    ) -> NDArray[np.float32]:
        """Return the [roll, pitch, yaw, thrust] action tracking the vertical climb."""
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
        """Build the rest-to-altitude vertical climb spline from the current pose."""
        start = np.asarray(frame.pos, dtype=np.float64).copy()
        start_vel = np.asarray(frame.vel, dtype=np.float64).copy()
        # Near the floor and nearly stationary, a spurious downward start velocity would
        # clamp the spline to a downward slope and drive it below z=0. Force the vertical
        # boundary condition non-negative so the climb opens upward.
        if float(np.linalg.norm(start_vel)) < 0.3:
            start_vel[2] = max(float(start_vel[2]), 0.0)
        target = np.array([start[0], start[1], self._alt])  # hold x/y, climb in z
        height = max(self._alt - float(start[2]), 0.0)
        # Duration comes from the target climb speed, not the cruise speed, so peak vertical
        # accel (~6 * height / t_climb^2) stays gentle and independent of race speed.
        t_climb = max(height / self._climb_speed, _MIN_CLIMB_TIME)
        knot_times = np.array([0.0, t_climb])
        bc = ((1, start_vel), (1, np.zeros(3)))  # settle to a hover at the top
        self._curve = CubicSpline(knot_times, np.vstack([start, target]), bc_type=bc)
        self._t_total = float(t_climb)
        self._start_tick = tick


class TakeoffPhase(_TakeoffPhaseV8):
    """v8's XY-held climb with a fixed (short) spline duration instead of the 0.6 s floor."""

    def __init__(self, settings: ControllerSettings, takeoff: LaunchTakeoffSettings):
        """Store the v8 fields plus the fixed climb duration."""
        super().__init__(settings, takeoff)
        self._climb_time = float(takeoff.climb_time)

    def _build(self, frame: DroneObservation, tick: int) -> None:
        """Build the climb spline exactly as v8 does, but over the fixed duration."""
        start = np.asarray(frame.pos, dtype=np.float64).copy()
        start_vel = np.asarray(frame.vel, dtype=np.float64).copy()
        if float(np.linalg.norm(start_vel)) < 0.3:  # keep the spline opening upward (as v8)
            start_vel[2] = max(float(start_vel[2]), 0.0)
        target = np.array([start[0], start[1], self._alt])  # hold x/y, climb in z
        knot_times = np.array([0.0, self._climb_time])
        bc = ((1, start_vel), (1, np.zeros(3)))  # settle toward hover at the top
        self._curve = CubicSpline(knot_times, np.vstack([start, target]), bc_type=bc)
        self._t_total = self._climb_time
        self._start_tick = tick
