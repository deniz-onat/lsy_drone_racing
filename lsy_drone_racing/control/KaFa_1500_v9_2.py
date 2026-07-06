"""MPCC drone racing controller for known tracks (KaFa_1500_v9.2).

v9.2 is v9.1 with one change: the MPCC reference no longer recedes at a constant rate. v9/v9.1
sample the look-ahead at a fixed arc spacing (V_REF * step_dt), so V_REF has to be capped low
enough to survive the tight gate turns, which leaves the straights slow. v9.2 replaces that
constant spacing with a curvature-aware speed profile (KaFa_v9_1.speed_profile): a
friction-circle parameterisation of the plan that slows the reference into corners and speeds
it up on the straights. The straights therefore run toward V_CAP (above v9.1's V_REF) while
the sharp gate-2/3 reversal self-brakes.

Everything else is inherited from v9.1: the same MPCC (its v_max/tilt/thrust limits stay the
safety net), the same stall governor, planner, and takeoff. Only how the reference is sampled
ahead of the current progress differs, so this is a thin subclass of KaFa1500V91. The profile
knobs live in KaFa_v9_2/cockpit.py.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.kafa1500_v6.attitude import _vector_to_attitude
from lsy_drone_racing.control.KaFa_1500_v9_1 import KaFa1500V91
from lsy_drone_racing.control.KaFa_v9_1.mpcc import MPCC, sample_path
from lsy_drone_racing.control.KaFa_v9_1.speed_profile import SpeedProfile
from lsy_drone_racing.control.KaFa_v9_2.settings import SpeedProfileSettings

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v6.state import DroneObservation


class KaFa1500V92(KaFa1500V91):
    """v9.1 flown with a curvature-aware reference (slow into turns, fast on straights)."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build v9.1, then attach the curvature speed-profile settings."""
        super().__init__(obs, info, config)
        self._prof_cfg = SpeedProfileSettings()
        # Rebuild the MPCC with v9.2's raised v_max so the tracker has headroom to chase the
        # faster profiled reference (v9.1's v_max would clip it back at the profile cap).
        mpcc_cfg = replace(self._settings.mpcc, v_max=self._prof_cfg.v_max)
        self._settings = replace(self._settings, mpcc=mpcc_cfg)
        self._mpcc = MPCC(mpcc_cfg, self._command.thrust_max / self._mass)
        # Gentler takeoff->nav ramp than v9.1's (the faster profile needs more easing onto the
        # first gate); these override the values v9.1 set from its own mpcc settings.
        self._ramp_start, self._ramp_s = self._prof_cfg.ramp_start, self._prof_cfg.ramp_s
        self._horizon = mpcc_cfg.horizon
        # The look-ahead is spaced over the MPCC's PREDICTION step (step_dt, ~0.05 s), not the
        # faster env step (self._dt = 1/freq), matching v9.1's self._arc = k * v_ref * step_dt.
        self._step_dt = mpcc_cfg.step_dt
        self._profile: SpeedProfile | None = None

    def reset(self) -> None:
        """Reset v9.1 state plus the cached speed profile."""
        super().reset()
        self._profile = None

    def _track_action(self, frame: DroneObservation) -> NDArray[np.floating]:
        """Same as v9.1 but the look-ahead is spaced by the curvature speed profile."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt or self._profile is None:  # new plan -> restart progress and rebuild profile
            self._progress_t = 0.0
            self._stall_ticks = 0
            self._profile = SpeedProfile(
                plan.curve, plan.t_total, self._prof_cfg.v_cap, self._prof_cfg.a_lat_max,
                self._prof_cfg.v_min,
            )
        proj_t = self._project(plan, frame.pos)
        self._progress_t = self._govern(proj_t, frame.vel, plan.t_total)
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        # Curvature-profiled look-ahead: arc offsets ahead of the current progress, spaced by
        # the local profiled speed (the ramp eases the look-ahead out after takeoff, as in v9.1).
        s0 = self._profile.arc_at_time(self._progress_t)
        offsets = self._profile.arc_offsets(s0, self._horizon, self._step_dt) * ramp
        ref_p, ref_t = sample_path(plan.curve, self._progress_t, plan.t_total, offsets)
        accel = self._mpcc.solve(frame.pos, frame.vel, ref_p, ref_t)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
