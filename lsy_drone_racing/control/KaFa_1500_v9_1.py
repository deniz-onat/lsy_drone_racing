"""MPCC drone racing controller for known tracks (KaFa_1500_v9.1).

v9.1 is v9 with the mid-track stall fixed. In v9 the MPCC reference was driven purely by a
forward-windowed nearest-point projection of the drone, so at sharp turns and loops the
drone could overshoot the path, the projection would stop advancing, the receding reference
would freeze about a horizon ahead, and the controller settled into a stable hover that it
never escaped (replans only happen on gate-pass). v9.1 keeps v9's planner, takeoff, and
MPCC and only changes how reference progress advances: a governor guarantees the reference
always creeps forward, bounded so it can't outrun a genuinely blocked drone, which destroys
that fixed point. It also uses the hardened MPCC solver in KaFa_v9_1.mpcc.

Only the controller's progress logic differs from v9, so this is a thin subclass of
KaFa1500V9. The governor knobs live in KaFa_v9_1/cockpit.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.kafa1500_v6.attitude import _vector_to_attitude
from lsy_drone_racing.control.KaFa_1500_v9 import KaFa1500V9
from lsy_drone_racing.control.KaFa_v9_1.mpcc import MPCC, sample_path
from lsy_drone_racing.control.KaFa_v9_1.settings import ControllerSettings

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v6.state import DroneObservation


class KaFa1500V91(KaFa1500V9):
    """v9 with a progress governor so the reference never stalls at sharp turns/loops."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build v9, then swap in the v9.1 settings, hardened MPCC, and governor state."""
        super().__init__(obs, info, config)
        # v9.1 settings add the progress governor; planner/takeoff/command are unchanged, so
        # the inherited ReferenceManager and TakeoffPhase are reused as-is.
        self._settings = ControllerSettings()
        self._prog = self._settings.progress
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(self._settings.mpcc, a_max)
        self._arc = np.arange(self._settings.mpcc.horizon + 1) * self._mpcc.ds
        self._stall_ticks = 0

    def reset(self) -> None:
        """Reset v9 state plus the governor's stall counter."""
        super().reset()
        self._stall_ticks = 0

    def _track_action(self, frame: DroneObservation) -> NDArray[np.floating]:
        """Same as v9 but the reference progress is run through the governor."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt:  # a new plan is re-parameterized from t=0, so restart progress tracking
            self._progress_t = 0.0
            self._stall_ticks = 0
        proj_t = self._project(plan, frame.pos)
        self._progress_t = self._govern(proj_t, frame.vel, plan.t_total)
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        ref_p, ref_t = sample_path(plan.curve, self._progress_t, plan.t_total, self._arc * ramp)
        accel = self._mpcc.solve(frame.pos, frame.vel, ref_p, ref_t)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)

    def _govern(self, proj_t: float, vel: NDArray[np.floating], t_total: float) -> float:
        """Advance reference progress so it always moves forward, bounded over a stuck drone.

        proj_t is the geometric nearest-point time (never behind the current progress). While
        the drone keeps moving, proj_t advances faster than the creep, so progress == proj_t
        and behaviour matches v9. When the projection freezes, the creep walks progress ahead
        until it leads proj_t by max_lead_t, which grows the lag error and pulls the drone out
        of the stall; the lead cap keeps a genuinely blocked drone from getting a runaway
        reference. A stall watchdog scales the creep and lead up when the drone stays slow.
        """
        prev = self._progress_t
        speed = float(np.linalg.norm(vel))
        self._stall_ticks = self._stall_ticks + 1 if speed < self._prog.v_stall else 0
        stalled = self._stall_ticks * self._dt >= self._prog.t_stall
        boost = self._prog.stall_boost if stalled else 1.0
        creep = self._prog.min_progress_rate * self._dt * boost
        lead = self._prog.max_lead_t * boost
        target = max(prev, proj_t)            # never go backward, catch up to the projection
        target = max(target, prev + creep)    # guaranteed forward creep (kills the stall)
        target = min(target, proj_t + lead)   # but stay bounded ahead of a blocked drone
        return float(min(target, t_total))
