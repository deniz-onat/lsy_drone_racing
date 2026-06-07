"""Arc-length view of a plan spline for the v10 time-optimal MPCC.

The MPCC reasons in arc length (so the progress state is metric and parameterisation-free).
This helper turns the planner's time-parameterised ``CubicSpline`` into two things the MPCC
needs each replan/step:

* ``nodes`` -- the path resampled onto the MPCC's FIXED arc-length grid, flat-extended past
  the plan end, fed to the optimiser as the b-spline coefficient parameters.
* ``project`` -- a forward nearest-point search returning the drone's current arc length,
  used to anchor the progress state to the real drone every step (this is what makes the
  progress-as-a-state formulation immune to the v9 mid-track stall).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline

# Forward window (m) for the projection search: comfortably more than one step of motion, so
# the projection advances smoothly without ever jumping back to an earlier loop of the path.
_PROJ_WINDOW_M = 2.0


class ArcPath:
    """Arc-length resampling and projection for a single plan spline."""

    def __init__(self, curve: CubicSpline, t_total: float, s_grid: NDArray[np.float64]):
        """Densely sample the plan, build its arc-length table, and resample onto ``s_grid``."""
        taus = np.linspace(0.0, float(t_total), 600)
        pts = np.asarray(curve(taus), dtype=np.float64)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        self._pts = pts
        self._s = s
        self.total = float(s[-1])
        # Resample onto the fixed grid; np.clip flat-extends past the plan end so progress
        # saturates at the goal instead of extrapolating off the spline.
        sg = np.clip(np.asarray(s_grid, dtype=np.float64), 0.0, self.total)
        self.nodes = np.column_stack([np.interp(sg, s, pts[:, i]) for i in range(3)])

    def project(self, pos: NDArray[np.float64], s_min: float) -> float:
        """Return the arc length of the nearest path point within a forward window of s_min."""
        lo = int(np.searchsorted(self._s, s_min))
        hi = max(int(np.searchsorted(self._s, s_min + _PROJ_WINDOW_M)), lo + 1)
        window = self._pts[lo:hi]
        nearest = int(np.argmin(np.linalg.norm(window - pos, axis=1)))
        return float(self._s[lo + nearest])
