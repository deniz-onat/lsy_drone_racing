"""v11.1 path view: the v11 tunnel tables plus the v10.6 plan's parity speed caps.

The smoothed reference arrives with parity caps (KaFa_v10_6.trajectory.SmoothedPlan): per-gate
reveal-window caps and per-passage obstacle arc-interval caps, both copied from the unsmoothed
plan's profile. Under v11 the all-gate reveal caps (2.5 m/s) usually bind tighter than the
gate parity caps; the obstacle interval caps still matter -- the de-paced tunnel flies the
straightened passages faster than v10.5 ever did, which is exactly where the v10.6 ledger
measured the smoothed route's risk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.KaFa_v11.arc_path import TunnelArcPath, _limit_longitudinal

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CappedTunnelArcPath(TunnelArcPath):
    """v11's TunnelArcPath with the smoothed plan's parity caps folded into the profile."""

    def __init__(
        self,
        *args,  # noqa: ANN002 -- forwarded verbatim to TunnelArcPath
        gate_window_caps: NDArray[np.float64] | None = None,
        window_pre: float = 0.7,
        obstacle_caps: NDArray[np.float64] | None = None,
        **kwargs,  # noqa: ANN003
    ):
        """Build the v11 tables, then apply the plan's parity caps (np.inf = inactive)."""
        super().__init__(*args, **kwargs)
        v = self._vcurv.copy()
        bound = False
        if gate_window_caps is not None:
            for s_g, cap in zip(self._gate_arcs, np.asarray(gate_window_caps, dtype=np.float64)):
                if np.isfinite(cap):
                    mask = (self._s >= s_g - float(window_pre)) & (self._s <= s_g)
                    v[mask] = np.minimum(v[mask], cap)
                    bound = True
        if obstacle_caps is not None:
            for s_lo, s_hi, cap in np.asarray(obstacle_caps, dtype=np.float64).reshape(-1, 3):
                if np.isfinite(cap):
                    mask = (self._s >= s_lo) & (self._s <= s_hi)
                    v[mask] = np.minimum(v[mask], cap)
                    bound = True
        if bound:
            self._vcurv = _limit_longitudinal(v, self._s, float(self._a_lat_max_v11))
