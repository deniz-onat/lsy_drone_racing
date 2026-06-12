"""Gate-aware arc-length view for v10.6: v10.5's anchor/caps + the smoothing parity caps.

Subclasses v10.5's GateArcPath (reactive gate caps + the bounded-projection anchor, both
unchanged) and additionally applies the parity caps a smoothed plan carries (see
KaFa_v10_6.trajectory): per-gate reveal-window caps and per-obstacle passage caps, each set to
the UNSMOOTHED plan's speed there, followed by the usual backward/forward feasibility passes.
A v10.4-style plan (no caps) leaves the profile byte-identical to v10.5's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.KaFa_v10_5.arc_path import GateArcPath as _GateArcPath

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline


def _limit_longitudinal(
    v: NDArray[np.float64], s: NDArray[np.float64], a_max: float
) -> NDArray[np.float64]:
    """Backward (brake in time) and forward (accelerate out) passes over a speed profile."""
    ds = np.diff(s)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_max * ds[i]))
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_max * ds[i - 1]))
    return v


class GateArcPath(_GateArcPath):
    """v10.5's GateArcPath plus the smoothed plan's gate-window and obstacle-passage caps."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        v_cap: float,
        a_lat_max: float,
        v_min: float,
        gate_pos: NDArray[np.float64],
        w_base: float,
        w_gate: float,
        gate_sigma: float,
        gate_caps: NDArray[np.float64],
        react_v_pre: float,
        react_v_post: float,
        n: int = 600,
        *,
        window_caps: NDArray[np.float64] | None = None,
        window_pre: float = 0.7,
        obstacle_caps: NDArray[np.float64] | None = None,
    ):
        """Build the v10.5 tables, then cap where the plan's parity caps bind.

        ``window_caps`` has one entry per row of ``gate_pos`` (cap over [s_gate - window_pre,
        s_gate]); ``obstacle_caps`` is (n, 3) rows of (s_lo, s_hi, cap) arc intervals on this
        path's own curve (the plan's obstacle passages). np.inf entries (and None) are inactive.
        """
        super().__init__(
            curve,
            t_total,
            v_cap,
            a_lat_max,
            v_min,
            gate_pos,
            w_base,
            w_gate,
            gate_sigma,
            gate_caps,
            react_v_pre,
            react_v_post,
            n,
        )
        v = self._vcurv.copy()
        bound = False
        if window_caps is not None:
            for s_g, cap in zip(self._gate_arcs, np.asarray(window_caps, dtype=np.float64)):
                if cap < float(v_cap):
                    mask = (self._s >= s_g - float(window_pre)) & (self._s <= s_g)
                    v[mask] = np.minimum(v[mask], cap)
                    bound = True
        if obstacle_caps is not None:
            for s_lo, s_hi, cap in np.asarray(obstacle_caps, dtype=np.float64).reshape(-1, 3):
                if cap < float(v_cap):
                    mask = (self._s >= s_lo) & (self._s <= s_hi)
                    v[mask] = np.minimum(v[mask], cap)
                    bound = True
        if bound:
            self._vcurv = _limit_longitudinal(v, self._s, float(a_lat_max))
