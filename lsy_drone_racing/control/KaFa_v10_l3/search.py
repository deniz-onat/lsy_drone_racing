"""Build the SEARCH sweep path (expanding circle) for KaFa_1500_v10_l3.

Returns a scipy CubicSpline over the arena that the v10 MPCC flies exactly like a race path
(via KaFa_v10.arc_path.ArcPath). The sweep is an Archimedean spiral r(theta)=a*theta from R0 to
R_MAX, clipped to the arena, starting at the drone's current XY so the hand-off from takeoff is
smooth. One revolution's radius grows by RADIAL_STEP (< the sensor swath) so successive loops
overlap and every gate in the arena is passed within the horizontal sensor range.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.KaFa_v10_l3.cockpit import SearchSettings


def build_search_curve(
    start_xy: NDArray[np.float64], cfg: SearchSettings
) -> tuple[CubicSpline, float]:
    """Return (curve, t_total): an arena-clipped expanding-circle sweep at cfg.alt.

    The curve starts at start_xy (lifted to cfg.alt) and spirals outward to cfg.r_max. It is
    parameterised by waypoint index; ArcPath re-parameterises it by arc length, so only
    monotonicity matters here.
    """
    a = cfg.radial_step / (2.0 * np.pi)  # Archimedean: radius gained per radian
    pts: list[list[float]] = [[float(start_xy[0]), float(start_xy[1]), cfg.alt]]
    theta = cfg.r0 / a if a > 0 else 0.0  # start at radius r0
    while a * theta <= cfg.r_max:
        r = a * theta
        x = float(np.clip(r * np.cos(theta), -cfg.arena_x, cfg.arena_x))
        y = float(np.clip(r * np.sin(theta), -cfg.arena_y, cfg.arena_y))
        pt = [x, y, cfg.alt]
        # Skip points that collapse onto the previous one (can happen after clipping).
        if np.hypot(pt[0] - pts[-1][0], pt[1] - pts[-1][1]) >= 0.25:
            pts.append(pt)
        theta += cfg.angle_step
    points = np.asarray(pts, dtype=np.float64)
    taus = np.arange(len(points), dtype=np.float64)
    curve = CubicSpline(taus, points)
    return curve, float(taus[-1])
