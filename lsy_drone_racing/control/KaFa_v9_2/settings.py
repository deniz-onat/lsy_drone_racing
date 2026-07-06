"""Configuration for the KaFa_1500_v9.2 controller.

v9.2 reuses v9.1's MPCC, governor, planner, and takeoff settings verbatim; the only addition
is the curvature speed-profile knobs, built from the v9.2 cockpit.
"""

from __future__ import annotations

from dataclasses import dataclass

from lsy_drone_racing.control.KaFa_v9_2 import cockpit as cp


@dataclass(frozen=True)
class SpeedProfileSettings:
    """Curvature-aware reference speed-profile knobs (from the v9.2 cockpit).

    v_max is the MPCC hard velocity cap v9.2 raises above v9.1's so the tracker has headroom to
    catch the faster (curvature-profiled) reference instead of clipping it at the profile cap.
    """

    v_cap: float = cp.V_CAP
    a_lat_max: float = cp.A_LAT_MAX
    v_min: float = cp.V_MIN
    v_max: float = cp.V_MAX
    ramp_start: float = cp.RAMP_START
    ramp_s: float = cp.RAMP_S
