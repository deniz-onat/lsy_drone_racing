"""Configuration for the KaFa_1500_v10_6 controller.

Inherits v10.5's full settings stack -- the MPCC settings are UNCHANGED (same OCP/solver-cache
key, so the compiled acados solver is shared with v10.4/v10.5) -- and extends only the planner
settings with the smoothing and guard knobs read by KaFa_v10_6.trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.KaFa_v10_4.settings import (
    LaunchPlannerSettings as _V104PlannerSettings,
)
from lsy_drone_racing.control.KaFa_v10_5.settings import (
    ControllerSettings as _V105ControllerSettings,
)
from lsy_drone_racing.control.KaFa_v10_6 import cockpit as cp


@dataclass(frozen=True)
class SmoothPlannerSettings(_V104PlannerSettings):
    """v10.4's planner settings plus the v10.6 smoothing and acceptance-guard knobs."""

    smooth_pull: float = cp.SMOOTH_PULL
    smooth_iters: int = cp.SMOOTH_ITERS
    reveal_window_m: float = cp.REVEAL_WINDOW_M
    obs_cap_radius: float = cp.OBS_CAP_RADIUS
    cross_tol_m: float = cp.CROSS_TOL_M
    min_gain_s: float = cp.MIN_GAIN_S


@dataclass(frozen=True)
class ControllerSettings(_V105ControllerSettings):
    """v10.5's settings (launch, anchor band, reactive caps) + the smoothing planner."""

    planner: SmoothPlannerSettings = field(default_factory=SmoothPlannerSettings)
