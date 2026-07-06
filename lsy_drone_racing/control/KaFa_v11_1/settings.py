"""Configuration for the KaFa_1500_v11_1 controller.

v11's settings (tunnel MPCC, same compiled solver) with v10.6's smoothing planner settings --
the guarded smoother's knobs and the parity-cap geometry are reused unchanged from the v10.6
ledger; only the consumer changes (the de-paced tunnel MPCC instead of v10.5's cost-paced one).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.KaFa_v10_6.settings import SmoothPlannerSettings
from lsy_drone_racing.control.KaFa_v11.settings import ControllerSettings as _V11ControllerSettings


@dataclass(frozen=True)
class ControllerSettings(_V11ControllerSettings):
    """v11's settings over the v10.6 guarded-smoothing planner."""

    planner: SmoothPlannerSettings = field(default_factory=SmoothPlannerSettings)
