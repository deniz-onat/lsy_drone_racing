"""Configuration for the KaFa_1500_v10_2 controller.

Inherits v10.1's gate-aware time-optimal MPCC settings unchanged and adds a single field --
``proj_band_m``, the half-width of the geometric correction applied around the solver's predicted
progress (the v10.2 dynamics-aware anchor). The MPCC solver structure and every speed/contouring/
solver knob are identical to v10.1, so the cached acados solver is shared; only how progress is
anchored to the drone differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.KaFa_v10_1.settings import (
    ControllerSettings as _V101ControllerSettings,
)
from lsy_drone_racing.control.KaFa_v10_1.settings import MPCCSettings as _V101MPCCSettings
from lsy_drone_racing.control.KaFa_v10_2 import cockpit as cp


@dataclass(frozen=True)
class MPCCSettings(_V101MPCCSettings):
    """v10.1's gate-aware time-optimal MPCC settings plus the dynamics-aware anchor band."""

    proj_band_m: float = cp.PROJ_BAND_M


@dataclass(frozen=True)
class ControllerSettings(_V101ControllerSettings):
    """v10.1's settings with the v10.2 MPCC settings (adds the predicted-progress anchor band)."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
