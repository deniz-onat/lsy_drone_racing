"""Configuration for the KaFa_1500_v10_51 controller.

Inherits v10.5's full settings stack unchanged (same OCP/solver-cache key -> shared compiled
acados solver; same anchor band, launch ramp, reactive caps) and adds one nested block,
``EstimatorSettings`` -- the thrust-gain Kalman filter knobs (and the latency-comp flag). None of
these touch the solver-cache key, so the solver stays shared with v10.5/v10.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.KaFa_v10_5.settings import (
    ControllerSettings as _V105ControllerSettings,
)
from lsy_drone_racing.control.KaFa_v10_5.settings import MPCCSettings as _V105MPCCSettings
from lsy_drone_racing.control.KaFa_v10_51 import cockpit as cp


@dataclass(frozen=True)
class MPCCSettings(_V105MPCCSettings):
    """v10.5's MPCC settings with the KF-enabled HOT LAUNCH RAMP (a runtime knob: solver shared)."""

    ramp_start: float = cp.RAMP_START
    ramp_s: float = cp.RAMP_S


@dataclass(frozen=True)
class EstimatorSettings:
    """Thrust-gain Kalman filter + latency-compensation knobs (see KaFa_v10_51.cockpit)."""

    enabled: bool = cp.KF_ENABLED
    q: float = cp.KF_Q
    r: float = cp.KF_R
    p0: float = cp.KF_P0
    k_init: float = cp.KF_INIT
    clamp_lo: float = cp.KF_CLAMP_LO
    clamp_hi: float = cp.KF_CLAMP_HI
    freeze_ticks: int = cp.KF_FREEZE_TICKS
    latency_comp_enabled: bool = cp.LATENCY_COMP_ENABLED


@dataclass(frozen=True)
class ControllerSettings(_V105ControllerSettings):
    """v10.5's settings plus the hot launch ramp and the thrust-gain estimator block."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
    estimator: EstimatorSettings = field(default_factory=EstimatorSettings)
