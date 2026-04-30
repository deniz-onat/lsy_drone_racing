"""KaFa_1500_v5 denetleyicisi için merkezi parametreler."""
# ruff: noqa: TC002

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array3 = NDArray[np.float64]


@dataclass(frozen=True)
class SpeedProfile:
    """Bir sektörün başında, ortasında ve sonunda kullanılan göreli hız çarpanları."""

    start: float
    mid: float
    end: float


@dataclass(frozen=True)
class PlannerSettings:
    """Rota ve referans üretimi ayarları."""

    nominal_leg_times: Array3 = field(
        default_factory=lambda: np.array([3.85, 2.50, 3.50, 2.25], dtype=np.float64)
    )
    global_time_scale: float = 0.84
    leg_time_scale: Array3 = field(
        default_factory=lambda: np.array([0.52, 0.68, 0.65, 0.65], dtype=np.float64)
    )
    speed_profiles: tuple[SpeedProfile, ...] = (
        SpeedProfile(1.0, 1.0, 1.4),
        SpeedProfile(1.0, 1.8, 1.4),
        SpeedProfile(1.0, 1.0, 1.0),
        SpeedProfile(1.0, 1.0, 1.0),
    )
    override_file: Path = Path("logs/qualification_route_waypoints.json")
    override_tol: float = 1e-8
    max_avoidance_depth: int = 3
    avoidance_samples: int = 40
    obstacle_by_sector: tuple[int, ...] = (0, 0, 1, 3)
    clearance_trigger: Array3 = field(
        default_factory=lambda: np.array([0.15, 0.15, 0.15, 0.10], dtype=np.float64)
    )
    clearance_margin: Array3 = field(
        default_factory=lambda: np.array([0.03, 0.03, 0.03, 0.02], dtype=np.float64)
    )
    clearance_push_max: Array3 = field(
        default_factory=lambda: np.array([0.12, 0.12, 0.12, 0.06], dtype=np.float64)
    )
    gate0_lateral_bias: float = 0.035
    gate0_vertical_bias: float = 0.02
    gate1_exit_offset: float = 0.35
    gate1_shift_toward_gate0: float = 0.02
    gate2_entry_offset: float = 0.30
    gate2_exit_offset: float = 0.08
    gate3_entry_offset: float = 0.20
    gate3_exit_offset: float = 0.40
    gate_axis_eps: float = 1e-9
    obstacle_eps: float = 1e-6
    speed_eps: float = 1e-3

    @property
    def leg_times(self) -> Array3:
        """Her kapı sektörü için denetleyici saniyesi cinsinden süre."""
        return self.nominal_leg_times * self.global_time_scale * self.leg_time_scale

    @property
    def leg_starts(self) -> Array3:
        """Her sektör için mutlak denetleyici başlangıç zamanı."""
        return np.concatenate(([0.0], np.cumsum(self.leg_times[:-1])))


@dataclass(frozen=True)
class RuntimeSettings:
    """Bölüm ve yeniden planlama politikası."""

    timeout_s: float = 25.0
    replan_gate_delta_m: float = 0.005
    replan_near_gate_xy_m: float = 0.7
    gravity: float = 9.81


@dataclass(frozen=True)
class FeedbackProfile:
    """Kademeli denetleyiciye çözümlenen eski biçimli kazançlar."""

    kp: Array3
    ki: Array3
    kd: Array3
    outer_i_limit: Array3


@dataclass(frozen=True)
class FeedbackSettings:
    """PID sınırları ve sektör kazanç tabloları."""

    outer_clamp: Array3 = field(
        default_factory=lambda: np.array([2.35, 2.35, 1.85], dtype=np.float64)
    )
    inner_i_limit: Array3 = field(
        default_factory=lambda: np.array([0.75, 0.75, 0.45], dtype=np.float64)
    )
    output_clamp: Array3 = field(
        default_factory=lambda: np.array([3.2, 3.2, 4.2], dtype=np.float64)
    )
    derivative_tau: Array3 = field(
        default_factory=lambda: np.array([0.045, 0.045, 0.060], dtype=np.float64)
    )
    eps: float = 1e-9
    profiles: tuple[FeedbackProfile, ...] = field(
        default_factory=lambda: (
            FeedbackProfile(
                np.array([0.60, 0.60, 1.65], dtype=np.float64),
                np.array([0.05, 0.05, 0.05], dtype=np.float64),
                np.array([0.35, 0.35, 0.50], dtype=np.float64),
                np.array([1.5, 1.5, 0.4], dtype=np.float64),
            ),
            FeedbackProfile(
                np.array([0.65, 0.65, 1.65], dtype=np.float64),
                np.array([0.045, 0.045, 0.05], dtype=np.float64),
                np.array([0.55, 0.55, 0.50], dtype=np.float64),
                np.array([1.5, 1.5, 0.4], dtype=np.float64),
            ),
            FeedbackProfile(
                np.array([0.65, 0.65, 1.55], dtype=np.float64),
                np.array([0.045, 0.045, 0.05], dtype=np.float64),
                np.array([0.45, 0.45, 0.50], dtype=np.float64),
                np.array([1.5, 1.5, 0.4], dtype=np.float64),
            ),
            FeedbackProfile(
                np.array([0.65, 0.65, 1.65], dtype=np.float64),
                np.array([0.045, 0.045, 0.05], dtype=np.float64),
                np.array([0.30, 0.30, 0.50], dtype=np.float64),
                np.array([1.5, 1.5, 0.4], dtype=np.float64),
            ),
        )
    )


@dataclass(frozen=True)
class CommandSettings:
    """Feedforward, attitude ve nihai eylem sınırları."""

    lateral_accel_limit: float = 16.0
    feedforward_scale: float = 0.75
    norm_eps: float = 1e-6
    clip_actions: bool = True
    euler_limit: float = np.pi / 2
    thrust_min: float = 0.0854505226
    thrust_max: float = 0.8


@dataclass(frozen=True)
class ControllerSettings:
    """Denetleyici tarafından kullanılan tüm ayarlanabilir değerler."""

    planner: PlannerSettings = field(default_factory=PlannerSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    feedback: FeedbackSettings = field(default_factory=FeedbackSettings)
    command: CommandSettings = field(default_factory=CommandSettings)


DEFAULT_GATE_POS = np.array(
    [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]], dtype=np.float64
)
DEFAULT_GATE_RPY = np.array(
    [[0.0, 0.0, -0.78], [0.0, 0.0, 2.35], [0.0, 0.0, 3.14], [0.0, 0.0, 0.0]], dtype=np.float64
)
DEFAULT_OBSTACLES = np.array(
    [[0.0, 0.75, 1.55], [1.0, 0.25, 1.55], [-1.5, -0.25, 1.55], [-0.5, -0.75, 1.55]],
    dtype=np.float64,
)
START_WAYPOINT = np.array([-1.5, 0.75, 0.05], dtype=np.float64)
GATE1_ARC_POINTS = np.array([[1.08, -0.16, 0.86], [1.38, 0.12, 1.08]], dtype=np.float64)
ROUTE2_MIDPOINT = np.array([0.0, 0.25, 1.0], dtype=np.float64)
ROUTE3_MIDPOINT = np.array([-0.55, -0.42, 0.85], dtype=np.float64)
