"""Tunable parameters for KaFa1500 attitude control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _vec(values: tuple[float, float, float]) -> NDArray[np.float32]:
    """Create a float32 vector dataclass default."""
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class PathConfig:
    """Gate-aware cubic path generation parameters."""

    takeoff_height: float = 0.82
    takeoff_reached_distance: float = 0.14
    gate_inner_width: float = 0.40
    gate_inner_height: float = 0.40
    gate_safety_margin: float = 0.075
    vertical_reference_offset: float = -0.035
    d_pre: float = 0.42
    d_post: float = 0.32
    d_pass: float = 0.56
    d_pre_per_gate: tuple[float, ...] = (0.50, 0.42, 0.38, 0.36)
    d_post_per_gate: tuple[float, ...] = (0.32, 0.34, 0.36, 0.38)
    obstacle_radius: float = 0.015
    drone_radius: float = 0.10
    obstacle_tracking_margin: float = 0.24
    bypass_extra: float = 0.22
    max_bypass_points: int = 3
    sample_spacing: float = 0.055
    control_smoothing_passes: int = 2
    control_smoothing_weight: float = 0.35

    @property
    def obstacle_clearance(self) -> float:
        """XY clearance used around obstacle poles."""
        return self.drone_radius + self.obstacle_radius + self.obstacle_tracking_margin


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Closed-loop reference advancement parameters."""

    target_reached_distance: float = 0.35
    target_hysteresis: float = 0.05
    min_ticks_between_advances: int = 0
    start_index: int = 1
    max_advance_per_step: int = 16
    nearest_forward_search: int = 6
    nominal_speed: float = 0.8
    gate_speed: float = 0.4
    final_speed: float = 0.25
    gate_window_samples: int = 5
    lookahead_samples: int = 2


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    """Attitude feedback gains and output limits."""

    cx_kp_xy: float = 1.2
    cx_kp_z: float = 1.2
    cx_ki_xy: float = 0.04
    cx_ki_z: float = 0.04
    cx_kd_xy: float = 1e-6
    cx_kd_z: float = 1e-6
    cx_integral_limit: float = 5.0
    cv_kp_xy: float = 25.0
    cv_kp_z: float = 25.0
    cv_ki_xy: float = 0.5
    cv_ki_z: float = 0.5
    cv_kd_xy: float = 0.01
    cv_kd_z: float = 0.01
    cv_integral_limit: float = 5.0
    max_v_ref_xy: float = 3.0
    max_v_ref_z: float = 1.5
    max_acc_xy: float = 8.0
    max_acc_z: float = 4.0
    yaw_kp: float = 3.0
    yaw_ki: float = 0.15
    yaw_kd: float = 0.4
    yaw_integral_limit: float = 15
    max_tilt: float = 10.05
    max_yaw_rate_step: float = 5.35
    attitude_smoothing: float = 0.85
    thrust_smoothing: float = 0.05
    hover_thrust_scale: float = 1.0
    gravity: float = 9.81
