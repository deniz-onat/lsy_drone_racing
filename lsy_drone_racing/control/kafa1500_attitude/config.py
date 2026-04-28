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

    takeoff_height: float = 0.2
    takeoff_reached_distance: float = 0.1
    gate_inner_width: float = 0.40
    gate_inner_height: float = 0.40
    gate_safety_margin: float = 0.075
    d_pre: float = 0.55
    d_post: float = 0.55
    saturation_radius: float = 0.6
    obstacle_detour_margin: float = 0.25
    gate_avoidance_radius: float = 0.45
    max_detour_iterations: int = 5
    min_waypoint_spacing: float = 0.20
    collinear_angle_threshold_deg: float = 8.0
    sample_spacing: float = 0.04


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Closed-loop reference advancement parameters."""

    target_reached_distance: float = 0.15
    target_hysteresis: float = 0.01
    min_ticks_between_advances: int = 0
    start_index: int = 1
    max_advance_per_step: int = 16
    nearest_forward_search: int = 6
    nominal_speed: float = 0.8
    gate_speed: float = 0.4
    final_speed: float = 0.25
    gate_window_samples: int = 5
    lookahead_samples: int = 2
    base_lookahead_samples: int = 2
    speed_lookahead_gain: float = 1.0
    max_lookahead_samples: int = 10
    yaw_preview_samples: int = 5
    curvature_speed_enabled: bool = True
    max_lateral_acc: float = 6.0
    min_turn_speed: float = 0.8
    max_turn_speed: float = 4.0
    max_reference_acc_xy: float = 8.0
    max_reference_acc_z: float = 4.0


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    """Attitude feedback gains and output limits."""

    cx_kp_xy: float = 1.2
    cx_kp_z: float = 1.2
    cx_ki_xy: float = 0.04
    cx_ki_z: float = 0.02
    cx_kd_xy: float = 1e-6
    cx_kd_z: float = 1e-6
    cx_integral_limit: float = 5.0
    cv_kp_xy: float = 20.0
    cv_kp_z: float = 15.0
    cv_ki_xy: float = 0.05
    cv_ki_z: float = 0.01
    cv_kd_xy: float = 0.01
    cv_kd_z: float = 0.05
    cv_integral_limit: float = 5.0
    max_v_ref_xy: float = 3.0
    max_v_ref_z: float = 1.5
    max_acc_xy: float = 18.0
    max_acc_z: float = 12.0
    feedforward_acc_scale: float = 0.5
    max_feedforward_acc_xy: float = 8.0
    max_feedforward_acc_z: float = 4.0
    max_total_acc_xy: float = 10.0
    max_total_acc_z: float = 5.0
    yaw_kp: float = 5.0
    yaw_ki: float = 0.15
    yaw_kd: float = 0.2
    yaw_integral_limit: float = 15
    max_tilt: float = 10.05
    max_yaw_rate_step: float = 5.35
    attitude_smoothing: float = 0.85
    thrust_smoothing: float = 0.6
    hover_thrust_scale: float = 1.0
    gravity: float = 9.81
