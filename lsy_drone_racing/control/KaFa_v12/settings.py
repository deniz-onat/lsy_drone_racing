"""Tuning surface for KaFa_1500_v12 — the single place all parameters live.

These dataclasses are the fully *resolved* settings that KaFa_1500_v11_1 builds at runtime: the
original used a deep chain of cockpit ``import *`` constant modules and dataclass inheritance
(kafa1500_v6 -> KaFa_v8 -> KaFa_v10.4 -> KaFa_v10.6 -> KaFa_v11 -> KaFa_v11_1). Here every field
default is inlined as its resolved literal value, so this module is the one tuning surface with
no indirection. The values are verified field-by-field against the original v11_1 settings tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array3 = NDArray[np.float64]


@dataclass(frozen=True)
class FeedbackProfile:
    """Cascaded-PID gains (per world axis x, y, z)."""

    kp: Array3
    ki: Array3
    kd: Array3
    outer_i_limit: Array3


@dataclass(frozen=True)
class FeedbackSettings:
    """Cascaded-PID limits and the resolved gain profile."""

    outer_clamp: Array3 = field(
        default_factory=lambda: np.array([2.4, 2.35, 1.8], dtype=np.float64)
    )
    inner_i_limit: Array3 = field(
        default_factory=lambda: np.array([0.75, 0.75, 0.45], dtype=np.float64)
    )
    output_clamp: Array3 = field(
        default_factory=lambda: np.array([3.2, 3.2, 4.2], dtype=np.float64)
    )
    derivative_tau: Array3 = field(
        default_factory=lambda: np.array([0.05, 0.05, 0.06], dtype=np.float64)
    )
    eps: float = 1e-9
    profile: FeedbackProfile = field(
        default_factory=lambda: FeedbackProfile(
            np.array([0.60, 0.60, 1.65], dtype=np.float64),
            np.array([0.05, 0.05, 0.05], dtype=np.float64),
            np.array([0.35, 0.35, 0.50], dtype=np.float64),
            np.array([1.5, 1.5, 0.4], dtype=np.float64),
        )
    )


@dataclass(frozen=True)
class CommandSettings:
    """Feedforward, attitude, and final action limits."""

    lateral_accel_limit: float = 8.0
    feedforward_scale: float = 0.8
    norm_eps: float = 1e-6
    clip_actions: bool = True
    euler_limit: float = np.pi / 2
    thrust_min: float = 0.0854505226
    thrust_max: float = 0.8


@dataclass(frozen=True)
class RuntimeSettings:
    """Episode timing and replanning policy."""

    timeout_s: float = 30.0
    gravity: float = 9.81
    lookahead_s: float = 0.20
    projection_window_s: float = 0.6
    replan_gate_delta_m: float = 0.05
    replan_obstacle_delta_m: float = 0.05


@dataclass(frozen=True)
class PlannerSettings:
    """Observation-driven spline planning + v10.6 guarded-smoothing / parity-cap knobs.

    (This is the resolved v10.6 ``SmoothPlannerSettings``; the alias ``SmoothPlannerSettings``
    below keeps that name available for the planner code's type annotations.)
    """

    d_pre: float = 0.40
    d_post: float = 0.30
    d_stop: float = 0.30
    v_cruise: float = 1.25
    v_cruise_inter: float = 1.30
    max_speed: float = 1.60
    t_min_seg: float = 0.40
    r_obs: float = 0.20
    liftoff_z_threshold: float = 0.15
    liftoff_height: float = 0.55
    cold_start_min_seg: float = 0.45
    peri_gate_radius: float = 0.55
    clearance_height_delta: float = 0.15
    # False -> cross gates along the canonical +x axis (the env's counted direction).
    orient_gates_to_travel: bool = False
    turn_min_sharpness: float = 0.25
    turn_slow_gain: float = 1.0
    # Gate-post funnel: virtual columns injected at +/- gate_post_offset (see planner).
    funnel_enabled: bool = True
    gate_post_offset: float = 0.30
    # v10.4 launch-planner trim dial + wider obstacle keep-out.
    clr_ext_min: float = 1.0
    # v10.6 guarded waypoint smoothing + acceptance-guard knobs.
    smooth_pull: float = 0.5
    smooth_iters: int = 3
    reveal_window_m: float = 0.7
    obs_cap_radius: float = 0.45
    cross_tol_m: float = 0.15
    min_gain_s: float = 0.05


# The planner code annotates both names; v12 uses one resolved class for both.
SmoothPlannerSettings = PlannerSettings


@dataclass(frozen=True)
class TakeoffSettings:
    """Vertical takeoff-climb parameters (resolved v10.4 mini-takeoff values)."""

    alt: float = 0.42
    climb_speed: float = 0.9
    z_tol: float = 0.05
    time_margin: float = 1.0
    climb_time: float = 0.55


# v10.4's launch takeoff is the resolved takeoff used by v12.
LaunchTakeoffSettings = TakeoffSettings


@dataclass(frozen=True)
class MPCCSettings:
    """Tunnel-constrained time-optimal MPCC settings (resolved v11 values).

    The OCP solved by KaFa_v12.mpcc; also read by KaFa_v12.arc_path for the path/tunnel view.
    """

    horizon: int = 18
    step_dt: float = 0.05
    w_contour_base: float = 2.0
    w_contour_gate: float = 20.0
    gate_sigma: float = 0.5
    v_max: float = 3.2
    tilt_ratio: float = 0.85
    a_z_min: float = 7.0
    mu: float = 1.5
    v_theta_max: float = 3.2
    a_theta_max: float = 8.0
    r_dv: float = 0.01
    a_lat_max: float = 8.5
    v_min: float = 1.3
    w_lag: float = 1.0
    w_accel: float = 0.02
    ramp_s: float = 2.4
    ramp_start: float = 0.25
    max_iter: int = 20
    gravity: float = 9.81
    v_gate: float = 999.0
    gate_v_pre: float = 0.5
    gate_v_post: float = 0.15
    v_gate_react: float = 2.5
    react_delta_m: float = 9.9
    react_v_pre: float = 1.4
    react_v_post: float = 0.3
    proj_band_m: float = 0.6
    tunnel_w_max: float = 0.4
    tunnel_w_gate: float = 0.1
    tunnel_h_max: float = 0.35
    tunnel_h_gate: float = 0.07
    tunnel_taper_m: float = 1.0
    tunnel_w_min: float = 0.02
    tunnel_obs_margin: float = 0.2
    tunnel_post_margin: float = 0.18
    tunnel_z_floor: float = 0.1
    tunnel_z_ceil: float = 1.9
    tunnel_curv_frac: float = 0.5
    tunnel_slack_l1: float = 1000.0
    tunnel_slack_l2: float = 100.0
    v_gate_reveal: float = 2.5
    reveal_cap_all: bool = True
    reveal_pre_m: float = 0.7
    reveal_post_m: float = 0.3
    launch_cap_pre_m: float = 1.4


@dataclass(frozen=True)
class ControllerSettings:
    """All configurable values used by KaFa_1500_v12 (resolved v11_1 settings tree)."""

    planner: PlannerSettings = field(default_factory=PlannerSettings)
    feedback: FeedbackSettings = field(default_factory=FeedbackSettings)
    command: CommandSettings = field(default_factory=CommandSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    takeoff: TakeoffSettings = field(default_factory=TakeoffSettings)
    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
