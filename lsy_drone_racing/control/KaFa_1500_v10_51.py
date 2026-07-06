"""Gate-aware time-optimal MPCC with online mass estimation (KaFa_1500_v10_51).

v10.51 is v10.5 with a thrust-gain Kalman filter (KaFa_v10_51.estimator.ThrustGainKF). The racing
pipeline is byte-for-byte v10.5's -- the OCP, the gate-aware contouring, the fast launch, the
replan-continuity rebase, the reactive caps, and the v10.2/v10.5 dynamics-aware progress anchor,
all unchanged, sharing the same compiled acados solver. The ONLY change is how the MPCC's world-
frame acceleration becomes a collective-thrust command.

THE TARGET. sim observations are noise-free, so this is not measurement filtering. It attacks the
one uncertainty the architecture ignored: level2 randomises the drone mass +/-5 g on a ~43 g
airframe (+/-11.5%), while v10.x maps every MPCC acceleration to thrust with the NOMINAL mass --
``thrust = m_nom * (a_cmd + g.ez)`` -- so the airframe of true mass m_true realises
``a_real = (m_nom/m_true)(a_cmd + g.ez) - g.ez``, i.e. every commanded acceleration is scaled by the
scalar ``k = m_nom/m_true``. At gate speed that is overshoot/undershoot the contouring weights have
to absorb. The KF estimates k online and the thrust mapping uses ``m_nom / k_hat`` (the estimated
true mass) instead of ``m_nom``, so a_real tracks a_cmd. The OCP is untouched: ``a_max`` is baked
into the compiled constraint set, so the nominal envelope stays as a conservative bound (a heavy
draw has ~11% less authority than the OCP believes; the 0.45 tilt cap leaves margin).

MEASUREMENT (each NAVIGATE tick): ``a_meas = (vel_t - vel_{t-1})/dt`` (sim vel is exact, so the
finite difference is clean). The regressor is the specific force ACTUALLY APPLIED last step,
``F_applied/m_nom`` -- NOT the raw ``a_cmd + g.ez`` -- because once the 1/k_hat correction is live
the realised gain relative to the raw command is ``k/k_hat``, which would converge the filter to
``sqrt(k)``; regressing on the applied force recovers the true k (see KaFa_v10_51.estimator for the
derivation). Updates are FROZEN (i) for the first KF_FREEZE_TICKS NAVIGATE ticks (the
takeoff hand-off transient) and (ii) on any tick whose attitude command saturated (the commanded
thrust was not what flew). Replans do NOT freeze it -- velocity is physical across the rebase.

Both estimators are independently switchable (KF_ENABLED, LATENCY_COMP_ENABLED) so an eval can
attribute gains. The KF does NOT attack the +/-0.15 m reveal-correction ceiling (proven binding in
the v10.4 ledger); it attacks the mass axis. Implemented as a thin subclass of KaFa1500V105 (the
NAVIGATE flow is copied with the KF woven in, the house pattern). REQUIRES the acados environment.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.kafa1500_v6.attitude import _vector_to_attitude
from lsy_drone_racing.control.KaFa_1500_v10_5 import KaFa1500V105
from lsy_drone_racing.control.KaFa_v10_5.arc_path import GateArcPath
from lsy_drone_racing.control.KaFa_v10_5.mpcc import MPCC
from lsy_drone_racing.control.KaFa_v10_51.estimator import ThrustGainKF
from lsy_drone_racing.control.KaFa_v10_51.settings import ControllerSettings

if TYPE_CHECKING:
    from lsy_drone_racing.control.kafa1500_v6.state import DroneObservation

_SAT_EPS = 1e-3  # margin for declaring a clipped attitude/thrust command "saturated"

# Eval-only speed-knob overrides (env var -> MPCCSettings field) for the step-7 frontier re-probe
# (does the mass correction let a hotter config hold finish?). No effect unless the var is set.
_SPEED_OVERRIDES = {
    "KAFA_VTHETA": "v_theta_max",
    "KAFA_ALAT": "a_lat_max",
    "KAFA_RAMP_START": "ramp_start",
    "KAFA_RAMP_S": "ramp_s",
    "KAFA_WLAG": "w_lag",
    "KAFA_WGATE": "w_contour_gate",
}


def _eval_flag(name: str, default: bool) -> bool:
    """Read an optional eval-only override env var (1/0/true/false); default to the cockpit value.

    Lets the paired harness A/B the KF and latency-comp switches WITHOUT editing the cockpit
    (e.g. KAFA_KF=0 for the KF-off sanity column, KAFA_LATENCY=1 for the latency-comp column).
    Production behaviour is unchanged unless the var is set.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class KaFa1500V1051(KaFa1500V105):
    """v10.5's launch-optimised, fold-safe racing MPCC with an online thrust-gain (mass) filter."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.5, then attach the thrust-gain KF and its per-step measurement bookkeeping."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        # v10.5's __init__ cached the runtime knobs from v10.5's settings; v10.51 ships a hotter
        # ramp (a runtime knob), so re-read them from the v10.51 settings. The ramp is NOT in the
        # solver cache key, so the v10.5-built solver stays valid and shared -- no rebuild needed.
        self._refresh_runtime_knobs()
        est = self._settings.estimator
        self._kf_enabled = _eval_flag("KAFA_KF", bool(est.enabled))
        self._latency_comp = _eval_flag("KAFA_LATENCY", bool(est.latency_comp_enabled))
        self._freeze_ticks = int(est.freeze_ticks)
        self._kf = ThrustGainKF(est.q, est.r, est.p0, est.k_init, est.clamp_lo, est.clamp_hi)
        self._reset_kf_state()
        self._apply_speed_overrides()

    def _refresh_runtime_knobs(self) -> None:
        """Re-read the cached knobs the inherited NAVIGATE flow uses from the MPCC settings."""
        m = self._settings.mpcc
        self._v_theta_max = m.v_theta_max
        self._ramp_s, self._ramp_start = m.ramp_s, m.ramp_start
        self._a_lat_max, self._v_min = m.a_lat_max, m.v_min
        self._w_base, self._w_gate, self._gate_sigma = (
            m.w_contour_base, m.w_contour_gate, m.gate_sigma,
        )
        self._v_gate_react = m.v_gate_react
        self._react_delta = m.react_delta_m
        self._react_v_pre, self._react_v_post = m.react_v_pre, m.react_v_post
        self._proj_band = m.proj_band_m

    def _apply_speed_overrides(self) -> None:
        """Apply eval-only KAFA_* speed-knob env overrides (step-7 re-probe); no-op if none set.

        Rebuilds the MPCC settings/solver with the overridden knobs, then refreshes the cached
        runtime knobs. IMPORTANT LIMITATION: only RUNTIME knobs (v_theta_max's ramp/arc use,
        a_lat_max, ramp_start, ramp_s, w_contour_gate) actually change behaviour via env override.
        SOLVER-COST fields (w_lag, mu, a_theta_max) do NOT take effect in-process: rebuilding here
        regenerates into the shared ``kafa_v10_4`` codegen namespace and acados dlopen returns the
        dylib already built by the super().__init__ chain (verified: w_lag env probes were no-ops).
        To retune a solver-cost field, bake it into the cockpit so it is set at the FIRST build.
        """
        ov: dict[str, float] = {}
        for env_name, field in _SPEED_OVERRIDES.items():
            raw = os.environ.get(env_name)
            if raw is not None:
                ov[field] = float(raw)
        if not ov:
            return
        if "v_theta_max" in ov:
            ov.setdefault("v_max", ov["v_theta_max"])
        mpcc = dataclasses.replace(self._settings.mpcc, **ov)
        self._settings = dataclasses.replace(self._settings, mpcc=mpcc)
        self._mpcc = MPCC(mpcc, self._command.thrust_max / self._mass)
        self._refresh_runtime_knobs()

    def _reset_kf_state(self) -> None:
        """Clear the per-episode bookkeeping (last vel / applied thrust / saturation flag)."""
        self._last_vel: np.ndarray | None = None
        self._last_thrust_vec: np.ndarray | None = None
        self._last_accel_raw = np.zeros(3)  # raw MPCC accel (for the optional latency predict)
        self._last_saturated = False

    def reset(self) -> None:
        """Reset v10.5 state plus the KF estimate/covariance and the measurement bookkeeping."""
        super().reset()
        self._kf.reset()
        self._reset_kf_state()

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """v10.5's flow + thrust-gain KF: update -> (optional predict) -> solve -> 1/k_hat map."""
        # --- KF UPDATE from the previous command's realised acceleration (freeze rules apply) ---
        navigate_tick = self._tick - self._nav_start_tick
        if (
            self._kf_enabled
            and self._last_vel is not None
            and self._last_thrust_vec is not None
            and navigate_tick >= self._freeze_ticks
            and not self._last_saturated
        ):
            z, h = ThrustGainKF.measurement(
                frame.vel, self._last_vel, self._dt, self._last_thrust_vec,
                self._mass, self._gravity,
            )
            self._kf.update(z, h)

        # --- v10.5 plan / gate-aware path (re)build (copied verbatim) ---
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        new_plan = rebuilt or self._path is None
        if new_plan:
            first = max(frame.target_gate, 0)
            gates_ahead = plan.gate_pos_snapshot[first:]
            deltas = np.linalg.norm(
                plan.gate_pos_snapshot[first:] - self._gate_nominal[first:], axis=1
            )
            caps = np.where(deltas > self._react_delta, self._v_gate_react, np.inf)
            if first == 0:
                caps[0] = self._v_gate_react
            path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
                caps, self._react_v_pre, self._react_v_post,
            )
            self._s = path.project(frame.pos, 0.0)
            if self._path is None:
                self._mpcc.set_path(path)
            else:
                self._mpcc.rebase(path, self._s)
            self._path = path

        # --- optional latency comp: roll the SOLVE state forward one obs->actuation tick ---
        if self._latency_comp and self._kf_enabled:
            solve_pos, solve_vel = self._kf.predict_state(
                frame.pos, frame.vel, self._last_accel_raw, self._dt, self._gravity
            )
        else:
            solve_pos, solve_vel = frame.pos, frame.vel

        # --- v10.5 dynamics-aware progress anchor (copied verbatim, on the solve position) ---
        th_pred = self._mpcc.predicted_progress()
        if th_pred is None:
            self._s = self._path.project(solve_pos, self._s)
        else:
            self._s = self._path.project_near(solve_pos, th_pred, self._proj_band)
            self._band_calls += 1
            if abs(self._s - th_pred) >= self._proj_band - 1e-9:
                self._band_edge_hits += 1
        if not new_plan and self._anchor_prev_s is not None:
            self._anchor_jumps.append(abs(self._s - self._anchor_prev_s))
        self._anchor_prev_s = self._s

        # --- solve, then map acceleration to thrust with the ESTIMATED mass (m_nom / k_hat) ---
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(solve_pos, solve_vel, self._s, self._v_theta_max * ramp)
        self._last_accel_raw = np.asarray(accel, dtype=np.float64).copy()
        mass_eff = (self._mass / self._kf.k_hat) if self._kf_enabled else self._mass
        thrust_vector = mass_eff * (accel + np.array([0.0, 0.0, self._gravity]))
        action = _vector_to_attitude(thrust_vector, frame.quat, self._command)

        # --- record what was actually commanded for the next tick's measurement + freeze test ---
        self._last_thrust_vec = thrust_vector
        self._last_vel = np.asarray(frame.vel, dtype=np.float64).copy()
        self._last_saturated = self._action_saturated(action)
        return action

    def _action_saturated(self, action: np.ndarray) -> bool:
        """True if the attitude command hit the collective-thrust or tilt clip (a_cmd != flown)."""
        cmd = self._command
        collective = float(action[3])
        if collective >= cmd.thrust_max - _SAT_EPS or collective <= cmd.thrust_min + _SAT_EPS:
            return True
        return bool(np.any(np.abs(action[:2]) >= cmd.euler_limit - _SAT_EPS))

    def estimator_telemetry(self) -> dict[str, float]:
        """Current thrust-gain estimate (for paired eval vs the true sampled mass)."""
        return {"k_hat": float(self._kf.k_hat)}
