"""Thrust-gain Kalman filter for KaFa_1500_v10_51 (pure numpy, unit-testable in isolation).

THE ONE UNCERTAINTY THE MPCC IGNORES. level2 randomises the drone mass +/-5 g on a ~43 g airframe
(+/-11.5%) and the inertia, while the controller maps every MPCC acceleration to a collective
thrust with the NOMINAL mass. With attitude/collective-thrust control we command a force
``F = m_nom * f_cmd``, where ``f_cmd = a_cmd + g.ez`` is the desired specific force in nominal-mass
units; the airframe of TRUE mass ``m_true`` realises ``a_real = F/m_true - g.ez``, so

    a_real + g.ez = (m_nom / m_true) * f_cmd = k * f_cmd .

The mismatch is a single MULTIPLICATIVE scalar ``k = m_nom / m_true`` (k ~ [0.90, 1.13] under
+/-5 g), not an additive bias -- so we estimate the scalar gain, and the controller corrects by
mapping thrust with ``m_nom / k_hat`` (= the estimated true mass) instead of ``m_nom``.

CLOSED-LOOP CORRECTNESS (why the regressor is the APPLIED force, not ``a_cmd + g.ez``). Once the
controller divides the mass by ``k_hat``, the force actually applied last step is
``F_applied = (m_nom / k_hat_prev) * f_cmd``, and ``a_real + g.ez = F_applied / m_true``. If the
filter regressed the measured ``a_real + g.ez`` on the OPEN-LOOP ``f_cmd = a_cmd + g.ez`` it would
see the gain ``k / k_hat_prev`` and converge to ``sqrt(k)`` (under-correcting). Regressing instead
on the specific force ACTUALLY applied, ``H = F_applied / m_nom`` (which already folds the 1/k_hat
correction in), recovers the true ``k`` regardless of the correction. The caller therefore passes
the applied thrust vector divided by ``m_nom`` as ``H`` -- see ``measurement``.

KF (scalar state, 3-row vector measurement -- closed form, no matrix library):

    state    k, random walk k_{t+1} = k_t + w,  w ~ N(0, q)
    measure  z = a_meas + g.ez = k * H + v,      v ~ N(0, r I_3),  H = applied specific force

    P_pred = P + q
    denom  = r + P_pred * |H|^2
    k     += P_pred * (H . z - |H|^2 * k) / denom        # Sherman-Morrison rank-1 update
    P      = P_pred * r / denom

then ``k`` is clamped to ``[clamp_lo, clamp_hi]``. The constant gravity term in ``H`` (|H| >= ~g/k)
keeps the filter excited even in hover, so it converges within the launch ramp before the first
fast gate. Freeze rules (hand-off transient, command saturation) are the CALLER's responsibility;
this class only does the math and clamps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ThrustGainKF"]


class ThrustGainKF:
    """Scalar Kalman filter for the thrust gain ``k = m_nom / m_true`` (pure numpy)."""

    def __init__(
        self,
        q: float,
        r: float,
        p0: float,
        k_init: float,
        clamp_lo: float,
        clamp_hi: float,
    ):
        """Store the (constant) process/measurement variances, prior, and clamp bounds."""
        self._q = float(q)
        self._r = float(r)
        self._p0 = float(p0)
        self._k_init = float(k_init)
        self._lo = float(clamp_lo)
        self._hi = float(clamp_hi)
        self.reset()

    def reset(self) -> None:
        """Re-seed the estimate and covariance for a new episode (k_hat = prior, P = P0)."""
        self.k_hat = float(np.clip(self._k_init, self._lo, self._hi))
        self._p = self._p0

    def update(self, z: NDArray[np.float64], h: NDArray[np.float64]) -> float:
        """One closed-form scalar KF update from measurement ``z`` and regressor ``h``; gives k_hat.

        ``z = a_meas + g.ez`` (realised specific force), ``h = F_applied / m_nom`` (commanded
        specific force actually applied last step). Both are 3-vectors satisfying ``z ~ k * h``.
        """
        z = np.asarray(z, dtype=np.float64).reshape(3)
        h = np.asarray(h, dtype=np.float64).reshape(3)
        hh = float(h @ h)
        hz = float(h @ z)
        p_pred = self._p + self._q
        denom = self._r + p_pred * hh
        self.k_hat += p_pred * (hz - hh * self.k_hat) / denom
        self._p = p_pred * self._r / denom
        self.k_hat = float(np.clip(self.k_hat, self._lo, self._hi))
        return self.k_hat

    @staticmethod
    def measurement(
        vel: NDArray[np.float64],
        vel_prev: NDArray[np.float64],
        dt: float,
        applied_thrust_vec: NDArray[np.float64],
        m_nom: float,
        gravity: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build ``(z, h)`` for ``update`` from raw signals (shared by the controller and tests).

        ``z = (vel - vel_prev)/dt + g.ez`` is the realised specific force; ``h`` is the specific
        force actually applied last step, ``applied_thrust_vec / m_nom`` (folds in any 1/k_hat
        correction -- see the module docstring on closed-loop correctness).
        """
        g_ez = np.array([0.0, 0.0, float(gravity)])
        a_meas = (np.asarray(vel, dtype=np.float64) - np.asarray(vel_prev, dtype=np.float64)) / dt
        z = a_meas + g_ez
        h = np.asarray(applied_thrust_vec, dtype=np.float64) / float(m_nom)
        return z, h

    def predict_state(
        self,
        pos: NDArray[np.float64],
        vel: NDArray[np.float64],
        a_cmd_prev: NDArray[np.float64],
        dt: float,
        gravity: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Latency-comp predict: roll (pos, vel) forward one tick at the estimated real accel.

        ``a_hat = k_hat * (a_cmd_prev + g.ez) - g.ez`` is the open-loop realised acceleration of the
        previous command; propagating one obs->actuation tick by it gives the state the solver
        should plan from when there is a one-step delay. Secondary / flag-gated (see cockpit 2.2).
        """
        g_ez = np.array([0.0, 0.0, float(gravity)])
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        a_hat = self.k_hat * (np.asarray(a_cmd_prev, dtype=np.float64) + g_ez) - g_ez
        return pos + vel * dt + 0.5 * a_hat * dt**2, vel + a_hat * dt
