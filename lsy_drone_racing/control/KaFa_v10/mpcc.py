"""Time-optimal MPCC core for KaFa_1500_v10.

A point-mass model-predictive *contouring* controller in the time-optimal form (Liniger
2015; Romero/Scaramuzza, "MPCC for Time-Optimal Quadrotor Flight", IEEE T-RO 2022). It
keeps v9's triple-integrator dynamics and thrust/tilt/velocity limits, but promotes the path
progress (arc length ``theta``) and its rate ``v_theta`` to decision variables and pays the
optimiser to advance with a linear ``-mu * v_theta`` reward. Traversal speed is therefore an
output of the optimisation at the dynamic limit -- fast on straights, auto-braked into
corners -- instead of a hand-tuned receding-reference rate.

The reference ``p_d(theta)`` is a CasADi b-spline whose node positions are *parameters*, so
the NLP is built once and the path is swapped on a replan by setting those parameters (no
graph rebuild). The optimiser autodiffs the b-spline for the path tangent. Built once with
casadi + ipopt and re-solved (warm-started) every control step -- no acados build required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.KaFa_v10.settings import MPCCSettings

__all__ = ["MPCC"]


class MPCC:
    """Receding-horizon time-optimal contouring controller returning a world acceleration."""

    def __init__(self, settings: MPCCSettings, a_max: float, s_grid: NDArray[np.float64]):
        """Build the parametric NLP once; it is re-solved each step with new parameters."""
        self.s_grid = np.asarray(s_grid, dtype=np.float64)
        m = int(self.s_grid.size)
        self.n = int(settings.horizon)
        dt, g = float(settings.step_dt), float(settings.gravity)
        s_end = float(self.s_grid[-1])

        opti = ca.Opti()
        pos = opti.variable(3, self.n + 1)
        vel = opti.variable(3, self.n + 1)
        acc = opti.variable(3, self.n)
        th = opti.variable(1, self.n + 1)     # arc-length progress (m)
        vth = opti.variable(1, self.n + 1)    # progress rate (m/s)
        dvth = opti.variable(1, self.n)       # progress acceleration (m/s^2), the new control
        p0, v0 = opti.parameter(3), opti.parameter(3)
        th0 = opti.parameter(1)               # progress anchored to the drone's foot-point
        vth_cap = opti.parameter(1)           # ramped progress-rate cap
        cx, cy, cz = opti.parameter(m), opti.parameter(m), opti.parameter(m)
        gvec = ca.DM([0.0, 0.0, g])
        path = self._path_fn(self.s_grid, m)  # (theta, cx, cy, cz) -> (point, unit tangent)

        opti.subject_to(pos[:, 0] == p0)
        opti.subject_to(vel[:, 0] == v0)
        opti.subject_to(th[:, 0] == th0)
        cost = 0
        for k in range(self.n):
            opti.subject_to(pos[:, k + 1] == pos[:, k] + vel[:, k] * dt + 0.5 * acc[:, k] * dt**2)
            opti.subject_to(vel[:, k + 1] == vel[:, k] + acc[:, k] * dt)
            opti.subject_to(th[:, k + 1] == th[:, k] + vth[:, k] * dt)
            opti.subject_to(vth[:, k + 1] == vth[:, k] + dvth[:, k] * dt)
            thrust = acc[:, k] + gvec  # collective-thrust direction (per unit mass)
            opti.subject_to(ca.sumsqr(thrust) <= a_max**2)  # thrust magnitude limit
            opti.subject_to(thrust[2] >= settings.a_z_min)  # keep lift (no free-fall)
            opti.subject_to(ca.sumsqr(thrust[:2]) <= (settings.tilt_ratio * thrust[2]) ** 2)  # tilt
            opti.subject_to(ca.sumsqr(vel[:, k]) <= settings.v_max**2)
            opti.subject_to(opti.bounded(-settings.a_theta_max, dvth[:, k], settings.a_theta_max))
        for k in range(self.n + 1):
            opti.subject_to(opti.bounded(0.0, vth[:, k], vth_cap))  # forward, capped progress
            opti.subject_to(th[:, k] <= s_end)                      # stay on the grid
            p_ref, tangent = path(th[:, k], cx, cy, cz)
            err = pos[:, k] - p_ref
            lag = ca.dot(err, tangent)
            perp = err - lag * tangent
            cost += settings.w_contour * ca.sumsqr(perp) + settings.w_lag * lag**2
            cost += -settings.mu * vth[:, k]  # time-optimality driver: reward progress
        cost += settings.w_accel * ca.sumsqr(acc) + settings.r_dv * ca.sumsqr(dvth)
        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {"print_time": False},
            {"print_level": 0, "sb": "yes", "max_iter": int(settings.max_iter)},
        )
        self._opti = opti
        self._vars = (pos, vel, acc, th, vth, dvth)
        self._params = (p0, v0, th0, vth_cap, cx, cy, cz)
        self._dt = dt
        self._nodes: NDArray[np.float64] | None = None
        self.reset()

    @staticmethod
    def _path_fn(s_grid: NDArray[np.float64], m: int) -> ca.Function:
        """Build a (arc, node coeffs) -> (point, unit tangent) function from a b-spline once.

        Passing node positions as the b-spline coefficients keeps them as optimiser
        parameters (so the NLP never rebuilds on a replan); arc-length sampling makes the
        b-spline near unit-speed, so the autodiffed tangent only needs a gentle normalisation.
        """
        px = ca.interpolant("px", "bspline", [s_grid], 1)
        py = ca.interpolant("py", "bspline", [s_grid], 1)
        pz = ca.interpolant("pz", "bspline", [s_grid], 1)
        s = ca.MX.sym("s")
        cx, cy, cz = ca.MX.sym("cx", m), ca.MX.sym("cy", m), ca.MX.sym("cz", m)
        point = ca.vertcat(px(s, cx), py(s, cy), pz(s, cz))
        tangent = ca.jacobian(point, s)
        tangent = tangent / (ca.norm_2(tangent) + 1e-6)
        return ca.Function("path", [s, cx, cy, cz], [point, tangent])

    def set_path(self, nodes: NDArray[np.float64]) -> None:
        """Load a new arc-length-resampled path (M x 3 node positions) and drop the warm start."""
        self._nodes = np.asarray(nodes, dtype=np.float64)
        self.reset()

    def solve(
        self,
        pos: NDArray[np.float64],
        vel: NDArray[np.float64],
        th0: float,
        vth_cap: float,
    ) -> NDArray[np.float64]:
        """Return the first optimal acceleration; fall back to the last plan on failure."""
        p0, v0, th0p, vthp, cx, cy, cz = self._params
        opti = self._opti
        opti.set_value(p0, pos)
        opti.set_value(v0, vel)
        opti.set_value(th0p, th0)
        opti.set_value(vthp, max(float(vth_cap), 1e-3))
        opti.set_value(cx, self._nodes[:, 0])
        opti.set_value(cy, self._nodes[:, 1])
        opti.set_value(cz, self._nodes[:, 2])
        seed = self._seed(float(th0), max(float(vth_cap), 1e-3))
        for var, value in zip(self._vars, seed):
            opti.set_initial(var, value)
        try:
            sol = opti.solve()
            self._plan = {name: np.asarray(sol.value(var)).reshape(var.shape) for name, var in
                          zip(("pos", "vel", "acc", "th", "vth", "dvth"), self._vars)}
            self._have_plan = True
        except RuntimeError:
            self._plan = self._shift(self._plan) if self._have_plan else self._plan
        return self._plan["acc"][:, 0].copy()

    def reset(self) -> None:
        """Forget the warm start between episodes / on a new path."""
        n = self.n
        self._plan = {
            "pos": np.zeros((3, n + 1)), "vel": np.zeros((3, n + 1)), "acc": np.zeros((3, n)),
            "th": np.zeros((1, n + 1)), "vth": np.zeros((1, n + 1)), "dvth": np.zeros((1, n)),
        }
        self._have_plan = False

    @staticmethod
    def _shift(plan: dict[str, NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
        """Receding-horizon shift: drop the consumed first step and hold the last."""
        return {k: np.concatenate([v[:, 1:], v[:, -1:]], axis=1) for k, v in plan.items()}

    def _seed(self, th0: float, vth_cap: float) -> tuple[NDArray[np.float64], ...]:
        """Warm start: shift the last plan and re-anchor progress to the new foot-point."""
        if not self._have_plan:
            th = (th0 + np.arange(self.n + 1) * vth_cap * self._dt).reshape(1, -1)
            vth = np.full((1, self.n + 1), vth_cap)
            zeros = (np.zeros((3, self.n + 1)), np.zeros((3, self.n + 1)), np.zeros((3, self.n)))
            return (*zeros[:2], np.zeros((3, self.n)), th, vth, np.zeros((1, self.n)))
        s = self._shift(self._plan)
        s["th"] = s["th"] + (th0 - float(s["th"][0, 0]))  # re-base to the current anchor
        return (s["pos"], s["vel"], s["acc"], s["th"], s["vth"], s["dvth"])
