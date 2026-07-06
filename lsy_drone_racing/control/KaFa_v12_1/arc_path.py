"""Consolidated arc-length path / tunnel chain for KaFa_v12_1 (resolved-base merge).

This module flattens the single-inheritance arc-length path chain that was spread across the
KaFa_v10 -> KaFa_v10_1 -> KaFa_v10_3 -> KaFa_v10_4 -> KaFa_v10_5 -> KaFa_v11 -> KaFa_v11_1
source files into one file, preserving the original Method Resolution Order. The four
intermediate ``GateArcPath`` classes (which collided by name across versions) are renamed to
private names ``_GateArcPathV101``, ``_GateArcPathV103``, ``_GateArcPathV104``,
``_GateArcPathV105`` while keeping the chain identical. The public entry points are unchanged:

- ``ArcPath``  (v10): dense arc-length tables of path point, unit tangent, curvature speed.
- ``TunnelArcPath`` (v11): the same plus tunnel half-extent tables and all-gate reveal caps.
- ``CappedTunnelArcPath`` (v11.1): the v11 tables plus the smoothed plan's parity caps.

The module-level helper ``_limit_longitudinal`` (introduced in v11) is also kept public per the
package's required exports. Class bodies, docstrings, and logic are copied verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.KaFa_v12_1.speed_profile import SpeedProfile

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline

    from lsy_drone_racing.control.KaFa_v12_1.settings import MPCCSettings

_EPS = 1e-9
_PROJ_WINDOW_M = 2.0  # forward window (m) for the projection search


class ArcPath:
    """Dense arc-length tables (point, unit tangent, curvature speed) for one plan spline."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        v_cap: float,
        a_lat_max: float,
        v_min: float,
        n: int = 600,
    ):
        """Sample the plan densely and precompute arc length, unit tangent, and v_curv(s)."""
        taus = np.linspace(0.0, float(t_total), n)
        pts = np.asarray(curve(taus), dtype=np.float64)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        tan = np.gradient(pts, s, axis=0)  # dp/ds ~ unit tangent (arc-length parameterised)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + _EPS
        self._s, self._pts, self._tan = s, pts, tan
        self._vcurv = SpeedProfile(curve, t_total, v_cap, a_lat_max, v_min).at_arc(s)
        self.total = float(s[-1])

    def project(self, pos: NDArray[np.float64], s_min: float) -> float:
        """Arc length of the nearest path point within a forward window of s_min."""
        lo = int(np.searchsorted(self._s, s_min))
        hi = max(int(np.searchsorted(self._s, s_min + _PROJ_WINDOW_M)), lo + 1)
        nearest = int(np.argmin(np.linalg.norm(self._pts[lo:hi] - pos, axis=1)))
        return float(self._s[lo + nearest])

    def eval(
        self, arc: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return (point, unit tangent, curvature-speed) at each arc length (clamped to path)."""
        a = np.clip(np.asarray(arc, dtype=np.float64), 0.0, self.total)
        point = np.column_stack([np.interp(a, self._s, self._pts[:, i]) for i in range(3)])
        tan = np.column_stack([np.interp(a, self._s, self._tan[:, i]) for i in range(3)])
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + _EPS
        vcurv = np.interp(a, self._s, self._vcurv)
        return point, tan, vcurv


class _GateArcPathV101(ArcPath):
    """v10 ArcPath plus gate arc-positions and a gate-spiked contouring-weight profile."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        v_cap: float,
        a_lat_max: float,
        v_min: float,
        gate_pos: NDArray[np.float64],
        w_base: float,
        w_gate: float,
        gate_sigma: float,
        n: int = 600,
    ):
        """Build the v10 arc tables, then project each gate centre onto the path arc length."""
        super().__init__(curve, t_total, v_cap, a_lat_max, v_min, n)
        self._w_base = float(w_base)
        self._w_gate = float(w_gate)
        self._inv2s2 = 1.0 / (2.0 * float(gate_sigma) ** 2)
        # Arc length of each gate = arc of the nearest dense sample to the gate centre.
        gates = np.asarray(gate_pos, dtype=np.float64).reshape(-1, 3)
        self._gate_arcs = np.array(
            [self._s[int(np.argmin(np.linalg.norm(self._pts - g, axis=1)))] for g in gates],
            dtype=np.float64,
        )

    def w_contour(self, arc: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-arc contouring weight: a baseline plus a Gaussian bump at each gate's arc length."""
        a = np.asarray(arc, dtype=np.float64)
        w = np.full(a.shape, self._w_base, dtype=np.float64)
        for s_g in self._gate_arcs:
            w += self._w_gate * np.exp(-((a - s_g) ** 2) * self._inv2s2)
        return w


class _GateArcPathV103(_GateArcPathV101):
    """v10.1's GateArcPath plus a feasibility-repaired speed cap at each gate window."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        v_cap: float,
        a_lat_max: float,
        v_min: float,
        gate_pos: NDArray[np.float64],
        w_base: float,
        w_gate: float,
        gate_sigma: float,
        v_gate: float,
        gate_v_pre: float,
        gate_v_post: float,
        n: int = 600,
    ):
        """Build the v10.1 tables, then cap the speed profile inside each gate window.

        The window is asymmetric: ``[s_gate - gate_v_pre, s_gate + gate_v_post]``. The reveal
        correction must be absorbed on the APPROACH (between the 0.7 m sensor reveal and the
        gate plane), so that side carries the cap; the exit can take speed back immediately.
        The backward pass below shapes the deceleration into the window, so ``gate_v_pre``
        does not need to cover the braking distance.
        """
        super().__init__(
            curve, t_total, v_cap, a_lat_max, v_min, gate_pos, w_base, w_gate, gate_sigma, n
        )
        v_gate, v_cap = float(v_gate), float(v_cap)
        if v_gate >= v_cap:  # cap disabled -> exactly v10.1's profile
            return
        v = self._vcurv.copy()
        for s_g in self._gate_arcs:
            window = (self._s >= s_g - float(gate_v_pre)) & (self._s <= s_g + float(gate_v_post))
            v[window] = np.minimum(v[window], v_gate)
        ds = np.diff(self._s)
        for i in range(len(v) - 2, -1, -1):  # backward: brake in time for each gate window
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_lat_max * ds[i]))
        for i in range(1, len(v)):  # forward: accelerate out within the longitudinal limit
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_lat_max * ds[i - 1]))
        self._vcurv = v


class _GateArcPathV104(_GateArcPathV103):
    """v10.3's GateArcPath with per-gate (reactive) approach speed caps."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        v_cap: float,
        a_lat_max: float,
        v_min: float,
        gate_pos: NDArray[np.float64],
        w_base: float,
        w_gate: float,
        gate_sigma: float,
        gate_caps: NDArray[np.float64],
        react_v_pre: float,
        react_v_post: float,
        n: int = 600,
    ):
        """Build the v10.3 tables (global cap disabled), then cap the flagged gates' windows.

        ``gate_caps`` has one entry per row of ``gate_pos``: the approach-window speed cap for
        that gate, or np.inf to leave it at the curvature profile.
        """
        super().__init__(
            curve, t_total, v_cap, a_lat_max, v_min, gate_pos, w_base, w_gate, gate_sigma,
            v_gate=1e9, gate_v_pre=0.0, gate_v_post=0.0, n=n,  # permanent cap OFF
        )
        caps = np.asarray(gate_caps, dtype=np.float64).reshape(-1)
        active = [(s_g, c) for s_g, c in zip(self._gate_arcs, caps) if c < float(v_cap)]
        if not active:
            return
        v = self._vcurv.copy()
        for s_g, cap in active:
            window = (self._s >= s_g - float(react_v_pre)) & (self._s <= s_g + float(react_v_post))
            v[window] = np.minimum(v[window], cap)
        ds = np.diff(self._s)
        for i in range(len(v) - 2, -1, -1):  # backward: brake into each window in time
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_lat_max * ds[i]))
        for i in range(1, len(v)):  # forward: accelerate out within the longitudinal limit
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_lat_max * ds[i - 1]))
        self._vcurv = v


class _GateArcPathV105(_GateArcPathV104):
    """v10.4's reactive-cap GateArcPath plus v10.2's band-restricted nearest-point search."""

    def project_near(
        self, pos: NDArray[np.float64], center: float, band: float
    ) -> float:
        """Arc length of the nearest path point within ``+/- band`` of ``center`` (v10.2).

        ``center`` is the solver's predicted progress for this step. Restricting the search to a
        tight band around it keeps the progress anchor glued to the drone (the band is wider than
        the legitimate per-step fold advance) while making it impossible to select a far leg of a
        self-folding path (which lies outside the band) -- so the anchor can no longer teleport
        across a fold and skip a gate.
        """
        lo = int(np.searchsorted(self._s, center - band))
        hi = max(int(np.searchsorted(self._s, center + band)), lo + 1)
        nearest = int(np.argmin(np.linalg.norm(self._pts[lo:hi] - pos, axis=1)))
        return float(self._s[lo + nearest])


def _limit_longitudinal(
    v: NDArray[np.float64], s: NDArray[np.float64], a_max: float
) -> NDArray[np.float64]:
    """Backward (brake in time) and forward (accelerate out) passes over a speed profile."""
    ds = np.diff(s)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_max * ds[i]))
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_max * ds[i - 1]))
    return v


class TunnelArcPath(_GateArcPathV105):
    """v10.5's GateArcPath plus tunnel half-extent tables and all-gate reveal caps."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        settings: MPCCSettings,
        gate_pos: NDArray[np.float64],
        obstacles_pos: NDArray[np.float64],
        posts_pos: NDArray[np.float64],
        gate_is_target_zero: bool,
        n: int = 600,
    ):
        """Build the v10.5 tables, cap every gate's reveal window, and lay the tunnel.

        ``posts_pos`` are the frame-post columns of ALL gates (passed ones included -- the
        frames are physical regardless of target); they clip the tunnel like obstacles but
        with the smaller post margin. ``gate_is_target_zero`` widens gate 0's cap window to
        the launch corridor (v10.4's launch protection) -- True only while gate 0 is the
        current target.
        """
        s = settings
        # Reveal caps via the inherited v10.4 cap machinery: every remaining gate when
        # reveal_cap_all is set, otherwise none here (gate 0's launch cap is applied below).
        caps = np.full(len(gate_pos), float(s.v_gate_reveal) if s.reveal_cap_all else np.inf)
        super().__init__(
            curve,
            t_total,
            s.v_theta_max,
            s.a_lat_max,
            s.v_min,
            gate_pos,
            s.w_contour_base,
            s.w_contour_gate,
            s.gate_sigma,
            caps,
            s.reveal_pre_m,
            s.reveal_post_m,
            n,
        )
        if gate_is_target_zero and len(self._gate_arcs):
            v = self._vcurv.copy()
            s_g0 = self._gate_arcs[0]
            window = (self._s >= s_g0 - s.launch_cap_pre_m) & (self._s <= s_g0 + s.reveal_post_m)
            v[window] = np.minimum(v[window], s.v_gate_reveal)
            self._vcurv = _limit_longitudinal(v, self._s, s.a_lat_max)

        # --- Tunnel tables over the dense samples ---
        tangent = self._tan
        lat = np.column_stack([tangent[:, 1], -tangent[:, 0], np.zeros(len(tangent))])
        norms = np.linalg.norm(lat, axis=1)
        # Near-vertical tangents (launch climb) have no horizontal normal: carry the last
        # valid one forward (world-x before any valid sample exists).
        lat[norms < 0.2] = np.nan
        lat[0] = lat[0] if norms[0] >= 0.2 else np.array([1.0, 0.0, 0.0])
        for i in range(1, len(lat)):
            if np.isnan(lat[i, 0]):
                lat[i] = lat[i - 1]
        lat /= np.linalg.norm(lat, axis=1, keepdims=True) + _EPS

        # Curvature clamp: the OCP's linearised reference is only trustworthy within
        # ~1/kappa of the path. kappa is recovered from the (uncapped-by-windows) curvature
        # profile: kappa = a_lat / v_curv^2.
        kappa = s.a_lat_max / np.maximum(self._vcurv, 1e-3) ** 2
        w = np.minimum(
            np.full(len(self._s), float(s.tunnel_w_max)),
            np.maximum(float(s.tunnel_curv_frac) / np.maximum(kappa, 1e-6), s.tunnel_w_min),
        )
        clip_sets = (
            (np.asarray(obstacles_pos, dtype=np.float64).reshape(-1, 3), s.tunnel_obs_margin),
            (np.asarray(posts_pos, dtype=np.float64).reshape(-1, 3), s.tunnel_post_margin),
        )
        for points, margin in clip_sets:
            for point in points:
                d = np.linalg.norm(self._pts[:, :2] - point[:2], axis=1)
                w = np.minimum(w, np.maximum(d - margin, s.tunnel_w_min))
        h = np.minimum(
            np.full(len(self._s), float(s.tunnel_h_max)),
            np.maximum(
                np.minimum(self._pts[:, 2] - s.tunnel_z_floor, s.tunnel_z_ceil - self._pts[:, 2]),
                s.tunnel_w_min,
            ),
        )
        for s_g in self._gate_arcs:  # narrow to the opening into each gate
            ramp = np.minimum(np.abs(self._s - s_g) / float(s.tunnel_taper_m), 1.0)
            w = np.minimum(w, s.tunnel_w_gate + (s.tunnel_w_max - s.tunnel_w_gate) * ramp)
            h = np.minimum(h, s.tunnel_h_gate + (s.tunnel_h_max - s.tunnel_h_gate) * ramp)
        self._lat, self._w, self._h = lat, w, h
        self._a_lat_max_v11 = float(s.a_lat_max)  # for subclasses re-running the passes

        # --- Fix 4: gate+obstacle precision zone ---
        # A gate with a real obstacle within gate_obstacle_radius is flown slower (v_gate_obstacle),
        # tracked tighter (extra contouring weight, see w_contour), and through a narrower tunnel
        # (tunnel_w_obstacle) -- precision over speed where a crash is otherwise likely.
        obs = np.asarray(obstacles_pos, dtype=np.float64).reshape(-1, 3)
        gates = np.asarray(gate_pos, dtype=np.float64).reshape(-1, 3)
        self._w_contour_obs = float(s.w_contour_obstacle)
        self._obs_gate_arcs = np.empty(0, dtype=np.float64)
        if len(obs) and len(gates) and len(self._gate_arcs):
            ng = min(len(gates), len(self._gate_arcs))
            near = np.array(
                [
                    float(np.min(np.linalg.norm(obs[:, :2] - gates[i, :2], axis=1)))
                    < float(s.gate_obstacle_radius)
                    for i in range(ng)
                ]
            )
            self._obs_gate_arcs = self._gate_arcs[:ng][near]
        if len(self._obs_gate_arcs):
            v = self._vcurv.copy()
            for s_g in self._obs_gate_arcs:  # slow: cap the approach + exit window
                win = (self._s >= s_g - float(s.reveal_pre_m)) & (
                    self._s <= s_g + float(s.reveal_post_m)
                )
                v[win] = np.minimum(v[win], float(s.v_gate_obstacle))
            self._vcurv = _limit_longitudinal(v, self._s, float(s.a_lat_max))
            for s_g in self._obs_gate_arcs:  # tighten: narrow the tunnel around these gates
                ramp = np.minimum(np.abs(self._s - s_g) / float(s.tunnel_taper_m), 1.0)
                tight = float(s.tunnel_w_obstacle)
                self._w = np.minimum(self._w, tight + (float(s.tunnel_w_max) - tight) * ramp)

    def w_contour(self, arc: NDArray[np.float64]) -> NDArray[np.float64]:
        """v10.1's gate-spiked contouring weight plus a fix-4 bump at gate+obstacle gates."""
        w = super().w_contour(arc)
        if len(self._obs_gate_arcs):
            a = np.asarray(arc, dtype=np.float64)
            for s_g in self._obs_gate_arcs:
                w = w + self._w_contour_obs * np.exp(-((a - s_g) ** 2) * self._inv2s2)
        return w

    def tunnel(
        self, arc: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return (lateral unit basis, half-width, half-height) at each arc length."""
        a = np.clip(np.asarray(arc, dtype=np.float64), 0.0, self.total)
        lat = np.column_stack([np.interp(a, self._s, self._lat[:, i]) for i in range(3)])
        lat /= np.linalg.norm(lat, axis=1, keepdims=True) + _EPS
        return lat, np.interp(a, self._s, self._w), np.interp(a, self._s, self._h)


class CappedTunnelArcPath(TunnelArcPath):
    """v11's TunnelArcPath with the smoothed plan's parity caps folded into the profile."""

    def __init__(
        self,
        *args,  # noqa: ANN002 -- forwarded verbatim to TunnelArcPath
        gate_window_caps: NDArray[np.float64] | None = None,
        window_pre: float = 0.7,
        obstacle_caps: NDArray[np.float64] | None = None,
        **kwargs,  # noqa: ANN003
    ):
        """Build the v11 tables, then apply the plan's parity caps (np.inf = inactive)."""
        super().__init__(*args, **kwargs)
        v = self._vcurv.copy()
        bound = False
        if gate_window_caps is not None:
            for s_g, cap in zip(self._gate_arcs, np.asarray(gate_window_caps, dtype=np.float64)):
                if np.isfinite(cap):
                    mask = (self._s >= s_g - float(window_pre)) & (self._s <= s_g)
                    v[mask] = np.minimum(v[mask], cap)
                    bound = True
        if obstacle_caps is not None:
            for s_lo, s_hi, cap in np.asarray(obstacle_caps, dtype=np.float64).reshape(-1, 3):
                if np.isfinite(cap):
                    mask = (self._s >= s_lo) & (self._s <= s_hi)
                    v[mask] = np.minimum(v[mask], cap)
                    bound = True
        if bound:
            self._vcurv = _limit_longitudinal(v, self._s, float(self._a_lat_max_v11))
