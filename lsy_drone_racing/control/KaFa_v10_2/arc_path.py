"""Gate-aware arc-length view of a plan spline with a BOUNDED-CORRECTION projection (v10.2).

Subclasses v10.1's GateArcPath (dense arc tables + gate arc-positions + gate-spiked contouring
weight, all reused unchanged) and adds one method, ``project_near``: the nearest-point search is
restricted to a tight arc-length band around a supplied CENTRE (the solver's predicted progress),
instead of v10.1's global nearest point over a 2.0 m forward window.

Why: on a sharp slalom the planned path folds back on itself near a gate (the steep-angle crossing
plus the d_pre/d_post run-in/out place the approach/exit waypoints so the spline overshoots into a
cusp). v10.1 anchors progress at the GLOBAL nearest path point; when the drone is near such a fold,
a far-along-the-arc leg of the cusp is spatially closer than the apex, so the anchor snaps ~1-2 m
forward across the fold in a single step and the drone skips the gate. Constraining that *forward*
search alone does not work -- at the fold the legitimate foot-point genuinely advances ~0.7 m/step
as the drone rounds the doubled-over path, so any cap tight enough to block the skip also stalls the
legitimate motion (verified: every forward-window variant regressed, one catastrophically).

The robust cure is to centre the search on the solver's own predicted progress (see
KaFa_v10_2.mpcc.MPCC.predicted_progress), which is dynamics-feasible and cannot teleport, and only
let geometry correct it within +/- a band sized just above the legitimate fold rate. The far fold
leg lies outside that band, so it can never be selected; the apex motion lies inside it, so tracking
is unaffected. This keeps v10.1's behaviour everywhere except the fold, where it removes the skip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.KaFa_v10_1.arc_path import GateArcPath as _GateArcPath

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GateArcPath(_GateArcPath):
    """v10.1 GateArcPath plus a nearest-point search restricted to a band around a given centre."""

    def project_near(
        self, pos: NDArray[np.float64], center: float, band: float
    ) -> float:
        """Arc length of the nearest path point within ``+/- band`` of ``center``.

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
