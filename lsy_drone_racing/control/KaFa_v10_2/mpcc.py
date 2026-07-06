"""Gate-aware time-optimal MPCC core for KaFa_1500_v10_2.

The OCP is byte-for-byte v10.1's -- v10.2 changes only how progress is ANCHORED to the drone, not
the solver. This subclass adds one read-only accessor, ``predicted_progress()``, returning the
solver's own dynamics-feasible progress one step ahead (the th state at stage 1 of the last
solution). The v10.2 controller anchors the next step's progress to that value instead of to a raw
geometric nearest-point search, because the predicted progress advances at the bounded rate vth and
so can never teleport across a self-folding path -- which is what causes the slalom gate-skip under
v10.1's pure geometric anchor. See KaFa_1500_v10_2 for the full rationale.
"""

from __future__ import annotations

from lsy_drone_racing.control.KaFa_v10_1.mpcc import MPCC as _MPCC

__all__ = ["MPCC"]


class MPCC(_MPCC):
    """v10.1's MPCC plus a read-only accessor for the solver's predicted next-step progress."""

    def predicted_progress(self) -> float | None:
        """Return the solver's progress one step ahead (th at stage 1), or None before any solve.

        This is the dynamics-feasible progress the OCP itself predicts for the next control step;
        because it evolves at the bounded progress rate vth, it cannot jump across a fold the way a
        raw geometric nearest-point search can. The controller uses it as the centre of a tight
        geometric correction so the progress anchor tracks the drone without ever teleporting.
        """
        if self._x_sol is None:
            return None
        return float(self._x_sol[6, 1])
