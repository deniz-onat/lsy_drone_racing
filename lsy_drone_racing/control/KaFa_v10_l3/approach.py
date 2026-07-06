"""Build the STAGE approach path for KaFa_1500_v10_l3 (search -> navigate hand-off).

The search sweep ends at an arbitrary pose the instant the last gate is detected. On level3
the gate yaw is randomised and the env counts gate 0 only when crossed in its canonical +x
direction, so when the sweep happens to end on the **exit (+x) side** of gate 0 the race
planner has to reverse back through/around the gate to reach the entry side -- a cusp the MPCC
crashes on (it clips the frame or hairpins). This was the dominant search->navigate failure.

STAGE fixes it geometrically: when the hand-off pose is on the wrong side of gate 0, fly a short
arena-clear curve that loops the drone around to **behind gate 0's entry**, then race. The whole
detour stays at the safe sweep altitude (detection/clearance is horizontal, obstacles sit below),
so it adds no new collision exposure and leaves the descent to the (already tuned) NAVIGATE phase.

``build_approach_curve`` returns ``None`` when no staging is needed (the drone is already on the
entry side), so a clean hand-off races directly as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.KaFa_v10_l3.cockpit import SearchSettings


def build_approach_curve(
    pos: NDArray[np.float64],
    gate_pos: NDArray[np.float64],
    gate_quat: NDArray[np.float64],
    cfg: SearchSettings,
) -> tuple[CubicSpline, float, NDArray[np.float64]] | None:
    """Return (curve, t_total, entry_xy) for the gate-0 staging detour, or None if not needed.

    ``gate_quat`` gives gate 0's canonical +x (the required crossing direction). If the drone is
    on the entry (-x) side already, no detour is built (returns None). Otherwise the curve runs
    pos -> a frame-clearing side point -> the entry point behind the gate, all at the sweep
    altitude. ``entry_xy`` is the XY hand-off target so the caller can detect arrival.
    """
    pos = np.asarray(pos, dtype=np.float64)
    gate_pos = np.asarray(gate_pos, dtype=np.float64)
    rot = Rotation.from_quat(np.asarray(gate_quat, dtype=np.float64)).as_matrix()
    fwd = rot[:, 0].copy()      # canonical +x: required crossing direction
    lat = rot[:, 1].copy()      # gate lateral (+y)
    fwd[2] = lat[2] = 0.0       # work in the horizontal plane (sweep altitude is fixed)
    fwd /= np.linalg.norm(fwd) + 1e-9
    lat /= np.linalg.norm(lat) + 1e-9

    along = float(np.dot(pos - gate_pos, fwd))  # >0 => drone on the exit (+x) side -> bad
    if along <= cfg.stage_min_along:
        return None  # already on the entry side: race directly, current behaviour is fine

    alt = float(pos[2])
    side = float(np.sign(np.dot(pos - gate_pos, lat))) or 1.0
    entry = gate_pos - cfg.stage_d_entry * fwd        # hand-off target, behind the gate
    side_pt = gate_pos + side * cfg.stage_side_w * lat  # clears the frame on the drone's side
    pts = np.array(
        [
            [pos[0], pos[1], alt],
            [side_pt[0], side_pt[1], alt],
            [entry[0], entry[1], alt],
        ],
        dtype=np.float64,
    )
    taus = np.arange(len(pts), dtype=np.float64)
    curve = CubicSpline(taus, pts)
    return curve, float(taus[-1]), np.array([entry[0], entry[1]], dtype=np.float64)
