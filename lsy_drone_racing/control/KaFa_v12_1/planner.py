"""Reference planning for KaFa_1500_v12_1: guarded-smoothed, parity-capped global spline.

Consolidates v11_1's reference-planning stack into one module:

* the gate-funnelled, obstacle-aware global waypoint chain (v10.4's trimmed-clearance geometry
  over v8's planner) and its clamped-cubic-spline timing / obstacle repair (v8 timing);
* v10.6's GUARDED waypoint smoothing with parity speed caps (``SmoothedPlan``), adopted once per
  episode behind the crossings/clearance/gain guard;
* ``ReferenceManager`` -- the episode-sticky planner the controller drives. Its live behaviour is
  v10.6's ``build``/``__init__``/``reset`` over v8's ``ensure_plan``/``_needs_plan``/
  ``_planning_obstacles``; the intermediate v8/v10.4 ``build`` methods were shadowed and are not
  carried here.

This is a behaviour-preserving consolidation of KaFa_v8.trajectory + KaFa_v8.timing +
KaFa_v10_4.trajectory + KaFa_v10_6.trajectory, verified to produce bit-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.KaFa_v12_1.avoidance import (
    nudge_lateral,
    push_off_obstacles,
    reversal_turn,
)
from lsy_drone_racing.control.KaFa_v12_1.speed_profile import SpeedProfile

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.KaFa_v12_1.observation import DroneObservation
    from lsy_drone_racing.control.KaFa_v12_1.settings import PlannerSettings, SmoothPlannerSettings

ReferenceCurve = CubicSpline

_FRAME_HALF_M = 0.36  # gate frame outer half-extent (0.72 m square)

# --- Obstacle-handling rules (v12_1) ---
# An obstacle within GATE_PROXIMITY_RADIUS (XY) of any gate is handled by shifting the gate's
# existing waypoints (the nudge / push-off in build_waypoints) rather than inserting a new
# avoidance waypoint for it (Rule 1).
GATE_PROXIMITY_RADIUS = 0.4
# Two obstacles closer than OBSTACLE_MERGE_DIST (XY) are merged into a single obstacle at their
# midpoint, kept at the normal obstacle radius (r_obs) (Rule 2).
OBSTACLE_MERGE_DIST = 0.2
# A spline that bulges more than OVERSHOOT_MAX_DEV (m) from the straight waypoint polyline in
# free space gets a corrective chord point inserted to pull it back (P2 overshoot straightening).
OVERSHOOT_MAX_DEV = 0.20
# Fix 3-1: only trim "behind the drone" leading waypoints when the drone is moving faster than
# this (m/s); below it (cold start / standing reversal) backward points can be legitimate.
BACKWARD_CLIP_MIN_SPEED = 0.5


@dataclass(frozen=True)
class ReferencePlan:
    """A global plan through all remaining gates from a fixed start state."""

    curve: CubicSpline
    t_total: float
    waypoints: NDArray[np.float64]
    gate_pos_snapshot: NDArray[np.float64]
    obstacle_pos_snapshot: NDArray[np.float64]
    built_target_gate: int


@dataclass(frozen=True)
class SmoothedPlan(ReferencePlan):
    """A reference plan plus the parity speed caps that make the smoothed route shippable.

    ``gate_window_caps`` has one entry per remaining gate (np.inf = no cap); ``obstacle_caps``
    one (s_lo, s_hi, v_max) arc-interval row per obstacle passage of this plan's own curve.
    Both are np.inf/empty when ``smoothed`` is False (the plan is byte-identical to v10.4's).
    """

    gate_window_caps: NDArray[np.float64]
    obstacle_caps: NDArray[np.float64]
    smoothed: bool


def _merge_close_obstacles(
    obstacles: NDArray[np.float64], merge_dist: float = OBSTACLE_MERGE_DIST
) -> NDArray[np.float64]:
    """Merge obstacles closer than ``merge_dist`` (XY) into one obstacle at their midpoint.

    Tightly clustered obstacles otherwise produce overlapping keep-out nudges/repairs that
    destabilise the spline. The merged point is returned as an ordinary obstacle position; the
    planner applies the normal ``r_obs`` radius to every obstacle, so a merged pair is treated
    exactly like a single obstacle (Rule 2). Greedy pairwise merge, repeated until no pair is
    closer than ``merge_dist``; midpoints are 3-D averages.
    """
    pts = [o for o in np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)]
    merged = True
    while merged and len(pts) > 1:
        merged = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if float(np.linalg.norm(pts[i][:2] - pts[j][:2])) < merge_dist:
                    mid = 0.5 * (pts[i] + pts[j])
                    pts = [p for k, p in enumerate(pts) if k not in (i, j)]
                    pts.append(mid)
                    merged = True
                    break
            if merged:
                break
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def gate_post_obstacles(
    gates_pos: NDArray[np.float64],
    gates_quat: NDArray[np.float64],
    target_gate: int,
    offset: float,
) -> NDArray[np.float64]:
    """Return two virtual column positions per remaining gate (the gate-post funnels).

    Columns sit at +/-offset along each gate's lateral axis so the planner routes the spline
    through the opening centre. Only gates from target_gate onward are funnelled, since
    already-passed gates sit behind the start and would just perturb the spline's first
    segment. The planner's avoidance is 2-D, so the column z is set to the gate-centre
    height for visual clarity only.
    """
    gates_pos = np.asarray(gates_pos, dtype=np.float64)
    gates_quat = np.asarray(gates_quat, dtype=np.float64)
    posts: list[NDArray[np.float64]] = []
    for gate_index in range(max(target_gate, 0), len(gates_pos)):
        gate_pos = gates_pos[gate_index]
        lateral = Rotation.from_quat(gates_quat[gate_index]).as_matrix()[:, 1]
        posts.append(gate_pos + offset * lateral)
        posts.append(gate_pos - offset * lateral)
    if not posts:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(posts, dtype=np.float64)


def _oriented_forward(
    quat: NDArray[np.float64],
    gate_pos: NDArray[np.float64],
    reference: NDArray[np.float64],
    flip_to_travel: bool = True,
) -> NDArray[np.float64]:
    """Return the gate forward axis.

    When flip_to_travel is True the axis is flipped to point along the travel direction
    (from reference toward the gate). When False the gate's canonical +x axis is returned
    unchanged, which is the direction the env requires the gate to be crossed in
    (gate-local -x -> +x).
    """
    forward = Rotation.from_quat(quat).as_matrix()[:, 0]
    if flip_to_travel and float(np.dot(forward, gate_pos - reference)) < 0.0:
        forward = -forward
    return forward


def _clearance_points(
    prev_pos: NDArray[np.float64],
    prev_forward: NDArray[np.float64],
    next_pos: NDArray[np.float64],
    next_forward: NDArray[np.float64],
    settings: PlannerSettings,
) -> list[NDArray[np.float64]]:
    """v8's clearance/turn-apex insertion with the v10.4 audited trims (see ReferenceManager)."""
    next_approach = next_pos - settings.d_pre * next_forward
    next_z = float(next_approach[2])
    prev_z = float(prev_pos[2])
    if abs(next_z - prev_z) <= settings.clearance_height_delta:
        return []
    to_next = (next_pos - prev_pos)[:2]
    pf = prev_forward[:2]
    denom = float(np.linalg.norm(pf) * np.linalg.norm(to_next))
    cos_to_next = float(np.dot(pf, to_next)) / denom if denom > 1e-9 else 1.0
    if cos_to_next < -0.3:  # clear reversal: keep v8's wide U-turn swing (necessary geometry)
        exit_xy = (prev_pos + settings.d_post * prev_forward)[:2]
        return reversal_turn(exit_xy, prev_pos, prev_forward, next_approach, to_next, settings)
    # Forward turn: the straight climbing run-out is only useful when the next gate actually
    # lies ahead -- scale the extension by the alignment instead of a fixed 0.60 m, dialled by
    # clr_ext_min (0 = full trim, 1 = v8's fixed 0.60). (The two other audited trims --
    # dropping the climb-z overshoot and the 0.10 m apex push -- were measured TOO HOT in
    # flight: paired eval 13/20 with gate-3 U-turn fails and extra launch crashes; the cubic
    # spline is global, so they also reshape neighbouring segments. Keep v8's values for both.)
    extension = 0.60 * max(cos_to_next, float(getattr(settings, "clr_ext_min", 1.0)))
    clearance_xy = (prev_pos + (settings.d_post + extension) * prev_forward)[:2]
    if next_z > prev_z:
        clearance_z = max(prev_z + 0.55, next_z - 0.05)
        apex_z = next_z - 0.05
    else:
        clearance_z = max(prev_z - 0.30, next_z + 0.15)
        apex_z = next_z + 0.05
    mid_xy = 0.5 * (clearance_xy + next_approach[:2])
    away = mid_xy - prev_pos[:2]
    away_norm = float(np.linalg.norm(away))
    if away_norm > 1e-6:
        mid_xy = mid_xy + (away / away_norm) * 0.10
    return [
        np.array([clearance_xy[0], clearance_xy[1], clearance_z]),
        np.array([mid_xy[0], mid_xy[1], apex_z]),
    ]


def _insert_exited_clearance(
    waypoints: list[NDArray[np.float64]],
    start_pos: NDArray[np.float64],
    start_vel: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    gates_quat: NDArray[np.float64],
    target_gate: int,
    settings: PlannerSettings,
) -> None:
    """v8's post-gate replan arc, byte-for-byte, using the trimmed _clearance_points."""
    prev_pos = gates_pos[target_gate - 1]
    if float(np.linalg.norm(start_pos - prev_pos)) >= 0.6:
        return
    prev_forward = Rotation.from_quat(gates_quat[target_gate - 1]).as_matrix()[:, 0]
    if settings.orient_gates_to_travel:
        travel = np.array([start_vel[0], start_vel[1], 0.0])
        if float(np.linalg.norm(travel)) > 0.1:
            reference = float(np.dot(prev_forward, travel))
        else:
            reference = float(np.dot(prev_forward, gates_pos[target_gate] - prev_pos))
        if reference < 0.0:
            prev_forward = -prev_forward
    prev_exit = prev_pos + settings.d_post * prev_forward
    next_forward = _oriented_forward(
        gates_quat[target_gate], gates_pos[target_gate], prev_exit, settings.orient_gates_to_travel
    )
    points = _clearance_points(
        prev_pos, prev_forward, gates_pos[target_gate], next_forward, settings
    )
    for offset, point in enumerate(points):
        waypoints.insert(1 + offset, point)


def build_waypoints(
    start_pos: NDArray[np.float64],
    start_vel: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    gates_quat: NDArray[np.float64],
    obstacles_pos: NDArray[np.float64],
    target_gate: int,
    settings: PlannerSettings,
) -> NDArray[np.float64]:
    """v8's obstacle-aware waypoint chain, byte-for-byte, over the trimmed clearance geometry."""
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_vel = np.asarray(start_vel, dtype=np.float64)
    gates_pos = np.asarray(gates_pos, dtype=np.float64)
    gates_quat = np.asarray(gates_quat, dtype=np.float64)
    remaining_pos = gates_pos[target_gate:]
    remaining_quat = gates_quat[target_gate:]
    obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)

    waypoints: list[NDArray[np.float64]] = [start_pos.copy()]
    if start_pos[2] < settings.liftoff_z_threshold and len(remaining_pos) > 0:
        target_z = max(settings.liftoff_height, 0.85 * float(remaining_pos[0][2]))
        toward = remaining_pos[0][:2] - start_pos[:2]
        distance = float(np.linalg.norm(toward))
        offset = toward / distance * 0.40 if distance > 1e-6 else np.zeros(2)
        waypoints.append(np.array([start_pos[0] + offset[0], start_pos[1] + offset[1], target_z]))
    elif target_gate > 0 and len(remaining_pos) > 0:
        _insert_exited_clearance(
            waypoints, start_pos, start_vel, gates_pos, gates_quat, target_gate, settings
        )

    # Fix 3-1: when the drone is moving, drop leading waypoints that sit BEHIND its motion. A
    # replan after the drone advanced past a gate can insert a clearance point behind it; since
    # the MPCC tracks increasing arc length, such a point would pull the drone backward. We trim
    # only the contiguous leading block (the pre-gate liftoff/continuity points); the forward gate
    # sequence and later reversal swings are untouched.
    speed = float(np.linalg.norm(start_vel))
    if speed > BACKWARD_CLIP_MIN_SPEED:
        fwd = np.asarray(start_vel, dtype=np.float64) / speed
        while len(waypoints) > 1 and float(np.dot(waypoints[1] - waypoints[0], fwd)) < -0.02:
            waypoints.pop(1)

    n_gates = len(remaining_pos)
    forwards: list[NDArray[np.float64]] = []
    reference = waypoints[-1]
    for index in range(n_gates):
        gate_pos = remaining_pos[index]
        fwd = _oriented_forward(
            remaining_quat[index], gate_pos, reference, settings.orient_gates_to_travel
        )
        forwards.append(fwd)
        reference = gate_pos + settings.d_post * fwd
    gate_indices: set[int] = set()
    for index in range(n_gates):
        gate_pos = remaining_pos[index]
        forward = forwards[index]
        lateral = Rotation.from_quat(remaining_quat[index]).as_matrix()[:, 1]
        previous = waypoints[-1]
        approach_raw = gate_pos - settings.d_pre * forward
        exit_raw = gate_pos + settings.d_post * forward
        lateral_bias = float(np.dot((previous - approach_raw)[:2], lateral[:2]))
        bias_sign = float(np.sign(lateral_bias)) if abs(lateral_bias) > 1e-3 else 0.0
        approach = nudge_lateral(approach_raw, lateral, obstacles_pos, settings.r_obs, bias_sign)
        exit_point = nudge_lateral(exit_raw, lateral, obstacles_pos, settings.r_obs)
        far_raw = gate_pos - (settings.d_pre + 0.30) * forward
        far_approach = nudge_lateral(far_raw, lateral, obstacles_pos, settings.r_obs, bias_sign)
        behind = float(np.dot(previous - far_approach, forward)) < 0.0
        if behind and float(np.linalg.norm((far_approach - previous)[:2])) > 0.20:
            waypoints.append(far_approach)
        waypoints.extend([approach, gate_pos.copy(), exit_point])
        gate_indices.add(len(waypoints) - 2)
        if index + 1 < n_gates:
            waypoints.extend(
                _clearance_points(
                    gate_pos, forward, remaining_pos[index + 1], forwards[index + 1], settings
                )
            )

    final_post = settings.d_post + settings.d_stop
    waypoints.append(remaining_pos[-1] + final_post * forwards[-1])
    return push_off_obstacles(
        np.asarray(waypoints), gate_indices, obstacles_pos, settings.r_obs + 0.06
    )


def obstacle_slowdown(
    waypoints: NDArray[np.floating],
    segment_times: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    slow_radius: float = 0.32,
) -> NDArray[np.floating]:
    """Stretch segment durations that pass close to an obstacle for safer tracking."""
    if len(obstacles_pos) == 0:
        return segment_times
    segment_times = np.asarray(segment_times, dtype=np.float64).copy()
    for index in range(len(waypoints) - 1):
        start = waypoints[index, :2]
        end = waypoints[index + 1, :2]
        segment = end - start
        segment_sq = float(np.dot(segment, segment))
        if segment_sq < 1e-9:
            continue
        min_distance = np.inf
        for obstacle in obstacles_pos:
            ratio = float(np.clip(np.dot(obstacle[:2] - start, segment) / segment_sq, 0.0, 1.0))
            closest = start + ratio * segment
            min_distance = min(min_distance, float(np.linalg.norm(obstacle[:2] - closest)))
        if min_distance < slow_radius:
            segment_times[index] *= 1.0 + 0.6 * (slow_radius - min_distance) / slow_radius
    return segment_times


def turn_slowdown(
    waypoints: NDArray[np.floating],
    segment_times: NDArray[np.floating],
    min_sharpness: float,
    slow_gain: float,
) -> NDArray[np.floating]:
    """Stretch the segments around sharp corners so reversals/hairpins are flown slower.

    Corners sharper than ``min_sharpness`` are slowed in proportion to sharpness, so the
    tracker can follow the tight U-turns without overshooting into a frame or obstacle.
    """
    segment_times = np.asarray(segment_times, dtype=np.float64).copy()
    diffs = np.diff(waypoints, axis=0)
    for index in range(1, len(waypoints) - 1):
        before, after = diffs[index - 1], diffs[index]
        n_before, n_after = float(np.linalg.norm(before)), float(np.linalg.norm(after))
        if n_before < 1e-6 or n_after < 1e-6:
            continue
        cos_angle = float(np.dot(before, after) / (n_before * n_after))
        sharpness = (1.0 - cos_angle) / 2.0  # 0 straight, 1 reversal
        if sharpness <= min_sharpness:
            continue
        factor = 1.0 + slow_gain * sharpness
        segment_times[index - 1] *= factor
        segment_times[index] *= factor
    return segment_times


def build_spline(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    settings: PlannerSettings,
) -> tuple[NDArray[np.floating], CubicSpline]:
    """Time-parameterize the waypoints into a clamped cubic spline."""
    start_vel = np.asarray(start_vel, dtype=np.float64)
    gates_pos = np.asarray(gates_pos, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    inter_speed = settings.v_cruise_inter if settings.v_cruise_inter > 0 else settings.v_cruise
    cold_start = float(np.linalg.norm(start_vel)) < 0.3
    # A negative start_vel.z at low altitude clamps the BC to a downward slope and
    # drives the spline below z=0.  Force the z-component non-negative whenever the
    # drone is near the floor and nearly stationary so the spline opens upward.
    if cold_start and float(waypoints[0, 2]) <= settings.liftoff_z_threshold:
        start_vel = start_vel.copy()
        start_vel[2] = max(float(start_vel[2]), 0.0)
    segment_times = np.empty(len(segment_lengths), dtype=np.float64)
    for index in range(len(segment_lengths)):
        peri = any(
            float(np.linalg.norm(waypoints[index, :2] - gate[:2])) < settings.peri_gate_radius
            or float(np.linalg.norm(waypoints[index + 1, :2] - gate[:2]))
            < settings.peri_gate_radius
            for gate in gates_pos
        )
        speed = settings.v_cruise if peri else inter_speed
        segment_times[index] = max(segment_lengths[index] / speed, settings.t_min_seg)
        if cold_start and index < 2:
            segment_times[index] = max(segment_times[index], settings.cold_start_min_seg)
    segment_times = obstacle_slowdown(waypoints, segment_times, obstacles_pos)
    segment_times = turn_slowdown(
        waypoints, segment_times, settings.turn_min_sharpness, settings.turn_slow_gain
    )
    segment_times = _cap_peak_velocity(
        waypoints, segment_times, start_vel, settings.max_speed, skip=2
    )
    knot_times = np.concatenate([[0.0], np.cumsum(segment_times)])
    bc = ((1, start_vel), (1, np.zeros(3)))
    return knot_times, CubicSpline(knot_times, waypoints, bc_type=bc)


def _cap_peak_velocity(
    waypoints: NDArray[np.floating],
    segment_times: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    max_speed: float,
    skip: int = 0,
) -> NDArray[np.floating]:
    """Stretch segments whose spline velocity exceeds ``max_speed`` (tames overshoot)."""
    bc = ((1, np.asarray(start_vel, dtype=np.float64)), (1, np.zeros(3)))
    for _ in range(4):
        knot_times = np.concatenate([[0.0], np.cumsum(segment_times)])
        velocity = CubicSpline(knot_times, waypoints, bc_type=bc).derivative(1)
        worst = 1.0
        for index in range(skip, len(segment_times)):
            sample = np.linspace(knot_times[index], knot_times[index + 1], 8)
            peak = float(np.max(np.linalg.norm(velocity(sample), axis=1)))
            if peak > max_speed:
                ratio = peak / max_speed
                segment_times[index] *= ratio
                worst = max(worst, ratio)
        if worst <= 1.02:
            break
    return segment_times


def _repair_obstacle_pass(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    settings: PlannerSettings,
    skip_gate_near: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating], CubicSpline]:
    """Insert push-out waypoints until the sampled spline clears every obstacle."""
    knot_times, curve = build_spline(waypoints, start_vel, gates_pos, obstacles_pos, settings)
    if len(obstacles_pos) == 0:
        return waypoints, knot_times, curve
    margin = settings.r_obs + 0.12
    # Rule 1: an obstacle within GATE_PROXIMITY_RADIUS of any gate is left to the gate-waypoint
    # shift (nudge / push-off in build_waypoints); it gets NO inserted avoidance waypoint here.
    # The P3 conservative fallback sets skip_gate_near=False to repair every obstacle.
    gates_xy = np.asarray(gates_pos, dtype=np.float64).reshape(-1, 3)[:, :2]
    if skip_gate_near:
        repair_targets = [
            o
            for o in obstacles_pos
            if len(gates_xy) == 0
            or float(np.min(np.linalg.norm(gates_xy - np.asarray(o, dtype=np.float64)[:2], axis=1)))
            >= GATE_PROXIMITY_RADIUS
        ]
    else:
        repair_targets = list(obstacles_pos)
    for _ in range(6):
        sample_t = np.linspace(0.0, float(knot_times[-1]), max(60, 20 * len(waypoints)))
        points = np.asarray(curve(sample_t), dtype=np.float64)
        worst_index, worst_distance, worst_obstacle = -1, settings.r_obs, None
        for obstacle in repair_targets:
            distances = np.linalg.norm(points[:, :2] - obstacle[:2], axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) < worst_distance:
                worst_index, worst_distance, worst_obstacle = (
                    nearest,
                    float(distances[nearest]),
                    obstacle,
                )
        if worst_obstacle is None:
            break
        violation = points[worst_index]
        direction = violation[:2] - worst_obstacle[:2]
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
        repair = np.array(
            [
                worst_obstacle[0] + direction[0] * margin,
                worst_obstacle[1] + direction[1] * margin,
                float(violation[2]),
            ]
        )
        segment = int(
            np.clip(np.searchsorted(knot_times, sample_t[worst_index]), 1, len(waypoints))
        )
        waypoints = np.insert(waypoints, segment, repair, axis=0)
        knot_times, curve = build_spline(waypoints, start_vel, gates_pos, obstacles_pos, settings)
    return waypoints, knot_times, curve


def _straighten_overshoot(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    settings: PlannerSettings,
    max_dev: float = OVERSHOOT_MAX_DEV,
    passes: int = 4,
) -> tuple[NDArray[np.floating], NDArray[np.floating], CubicSpline]:
    """P2: pull the cubic back toward the straight waypoint polyline where it bulges in free space.

    The path is a TIME-parameterized clamped cubic, so knot timing can make it bulge between
    clear waypoints (and clip a gate/obstacle). For each segment whose curve deviates more than
    ``max_dev`` from its chord, insert the chord point at the worst-deviation location, which
    constrains the curve back toward the straight line. Deliberate curvature is preserved: a
    chord point is skipped if it lies within r_obs of an obstacle (an obstacle detour) or within
    peri_gate_radius of a gate (gate-threading geometry).
    """
    knot_times, curve = build_spline(waypoints, start_vel, gates_pos, obstacles_pos, settings)
    gates_xy = np.asarray(gates_pos, dtype=np.float64).reshape(-1, 3)[:, :2]
    obs = np.asarray(obstacles_pos, dtype=np.float64).reshape(-1, 3)
    for _ in range(passes):
        best_seg, best_dev, best_pt = -1, max_dev, None
        for i in range(len(waypoints) - 1):
            a, b = waypoints[i], waypoints[i + 1]
            ab = b - a
            length = float(np.linalg.norm(ab))
            if length < 1e-6:
                continue
            cpts = np.asarray(curve(np.linspace(knot_times[i], knot_times[i + 1], 12)), np.float64)
            frac = np.clip((cpts - a) @ ab / (length * length), 0.0, 1.0)
            chord = a + frac[:, None] * ab
            dev = np.linalg.norm(cpts - chord, axis=1)
            k = int(np.argmax(dev))
            if float(dev[k]) <= best_dev:
                continue
            pt = chord[k]
            near_obs = len(obs) and float(np.min(np.linalg.norm(obs[:, :2] - pt[:2], axis=1))) < (
                settings.r_obs
            )
            near_gate = len(gates_xy) and float(
                np.min(np.linalg.norm(gates_xy - pt[:2], axis=1))
            ) < settings.peri_gate_radius
            if near_obs or near_gate:
                continue
            best_seg, best_dev, best_pt = i, float(dev[k]), pt
        if best_seg < 0:
            break
        waypoints = np.insert(waypoints, best_seg + 1, best_pt, axis=0)
        knot_times, curve = build_spline(waypoints, start_vel, gates_pos, obstacles_pos, settings)
    return waypoints, knot_times, curve


def repair_obstacles(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    settings: PlannerSettings,
    skip_gate_near: bool = True,
    straighten: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating], CubicSpline]:
    """Obstacle-clearance repair (Rule 1) followed by P2 overshoot-straightening.

    ``skip_gate_near=False`` repairs every obstacle (no Rule-1 exclusion) and ``straighten=False``
    disables the overshoot pass -- together they give the P3 conservative fallback.
    """
    waypoints, knot_times, curve = _repair_obstacle_pass(
        waypoints, start_vel, gates_pos, obstacles_pos, settings, skip_gate_near
    )
    if straighten:
        waypoints, knot_times, curve = _straighten_overshoot(
            waypoints, start_vel, gates_pos, obstacles_pos, settings
        )
    return waypoints, knot_times, curve


def smooth_waypoints(
    waypoints: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    gates_quat: NDArray[np.float64],
    obstacles_pos: NDArray[np.float64],
    clearance: float,
    pull: float,
    iterations: int,
) -> NDArray[np.float64]:
    """Pull free waypoints toward their neighbours' midpoint, then back out of obstacles.

    Protected (never moved): the chain ends, the first three points (start state and the
    replan-continuity / liftoff arc), each remaining gate's approach/gate/exit triplet, and
    the whole leg of every REVERSAL transition (same cos < -0.3 branch the planner takes in
    ``_clearance_points``) -- the U-turn swing/apex points are geometrically necessary per the
    v10.4 audit, and pulling them inward tightens the fold until the turn stops being
    trackable (measured: late gate crashes on the level2 g2->g3 reversal).
    """
    out = waypoints.copy()
    protected = {0, 1, 2, len(out) - 1}
    gate_idx = [int(np.argmin(np.linalg.norm(out - gate, axis=1))) for gate in gates_pos]
    for idx in gate_idx:
        protected.update({idx - 1, idx, idx + 1})
    for i in range(len(gates_pos) - 1):
        forward = Rotation.from_quat(gates_quat[i]).as_matrix()[:2, 0]
        to_next = (gates_pos[i + 1] - gates_pos[i])[:2]
        denom = float(np.linalg.norm(forward) * np.linalg.norm(to_next))
        if denom > 1e-9 and float(np.dot(forward, to_next)) / denom < -0.3:
            protected.update(range(gate_idx[i] + 1, gate_idx[i + 1]))
    for _ in range(iterations):
        for i in range(1, len(out) - 1):
            if i in protected:
                continue
            out[i] += pull * (0.5 * (out[i - 1] + out[i + 1]) - out[i])
            for obstacle in obstacles_pos:
                delta = out[i, :2] - obstacle[:2]
                norm = float(np.linalg.norm(delta))
                if norm < clearance:
                    direction = delta / norm if norm > 1e-6 else np.array([1.0, 0.0])
                    out[i, :2] = obstacle[:2] + direction * clearance
    return out


def _profile(
    curve: CubicSpline, t_total: float, v_cap: float, a_lat_max: float, v_min: float, n: int = 400
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Sample the spline densely: (arc grid, points, curvature-limited speed)."""
    taus = np.linspace(0.0, float(t_total), n)
    pts = np.asarray(curve(taus), dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    v = SpeedProfile(curve, t_total, v_cap, a_lat_max, v_min, n).at_arc(s)
    return s, pts, v


def _predicted_time(s: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """Ride time of the profile, sum(ds / v) with midpoint speeds."""
    return float(np.sum(np.diff(s) / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)))


def _gate_arc_positions(
    s: NDArray[np.float64], pts: NDArray[np.float64], gates_pos: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Arc length of the dense sample nearest to each gate centre."""
    return np.array([s[int(np.argmin(np.linalg.norm(pts - g, axis=1)))] for g in gates_pos])


def _window_caps(
    s: NDArray[np.float64],
    pts: NDArray[np.float64],
    v: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    window_pre: float,
) -> NDArray[np.float64]:
    """Per gate: the profile's max speed inside the reveal window [s_gate - pre, s_gate]."""
    caps = np.empty(len(gates_pos))
    for i, s_g in enumerate(_gate_arc_positions(s, pts, gates_pos)):
        mask = (s >= s_g - window_pre) & (s <= s_g)
        caps[i] = float(np.max(v[mask])) if np.any(mask) else np.inf
    return caps


def _passage_runs(
    pts: NDArray[np.float64], obstacle: NDArray[np.float64], radius: float
) -> list[NDArray[np.intp]]:
    """Contiguous sample-index runs where the path is within ``radius`` of the obstacle (XY)."""
    near = np.linalg.norm(pts[:, :2] - obstacle[:2], axis=1) < radius
    edges = np.nonzero(np.diff(near.astype(int)))[0] + 1
    return [run for run in np.split(np.arange(len(near)), edges) if near[run[0]]]


def _obstacle_caps(
    base: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    cand: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    obstacles_pos: NDArray[np.float64],
    radius: float,
) -> NDArray[np.float64]:
    """Arc-interval caps (s_lo, s_hi, v_max) on the candidate path's obstacle passages.

    ``base``/``cand`` are (arc grid, points, profile speed) triples. Each contiguous candidate
    passage within ``radius`` of an obstacle is capped at the base profile's max over the
    ORDER-MATCHED base passage of that obstacle -- per-passage parity, so one passage's speed
    never licenses (or taxes) another's. When the passage counts differ, every candidate
    passage gets the slowest base passage's cap; when the base path never enters the tube, its
    speed at the closest approach.
    """
    s_b, pts_b, v_b = base
    s_c, pts_c, v_c = cand
    rows: list[tuple[float, float, float]] = []
    for obstacle in obstacles_pos:
        cand_runs = _passage_runs(pts_c, obstacle, radius)
        if not cand_runs:
            continue
        base_runs = _passage_runs(pts_b, obstacle, radius)
        if base_runs:
            base_caps = [float(np.max(v_b[run])) for run in base_runs]
        else:
            d = np.linalg.norm(pts_b[:, :2] - obstacle[:2], axis=1)
            base_caps = [float(v_b[int(np.argmin(d))])]
        matched = len(base_caps) == len(cand_runs)
        pad = 0.06  # a couple of grid steps, so resampling in the path view can't shrink the tube
        for k, run in enumerate(cand_runs):
            cap = base_caps[k] if matched else min(base_caps)
            rows.append((float(s_c[run[0]]) - pad, float(s_c[run[-1]]) + pad, cap))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def _crossings_centred(
    pts: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    gates_quat: NDArray[np.float64],
    tol: float,
) -> bool:
    """True if every plane crossing inside a gate's frame square threads the opening centre."""
    for gate, quat in zip(gates_pos, gates_quat):
        rot = Rotation.from_quat(quat).as_matrix()
        rel = pts - gate
        f = rel @ rot[:, 0]
        for i in np.nonzero(np.signbit(f[:-1]) != np.signbit(f[1:]))[0]:
            w = f[i] / (f[i] - f[i + 1])
            p = rel[i] + w * (rel[i + 1] - rel[i])
            lat, vert = abs(float(p @ rot[:, 1])), abs(float(p @ rot[:, 2]))
            inside_frame = lat < _FRAME_HALF_M and vert < _FRAME_HALF_M
            if inside_frame and (lat > tol or vert > tol):
                return False
    return True


def _make_plan(
    frame: DroneObservation,
    curve: CubicSpline,
    knots: NDArray[np.float64],
    waypoints: NDArray[np.float64],
    gate_caps: NDArray[np.float64],
    obs_caps: NDArray[np.float64],
    smoothed: bool,
) -> SmoothedPlan:
    """Assemble a SmoothedPlan, snapshotting the frame the way v10.4's build does."""
    return SmoothedPlan(
        curve=curve,
        t_total=float(knots[-1]),
        waypoints=waypoints,
        gate_pos_snapshot=frame.gate_pos.copy(),
        obstacle_pos_snapshot=frame.obstacles_pos.copy(),
        built_target_gate=frame.target_gate,
        gate_window_caps=gate_caps,
        obstacle_caps=obs_caps,
        smoothed=smoothed,
    )


class ReferenceManager:
    """Episode-sticky planner building guarded-smoothed, parity-capped reference plans.

    Live behaviour is v10.6's ``__init__``/``reset``/``build`` over v8's
    ``ensure_plan``/``_needs_plan``/``_planning_obstacles`` (flattened from the
    v8 -> v10.4 -> v10.6 ReferenceManager chain; the shadowed v8/v10.4 ``build`` are dropped).
    """

    def __init__(
        self,
        settings: SmoothPlannerSettings,
        replan_gate_delta_m: float,
        replan_obstacle_delta_m: float,
        v_cap: float,
        a_lat_max: float,
        v_min: float,
    ):
        """Store replanning thresholds plus the speed-profile budget the guard scores with."""
        self._settings = settings
        self._gate_delta = float(replan_gate_delta_m)
        self._obstacle_delta = float(replan_obstacle_delta_m)
        self._plan: ReferencePlan | None = None
        self._v_cap = float(v_cap)
        self._a_lat_max = float(a_lat_max)
        self._v_min = float(v_min)
        self._mode: str | None = None  # episode-sticky: "smooth" or "base", set by build 1
        self.decisions: list[str] = []  # per build: "accept" or the guard that rejected

    @property
    def plan(self) -> ReferencePlan | None:
        """The active plan, if one exists."""
        return self._plan

    def reset(self) -> None:
        """Forget the cached plan and the episode's sticky smoothing mode."""
        self._plan = None
        self._mode = None

    def _real_obstacles(self, frame: DroneObservation) -> NDArray[np.float64]:
        """Merged real obstacles (Rule 2), WITHOUT the funnel posts.

        This is the set used for HARD obstacle repair: the funnel posts are soft and applied only
        through build_waypoints' lateral nudging, so treating them as hard keep-outs (which would
        push the path off the gate centreline) is wrong (fix 4: P3 fallback funnel-post bug).
        """
        return _merge_close_obstacles(
            np.asarray(frame.obstacles_pos, dtype=np.float64), OBSTACLE_MERGE_DIST
        )

    def _planning_obstacles(self, frame: DroneObservation) -> NDArray[np.float64]:
        """Real obstacles (merged) plus gate-post funnel columns, for waypoint building/nudging."""
        obstacles = self._real_obstacles(frame)
        if not (self._settings.funnel_enabled and self._settings.gate_post_offset > 0.0):
            return obstacles
        posts = gate_post_obstacles(
            frame.gate_pos, frame.gate_quat, frame.target_gate, self._settings.gate_post_offset
        )
        if len(posts) == 0:
            return obstacles
        if len(obstacles) == 0:
            return posts
        return np.concatenate([obstacles, posts], axis=0)

    def build(
        self,
        start_pos: NDArray[np.float64],
        start_vel: NDArray[np.float64],
        frame: DroneObservation,
    ) -> SmoothedPlan:
        """Build the plan, then P3-guard it.

        If the chosen curve clips a real obstacle (< r_obs) or misses a remaining gate, fall back
        to a conservative full-repair plan (Rule 1 disabled, no overshoot straightening) instead
        of flying a known-bad spline.
        """
        plan = self._build_candidate(start_pos, start_vel, frame)
        if self._plan_feasible(plan.curve, plan.t_total, frame):
            return plan
        # P3 fallback: full obstacle repair (every obstacle), no overshoot straightening.
        self.decisions.append("fallback")
        settings = self._settings
        planning_obstacles = self._planning_obstacles(frame)
        raw = build_waypoints(
            start_pos, start_vel, frame.gate_pos, frame.gate_quat,
            planning_obstacles, frame.target_gate, settings,
        )
        wps, knots, curve = repair_obstacles(
            raw, start_vel, frame.gate_pos, self._real_obstacles(frame), settings,
            skip_gate_near=False, straighten=False,
        )
        n_rem = len(np.asarray(frame.gate_pos, dtype=np.float64)[max(frame.target_gate, 0):])
        return _make_plan(frame, curve, knots, wps, np.full(n_rem, np.inf), np.empty((0, 3)), False)

    def _plan_feasible(self, curve: CubicSpline, t_total: float, frame: DroneObservation) -> bool:
        """P3 check: curve clears real obstacles (>= r_obs) and threads each remaining gate."""
        s = self._settings
        pts = np.asarray(curve(np.linspace(0.0, float(t_total), 200)), dtype=np.float64)
        obs = np.asarray(frame.obstacles_pos, dtype=np.float64).reshape(-1, 3)
        if len(obs):
            d = float(np.min(np.linalg.norm(pts[:, None, :2] - obs[None, :, :2], axis=2)))
            if d < s.r_obs:
                return False
        first = max(frame.target_gate, 0)
        gp = np.asarray(frame.gate_pos, dtype=np.float64)[first:]
        gq = np.asarray(frame.gate_quat, dtype=np.float64)[first:]
        return _crossings_centred(pts, gp, gq, s.cross_tol_m)

    def _build_candidate(
        self,
        start_pos: NDArray[np.float64],
        start_vel: NDArray[np.float64],
        frame: DroneObservation,
    ) -> SmoothedPlan:
        """Build the v10.4 plan and a smoothed candidate; ship the candidate only if it guards."""
        settings: SmoothPlannerSettings = self._settings
        planning_obstacles = self._planning_obstacles(frame)
        real_obstacles = self._real_obstacles(frame)  # hard-repair set (no funnel posts)
        first = max(frame.target_gate, 0)
        remaining_pos = np.asarray(frame.gate_pos, dtype=np.float64)[first:]
        remaining_quat = np.asarray(frame.gate_quat, dtype=np.float64)[first:]

        raw = build_waypoints(
            start_pos,
            start_vel,
            frame.gate_pos,
            frame.gate_quat,
            planning_obstacles,
            frame.target_gate,
            settings,
        )
        base_wps, base_knots, base_curve = repair_obstacles(
            raw, start_vel, frame.gate_pos, real_obstacles, settings, skip_gate_near=False
        )

        def base_plan(reason: str) -> SmoothedPlan:
            if self._mode is None:
                self._mode = "base"
            self.decisions.append(reason)
            return _make_plan(
                frame,
                base_curve,
                base_knots,
                base_wps,
                np.full(len(remaining_pos), np.inf),
                np.empty((0, 3)),
                False,
            )

        # The smooth-or-not decision is STICKY for the episode: a mid-flight flip between the
        # smoothed and base routes teleports the reference under the drone by up to ~0.5 m
        # (measured: per-rebuild flips erased the whole arc gain in tracking transients and
        # piled crashes at the last gate). Build 1 decides; replans keep the mode and only the
        # safety guards below may still demote a single rebuild.
        if self._mode == "base":
            return base_plan("base-locked")
        first_build = self._mode is None

        smooth_raw = smooth_waypoints(
            raw,
            remaining_pos,
            remaining_quat,
            planning_obstacles,
            settings.r_obs + 0.06,
            settings.smooth_pull,
            settings.smooth_iters,
        )
        if np.allclose(smooth_raw, raw):  # nothing was free to move (e.g. sharp slalom)
            return base_plan("noop")
        cand_wps, cand_knots, cand_curve = repair_obstacles(
            smooth_raw, start_vel, frame.gate_pos, real_obstacles, settings, skip_gate_near=False
        )

        budget = (self._v_cap, self._a_lat_max, self._v_min)
        s_b, pts_b, v_b = _profile(base_curve, float(base_knots[-1]), *budget)
        s_c, pts_c, v_c = _profile(cand_curve, float(cand_knots[-1]), *budget)
        gate_caps = _window_caps(s_b, pts_b, v_b, remaining_pos, settings.reveal_window_m)
        obs_caps = _obstacle_caps(
            (s_b, pts_b, v_b),
            (s_c, pts_c, v_c),
            np.asarray(frame.obstacles_pos),
            settings.obs_cap_radius,
        )

        # Guard 1/2: crossings centred and real-obstacle clearance held on the candidate.
        if not _crossings_centred(pts_c, remaining_pos, remaining_quat, settings.cross_tol_m):
            return base_plan("crossings")
        if (
            len(frame.obstacles_pos)
            and float(
                np.min(
                    np.linalg.norm(pts_c[:, None, :2] - frame.obstacles_pos[None, :, :2], axis=2)
                )
            )
            < settings.r_obs
        ):
            return base_plan("clearance")

        # Guard 3 (build 1 only -- replans inherit the mode): the candidate must still win
        # AFTER the parity caps are priced in.
        if first_build:
            v_capped = v_c.copy()
            for s_g, cap in zip(_gate_arc_positions(s_c, pts_c, remaining_pos), gate_caps):
                mask = (s_c >= s_g - settings.reveal_window_m) & (s_c <= s_g)
                v_capped[mask] = np.minimum(v_capped[mask], cap)
            for s_lo, s_hi, cap in obs_caps:
                mask = (s_c >= s_lo) & (s_c <= s_hi)
                v_capped[mask] = np.minimum(v_capped[mask], cap)
            if _predicted_time(s_c, v_capped) > _predicted_time(s_b, v_b) - settings.min_gain_s:
                return base_plan("gain")

        self._mode = "smooth"
        self.decisions.append("accept")
        return _make_plan(frame, cand_curve, cand_knots, cand_wps, gate_caps, obs_caps, True)

    def ensure_plan(self, frame: DroneObservation) -> tuple[ReferencePlan, bool]:
        """Return the active plan and whether it was rebuilt this call."""
        if self._needs_plan(frame):
            self._plan = self.build(frame.pos, frame.vel, frame)
            return self._plan, True
        return self._plan, False

    def _needs_plan(self, frame: DroneObservation) -> bool:
        plan = self._plan
        if plan is None or frame.target_gate != plan.built_target_gate:
            return True
        remaining = frame.gate_pos[frame.target_gate :]
        snapshot = plan.gate_pos_snapshot[frame.target_gate :]
        if len(remaining) and float(np.max(np.linalg.norm(remaining - snapshot, axis=1))) > (
            self._gate_delta
        ):
            return True
        if (
            len(frame.obstacles_pos)
            and float(
                np.max(np.linalg.norm(frame.obstacles_pos - plan.obstacle_pos_snapshot, axis=1))
            )
            > self._obstacle_delta
        ):
            return True
        return False
