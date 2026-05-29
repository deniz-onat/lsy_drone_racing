"""Sector reference planning for KaFa_1500_v5."""
# ruff: noqa: TC001, TC002

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator

from lsy_drone_racing.control.legacy.kafa1500_v5.geometry import (
    gate_axis,
    gate_entry_exit,
    horizontal_unit,
)
from lsy_drone_racing.control.legacy.kafa1500_v5.settings import (
    DEFAULT_OBSTACLES,
    GATE1_ARC_POINTS,
    ROUTE2_MIDPOINT,
    ROUTE3_MIDPOINT,
    START_WAYPOINT,
    PlannerSettings,
    SpeedProfile,
)
from lsy_drone_racing.control.legacy.kafa1500_v5.state import DroneObservation

ReferenceCurve = CubicSpline | PchipInterpolator


@dataclass(frozen=True)
class ReferencePlan:
    """Sector-local plan expressed in absolute controller time."""

    sector: int
    curve: ReferenceCurve
    t_start: float
    t_end: float
    waypoints: NDArray[np.float64]
    mode: str


def knot_times(
    t_start: float, t_end: float, waypoints: NDArray[np.float64], profile: SpeedProfile, eps: float
) -> NDArray[np.float64]:
    """Distribute non-uniform waypoint times from a piecewise speed profile."""
    pts = np.asarray(waypoints, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError(f"waypoints must be 2-D, got {pts.shape}")
    if len(pts) == 0:
        raise ValueError("waypoints cannot be empty")
    if len(pts) == 1:
        return np.array([t_start], dtype=np.float64)
    duration = float(t_end - t_start)
    if duration <= 0.0:
        raise ValueError(f"sector duration must be positive, got {duration}")

    segment_count = len(pts) - 1
    centers = (np.arange(segment_count, dtype=np.float64) + 0.5) / segment_count
    multipliers = np.interp(centers, [0.0, 0.5, 1.0], [profile.start, profile.mid, profile.end])
    interval_weight = 1.0 / np.maximum(multipliers, eps)
    intervals = duration * interval_weight / np.sum(interval_weight)
    return np.concatenate(([t_start], t_start + np.cumsum(intervals)))


class RouteOverrides:
    """Optional external XY route edits that stay deterministic when the file is absent."""

    def __init__(self, settings: PlannerSettings):
        """Load route overrides once at controller startup."""
        self._settings = settings
        self._routes = self._read_file()

    def _read_file(self) -> tuple[NDArray[np.float64] | None, ...] | None:
        path = self._settings.override_file
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        raw_routes = data.get("qualification_route_points_xy", data.get("waypoints_xy"))
        if not isinstance(raw_routes, list):
            return None
        parsed: list[NDArray[np.float64] | None] = []
        for item in raw_routes[:4]:
            arr = np.asarray(item, dtype=np.float64)
            parsed.append(arr if arr.ndim == 2 and arr.shape[1] == 2 else None)
        while len(parsed) < 4:
            parsed.append(None)
        return tuple(parsed)

    def apply(
        self,
        sector: int,
        default_waypoints: NDArray[np.float64],
        extra_point: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Apply a sector override unless an avoidance point is being inserted."""
        if self._routes is None or extra_point is not None:
            return default_waypoints
        override = self._routes[sector]
        if override is None:
            return default_waypoints
        if len(override) == len(default_waypoints):
            merged = default_waypoints.copy()
            delta = np.linalg.norm(override - default_waypoints[:, :2], axis=1)
            changed = np.isfinite(override).all(axis=1) & (delta > self._settings.override_tol)
            merged[changed, :2] = override[changed]
            return merged

        finite = np.isfinite(override).all(axis=1)
        if not np.any(finite):
            return default_waypoints
        xy = override[finite]
        z = _interpolate_z(default_waypoints, xy)
        return np.column_stack((xy, z))


def _interpolate_z(
    default_waypoints: NDArray[np.float64], xy: NDArray[np.float64]
) -> NDArray[np.float64]:
    if len(xy) == 1:
        return np.array([default_waypoints[0, 2]], dtype=np.float64)
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    progress = np.concatenate(([0.0], np.cumsum(distances)))
    if progress[-1] <= 1e-12:
        return np.full(len(xy), default_waypoints[0, 2], dtype=np.float64)
    return np.interp(
        progress / progress[-1], [0.0, 1.0], [default_waypoints[0, 2], default_waypoints[-1, 2]]
    )


class SectorReferenceBuilder:
    """Build route waypoints, timing, interpolators, and clearance points."""

    def __init__(self, settings: PlannerSettings):
        """Initialize route generation and optional override handling."""
        self._settings = settings
        self._overrides = RouteOverrides(settings)

    def build(
        self, sector: int, gate_pos: NDArray[np.float64], gate_rpy: NDArray[np.float64]
    ) -> ReferencePlan:
        """Build a complete reference plan for a target gate sector."""
        return self._build_recursive(sector, gate_pos, gate_rpy, extra_point=None, depth=0)

    def _build_recursive(
        self,
        sector: int,
        gate_pos: NDArray[np.float64],
        gate_rpy: NDArray[np.float64],
        extra_point: NDArray[np.float64] | None,
        depth: int,
    ) -> ReferencePlan:
        waypoints = self._waypoints(sector, gate_pos, gate_rpy, extra_point)
        t_start = float(self._settings.leg_starts[sector])
        t_end = t_start + float(self._settings.leg_times[sector])
        times = knot_times(
            t_start,
            t_end,
            waypoints,
            self._settings.speed_profiles[sector],
            self._settings.speed_eps,
        )
        curve: ReferenceCurve
        if sector < 2:
            curve = CubicSpline(times, waypoints, axis=0)
        else:
            curve = PchipInterpolator(times, waypoints, axis=0)
        plan = ReferencePlan(
            sector,
            curve,
            t_start,
            t_end,
            waypoints,
            "avoidance" if extra_point is not None else "nominal",
        )

        next_extra = self._clearance_point(sector, plan)
        if next_extra is None or depth >= self._settings.max_avoidance_depth:
            return plan
        return self._build_recursive(sector, gate_pos, gate_rpy, next_extra, depth + 1)

    def _waypoints(
        self,
        sector: int,
        gate_pos: NDArray[np.float64],
        gate_rpy: NDArray[np.float64],
        extra_point: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if sector == 0:
            base = self._route0(gate_pos, gate_rpy)
        elif sector == 1:
            base = self._route1(gate_pos, gate_rpy)
        elif sector == 2:
            base = self._route2(gate_pos, gate_rpy)
        elif sector == 3:
            base = self._route3(gate_pos, gate_rpy)
        else:
            raise ValueError(f"unsupported sector {sector}")
        base = self._overrides.apply(sector, base, extra_point)
        if extra_point is None:
            return base
        insert_at = max(1, len(base) // 2)
        return np.insert(base, insert_at, np.asarray(extra_point, dtype=np.float64), axis=0)

    def _route0(
        self, gate_pos: NDArray[np.float64], gate_rpy: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _, g0_exit = gate_entry_exit(gate_pos[0], gate_rpy[0], eps=self._settings.gate_axis_eps)
        lateral = gate_axis(gate_rpy[0], local_axis=1, eps=self._settings.gate_axis_eps)
        if np.dot(lateral[:2], gate_pos[0, :2] - START_WAYPOINT[:2]) < 0.0:
            lateral = -lateral
        crossing = gate_pos[0].copy()
        crossing += self._settings.gate0_lateral_bias * lateral
        crossing[2] += self._settings.gate0_vertical_bias
        return np.vstack((START_WAYPOINT, crossing, g0_exit))

    def _route1(
        self, gate_pos: NDArray[np.float64], gate_rpy: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _, g0_exit = gate_entry_exit(gate_pos[0], gate_rpy[0], eps=self._settings.gate_axis_eps)
        g1_axis = gate_axis(gate_rpy[1], local_axis=0, eps=self._settings.gate_axis_eps)
        toward_g0 = horizontal_unit(gate_pos[0] - gate_pos[1], self._settings.gate_axis_eps)
        shift = self._settings.gate1_shift_toward_gate0 * toward_g0
        entry = gate_pos[1] - 0.25 * g1_axis + shift
        exit_point = gate_pos[1] + self._settings.gate1_exit_offset * g1_axis + shift
        return np.vstack((g0_exit, GATE1_ARC_POINTS, entry, exit_point))

    def _route2(
        self, gate_pos: NDArray[np.float64], gate_rpy: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        g1_axis = gate_axis(gate_rpy[1], local_axis=0, eps=self._settings.gate_axis_eps)
        toward_g0 = horizontal_unit(gate_pos[0] - gate_pos[1], self._settings.gate_axis_eps)
        start = gate_pos[1] + self._settings.gate1_exit_offset * g1_axis
        start += self._settings.gate1_shift_toward_gate0 * toward_g0
        entry, _ = gate_entry_exit(
            gate_pos[2],
            gate_rpy[2],
            entry_distance=self._settings.gate2_entry_offset,
            exit_distance=self._settings.gate2_exit_offset,
            eps=self._settings.gate_axis_eps,
        )
        _, exit_point = gate_entry_exit(
            gate_pos[2],
            gate_rpy[2],
            entry_distance=self._settings.gate2_entry_offset,
            exit_distance=self._settings.gate2_exit_offset,
            eps=self._settings.gate_axis_eps,
        )
        return np.vstack((start, ROUTE2_MIDPOINT, entry, gate_pos[2], exit_point))

    def _route3(
        self, gate_pos: NDArray[np.float64], gate_rpy: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _, start = gate_entry_exit(
            gate_pos[2],
            gate_rpy[2],
            entry_distance=self._settings.gate2_entry_offset,
            exit_distance=self._settings.gate2_exit_offset,
            eps=self._settings.gate_axis_eps,
        )
        _, exit_point = gate_entry_exit(
            gate_pos[3],
            gate_rpy[3],
            entry_distance=self._settings.gate3_entry_offset,
            exit_distance=self._settings.gate3_exit_offset,
            eps=self._settings.gate_axis_eps,
        )
        return np.vstack((start, ROUTE3_MIDPOINT, gate_pos[3], exit_point))

    def _clearance_point(self, sector: int, plan: ReferencePlan) -> NDArray[np.float64] | None:
        obstacle_idx = self._settings.obstacle_by_sector[sector]
        obstacle_xy = DEFAULT_OBSTACLES[obstacle_idx, :2]
        samples_t = np.linspace(plan.t_start, plan.t_end, self._settings.avoidance_samples)
        samples = np.asarray(plan.curve(samples_t), dtype=np.float64)
        distances = np.linalg.norm(samples[:, :2] - obstacle_xy, axis=1)
        closest_idx = int(np.argmin(distances))
        closest_distance = float(distances[closest_idx])
        trigger = float(self._settings.clearance_trigger[sector])
        if closest_distance >= trigger:
            return None
        push = min(
            max(trigger + float(self._settings.clearance_margin[sector]) - closest_distance, 0.0),
            float(self._settings.clearance_push_max[sector]),
        )
        direction = samples[closest_idx, :2] - obstacle_xy
        norm = float(np.linalg.norm(direction))
        if norm < self._settings.obstacle_eps:
            direction = np.array([1.0, 0.0], dtype=np.float64)
            norm = 1.0
        shifted = samples[closest_idx].copy()
        shifted[:2] += push * direction / norm
        return shifted


class ReferenceManager:
    """Manage active sector plans and replanning triggers."""

    def __init__(
        self, settings: PlannerSettings, runtime_near_gate_xy_m: float, gate_delta_m: float
    ):
        """Create a manager with documented near-gate replanning thresholds."""
        self._builder = SectorReferenceBuilder(settings)
        self._near_gate_xy_m = float(runtime_near_gate_xy_m)
        self._gate_delta_m = float(gate_delta_m)
        self._plan: ReferencePlan | None = None
        self._planned_gate_pos: NDArray[np.float64] | None = None

    @property
    def plan(self) -> ReferencePlan | None:
        """The active plan, if one exists."""
        return self._plan

    def reset(self) -> None:
        """Forget cached plans and planned gate positions."""
        self._plan = None
        self._planned_gate_pos = None

    def ensure_plan(self, frame: DroneObservation, gate_rpy: NDArray[np.float64]) -> ReferencePlan:
        """Return an active plan, rebuilding when sector or gate observations change."""
        if self._needs_plan(frame):
            self._plan = self._builder.build(frame.target_gate, frame.gate_pos, gate_rpy)
            self._planned_gate_pos = frame.gate_pos.copy()
        if self._plan is None:
            raise RuntimeError("reference manager did not create a plan")
        return self._plan

    def _needs_plan(self, frame: DroneObservation) -> bool:
        if self._plan is None or self._planned_gate_pos is None:
            return True
        if frame.target_gate != self._plan.sector:
            return True
        gate_delta = float(
            np.linalg.norm(
                frame.gate_pos[frame.target_gate] - self._planned_gate_pos[frame.target_gate]
            )
        )
        horizontal_distance = float(
            np.linalg.norm(frame.pos[:2] - frame.gate_pos[frame.target_gate, :2])
        )
        return gate_delta > self._gate_delta_m and horizontal_distance < self._near_gate_xy_m
