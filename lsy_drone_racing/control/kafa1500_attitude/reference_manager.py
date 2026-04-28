"""Closed-loop reference target management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.kafa1500_attitude.types import Reference

if TYPE_CHECKING:
    from lsy_drone_racing.control.kafa1500_attitude.config import ReferenceConfig
    from lsy_drone_racing.control.kafa1500_attitude.types import CubicPath, Vec3


class ReferenceManager:
    """Advance the active path index based on observed position, not time."""

    def __init__(self, config: ReferenceConfig):
        """Initialize the manager."""
        self._config = config
        self._path: CubicPath | None = None
        self._index = config.start_index
        self._last_advance_tick = -config.min_ticks_between_advances
        self._last_yaw = 0.0

    @property
    def path(self) -> CubicPath | None:
        """Return the active path."""
        return self._path

    @property
    def index(self) -> int:
        """Return the active sample index."""
        return self._index

    def reset(self, path: CubicPath, yaw: float) -> None:
        """Load a new cubic path and reset target advancement."""
        self._path = path
        self._index = min(max(0, self._config.start_index), len(path.points) - 1)
        self._last_advance_tick = -self._config.min_ticks_between_advances
        self._last_yaw = yaw

    def update(self, pos: Vec3, vel: Vec3, tick: int) -> Reference:
        """Return the active reference, advancing only when close enough."""
        if self._path is None:
            return self.hold(pos)

        self._advance_to_nearby_forward_sample(pos, tick)
        for _ in range(self._config.max_advance_per_step):
            distance = float(np.linalg.norm(pos - self._path.points[self._index]))
            if not self._can_advance(distance, tick):
                break
            if self._index >= len(self._path.points) - 1:
                break
            self._index += 1
            self._last_advance_tick = tick

        distance = float(np.linalg.norm(pos - self._path.points[self._index]))
        speed_now = float(np.linalg.norm(vel))
        target_index = self._predictive_index(self._index, speed_now)
        return self._reference(target_index, distance)

    def hold(self, pos: Vec3) -> Reference:
        """Hold the current position."""
        zero = np.zeros(3, dtype=np.float32)
        return Reference(
            position=pos.astype(np.float32),
            velocity=zero,
            acceleration=zero,
            roll=0.0,
            pitch=0.0,
            yaw=self._last_yaw,
            index=self._index,
            distance=0.0,
            done=True,
        )

    def _can_advance(self, distance: float, tick: int) -> bool:
        if tick - self._last_advance_tick < self._config.min_ticks_between_advances:
            return False
        threshold = self._config.target_reached_distance
        if self._last_advance_tick == tick:
            threshold -= self._config.target_hysteresis
        return distance <= threshold

    def _advance_to_nearby_forward_sample(self, pos: Vec3, tick: int) -> None:
        if self._path is None:
            return

        stop = min(
            len(self._path.points),
            self._index + self._config.nearest_forward_search + 1,
        )

        candidates = self._path.points[self._index:stop]
        distances = np.linalg.norm(candidates - pos, axis=1)

        closest_offset = int(np.argmin(distances))

        if closest_offset <= 0:
            return

        advance = min(
            closest_offset + self._config.lookahead_samples,
            self._config.max_advance_per_step,
            len(self._path.points) - 1 - self._index,
        )

        if advance <= 0:
            return

        self._index += advance
        self._last_advance_tick = tick

    def _reference(self, index: int, distance: float) -> Reference:
        if self._path is None:
            raise RuntimeError("Reference requested without an active path.")

        param = float(self._path.params[index])
        position = np.asarray(self._path.spline(param), dtype=np.float32)
        tangent = np.asarray(self._path.velocity_spline(param), dtype=np.float32)
        curvature_vec = np.asarray(self._path.acceleration_spline(param), dtype=np.float32)
        tangent_norm = float(np.linalg.norm(tangent))
        speed = self._speed_for_index(index)
        if tangent_norm < 1e-6:
            velocity = np.zeros(3, dtype=np.float32)
            acceleration = np.zeros(3, dtype=np.float32)
        else:
            ds_dt = speed / tangent_norm
            velocity = (tangent / tangent_norm * speed).astype(np.float32)
            acceleration = (curvature_vec * ds_dt**2).astype(np.float32)
            acceleration = self._clip_horizontal_and_vertical(
                acceleration,
                self._config.max_reference_acc_xy,
                self._config.max_reference_acc_z,
            )
        yaw_tangent = self._yaw_tangent(index)
        if float(np.linalg.norm(yaw_tangent[:2])) > 1e-6:
            self._last_yaw = float(np.arctan2(yaw_tangent[1], yaw_tangent[0]))

        return Reference(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            # Roll and pitch are placeholders.  AttitudeFeedback normally derives
            # them from the desired force direction and uses these only as fallback.
            roll=0.0,
            pitch=0.0,
            yaw=self._last_yaw,
            index=index,
            distance=distance,
            done=index >= len(self._path.points) - 1,
        )

    def _predictive_index(self, progress_index: int, speed_now: float) -> int:
        if self._path is None:
            return progress_index
        dynamic_lookahead = self._config.base_lookahead_samples + int(
            self._config.speed_lookahead_gain * speed_now
        )
        dynamic_lookahead = min(dynamic_lookahead, self._config.max_lookahead_samples)
        return min(progress_index + dynamic_lookahead, len(self._path.points) - 1)

    def _yaw_tangent(self, target_index: int) -> Vec3:
        if self._path is None:
            return np.zeros(3, dtype=np.float32)
        yaw_index = min(target_index + self._config.yaw_preview_samples, len(self._path.points) - 1)
        param = float(self._path.params[yaw_index])
        return np.asarray(self._path.velocity_spline(param), dtype=np.float32)

    def _speed_for_index(self, index: int) -> float:
        if self._path is None:
            return 0.0
        if index >= len(self._path.points) - 1:
            return self._config.final_speed
        base_speed = self._config.nominal_speed
        gate_id = int(self._path.gate_indices[index])
        if gate_id >= 0:
            gate_window = self._path.gate_indices[
                max(0, index - self._config.gate_window_samples) : index
                + self._config.gate_window_samples
                + 1
            ]
            if np.any(gate_window == gate_id):
                base_speed = self._config.gate_speed
        if self._config.curvature_speed_enabled:
            base_speed = min(base_speed, self._curvature_speed(index))
        return float(np.clip(base_speed, self._config.min_turn_speed, self._config.max_turn_speed))

    def _curvature_speed(self, index: int) -> float:
        if self._path is None:
            return self._config.nominal_speed
        param = float(self._path.params[index])
        tangent = np.asarray(self._path.velocity_spline(param), dtype=np.float32)
        curvature_vec = np.asarray(self._path.acceleration_spline(param), dtype=np.float32)
        tangent_norm = float(np.linalg.norm(tangent))
        curvature = float(
            np.linalg.norm(np.cross(tangent, curvature_vec)) / (tangent_norm**3 + 1e-6)
        )
        if curvature <= 1e-6:
            return self._config.nominal_speed
        return float(np.sqrt(self._config.max_lateral_acc / curvature))

    @staticmethod
    def _clip_horizontal_and_vertical(vec: Vec3, max_xy: float, max_z: float) -> Vec3:
        clipped = vec.astype(np.float32).copy()
        xy_norm = float(np.linalg.norm(clipped[:2]))
        if xy_norm > max_xy and xy_norm > 1e-6:
            clipped[:2] *= max_xy / xy_norm
        clipped[2] = float(np.clip(clipped[2], -max_z, max_z))
        return clipped.astype(np.float32)
