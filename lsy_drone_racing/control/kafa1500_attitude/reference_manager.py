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

    def update(self, pos: Vec3, tick: int) -> Reference:
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
        return self._reference(self._index, distance)

    def hold(self, pos: Vec3) -> Reference:
        """Hold the current position."""
        return Reference(
            position=pos.astype(np.float32),
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
        if float(np.linalg.norm(tangent[:2])) > 1e-6:
            self._last_yaw = float(np.arctan2(tangent[1], tangent[0]))

        return Reference(
            position=position,
            roll=0.0,
            pitch=0.0,
            yaw=self._last_yaw,
            index=index,
            distance=distance,
            done=index >= len(self._path.points) - 1,
        )
