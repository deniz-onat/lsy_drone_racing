from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.kafa1500_v4 import flight_command as _kafa1500_attitude
from lsy_drone_racing.control.kafa1500_v4 import race_line as _kafa1500_trajectory
from lsy_drone_racing.control.kafa1500_v4.cascade_control import PositionPid
from lsy_drone_racing.control.kafa1500_v4.qualification_setup import gate1_offset_tuning
from lsy_drone_racing.control.kafa1500_v4.race_line import (
    ROUTE_OVERRIDE_FILE,
    RouteTuning,
    load_route_overrides,
)
from lsy_drone_racing.control.kafa1500_v4.track_math import (
    DEFAULT_GATE_POS,
    DEFAULT_GATE_RPY,
    DEFAULT_OBSTACLES,
    normalize_gate_index,
)

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline


_kafa1500_tracking_command = getattr(
    _kafa1500_attitude,
    "tracking_command",
    getattr(_kafa1500_attitude, "attitude_command", None),
)
if _kafa1500_tracking_command is None:
    raise ImportError("qualification attitude helper does not expose a tracking command")

_kafa1500_build_reference_curve = getattr(
    _kafa1500_trajectory,
    "build_reference_curve",
    getattr(_kafa1500_trajectory, "plan_sector_spline", None),
)
if _kafa1500_build_reference_curve is None:
    raise ImportError("qualification trajectory helper does not expose a reference builder")


@dataclass(frozen=True)
class _KaFa1500ObservationFrame:
    target_gate: int
    gate_pos: NDArray[np.floating]
    gate_quat: NDArray[np.floating]
    pos: NDArray[np.floating]
    vel: NDArray[np.floating]
    quat: NDArray[np.floating]


class QualificationController(Controller):
    """Yeterlilik koşuları için referans izleyen KaFa-1500 denetleyicisi."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Denetleyici durumunu ve ayar tablolarını hazırlar."""
        super().__init__(obs, info, config)
        self._kafa1500_freq_hz = float(config.env.freq)
        self._kafa1500_time_limit = 25.0

        self.gate_rpy = DEFAULT_GATE_RPY.copy()
        self._kafa1500_reference_gate_pos = DEFAULT_GATE_POS.copy()
        self._kafa1500_obstacles = DEFAULT_OBSTACLES.copy()
        self._kafa1500_route_overrides = load_route_overrides(ROUTE_OVERRIDE_FILE)

        self.tuning = gate1_offset_tuning()
        self.leg_times = self.tuning.leg_times.copy()
        self.leg_speed_profiles = self.tuning.speed_profiles
        self.route_tuning = getattr(self.tuning, "route_tuning", RouteTuning())
        self.pid_gains_by_section = getattr(
            self.tuning,
            "pid_gains_by_section",
            (self.tuning.pid_gains,) * 4,
        )

        kafa1500_drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.mass = float(kafa1500_drone_params["mass"])
        self.g = 9.81
        self.position_pid = PositionPid(self.pid_gains_by_section[0])

        self._kafa1500_reference: CubicSpline | None = None
        self._kafa1500_reference_t_end = self._kafa1500_time_limit
        self._kafa1500_tick = 0
        self._kafa1500_finished = False
        self._kafa1500_needs_initial_plan = True
        self._kafa1500_active_leg = -1
        self._kafa1500_leg_start_t = np.zeros(4, dtype=np.float64)
        self._kafa1500_last_action = np.zeros(4, dtype=np.float32)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Bir sonraki tutum ve itki komutunu hesaplar."""
        del info
        kafa1500_now = self._kafa1500_time()
        kafa1500_frame = self._kafa1500_observation_frame(obs)

        if self._kafa1500_terminal_gate(kafa1500_frame.target_gate):
            return self._kafa1500_last_action
        if kafa1500_now >= self._kafa1500_time_limit:
            self._kafa1500_finished = True

        self._kafa1500_update_gate_attitude(kafa1500_frame.gate_quat)
        if kafa1500_frame.target_gate != self._kafa1500_active_leg:
            self._kafa1500_select_section_pid(kafa1500_frame.target_gate)

        if self._kafa1500_should_replan(kafa1500_now, kafa1500_frame):
            self._kafa1500_refresh_reference(
                kafa1500_now,
                kafa1500_frame.target_gate,
                kafa1500_frame.gate_pos,
            )
            self._kafa1500_needs_initial_plan = False
            self._kafa1500_reference_gate_pos = kafa1500_frame.gate_pos.copy()

        kafa1500_eval_t = min(kafa1500_now, self._kafa1500_reference_t_end)
        kafa1500_action = _kafa1500_tracking_command(
            self._kafa1500_require_reference(),
            kafa1500_frame.pos,
            kafa1500_frame.vel,
            kafa1500_frame.quat,
            kafa1500_eval_t,
            position_pid=self.position_pid,
            dt=1.0 / self._kafa1500_freq_hz,
            mass=self.mass,
            gravity=self.g,
        )
        self._kafa1500_last_action = kafa1500_action
        return kafa1500_action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Saat sayacını ilerletir ve bitiş durumunu bildirir."""
        del action, obs, reward, info
        self._kafa1500_tick += 1
        if terminated or truncated:
            self._kafa1500_finished = True
        return self._kafa1500_finished

    def episode_callback(self) -> None:
        """Bölüm sonrasında uçuş durumunu sıfırlar."""
        self.reset()

    def reset(self) -> None:
        """Bölüme ait tüm geçici durumu sıfırlar."""
        self._kafa1500_tick = 0
        self._kafa1500_needs_initial_plan = True
        self._kafa1500_active_leg = -1
        self._kafa1500_finished = False
        self.position_pid.reset()
        self._kafa1500_last_action = np.zeros(4, dtype=np.float32)
        self._kafa1500_reference = None
        self._kafa1500_reference_t_end = self._kafa1500_time_limit
        self._kafa1500_leg_start_t[:] = 0.0

    def episode_reset(self) -> None:
        """Bölüme ait tüm geçici durumu sıfırlar."""
        self.reset()

    def render_callback(self, sim: Sim) -> None:
        """Geçerli referansı ve anlık hedef noktayı çizer."""
        if self._kafa1500_reference is None:
            return
        kafa1500_leg = max(self._kafa1500_active_leg, 0)
        kafa1500_now = min(self._kafa1500_time(), self._kafa1500_reference_t_end)
        kafa1500_setpoint = self._kafa1500_reference(kafa1500_now).reshape(1, -1)
        draw_points(sim, kafa1500_setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        kafa1500_start = self._kafa1500_leg_start_t[kafa1500_leg]
        kafa1500_stop = kafa1500_start + self.leg_times[kafa1500_leg]
        kafa1500_line = self._kafa1500_reference(np.linspace(kafa1500_start, kafa1500_stop, 100))
        draw_line(sim, kafa1500_line, rgba=(0.0, 1.0, 0.0, 1.0))

    def diagnostic(self) -> dict:
        """Web panosu için küçük bir durum özeti verir."""
        return {
            "controller_phase": "FINISHED" if self._kafa1500_finished else "TRACKING",
            "target_gate": self._kafa1500_active_leg,
            "traj_local_time": self._kafa1500_tick / self._kafa1500_freq_hz,
            "traj_total_time": self._kafa1500_reference_t_end,
            "plan_mode": "qualification_reference",
        }

    def _kafa1500_time(self) -> float:
        return min(self._kafa1500_tick / self._kafa1500_freq_hz, self._kafa1500_time_limit)

    def _kafa1500_observation_frame(
        self,
        obs: dict[str, NDArray[np.floating]],
    ) -> _KaFa1500ObservationFrame:
        return _KaFa1500ObservationFrame(
            target_gate=normalize_gate_index(obs["target_gate"]),
            gate_pos=np.asarray(obs["gates_pos"], dtype=np.float64),
            gate_quat=np.asarray(obs["gates_quat"], dtype=np.float64),
            pos=np.asarray(obs["pos"], dtype=np.float64),
            vel=np.asarray(obs["vel"], dtype=np.float64),
            quat=np.asarray(obs["quat"], dtype=np.float64),
        )

    def _kafa1500_terminal_gate(self, target_gate: int) -> bool:
        if target_gate != -1:
            return False
        self._kafa1500_finished = True
        return True

    def _kafa1500_update_gate_attitude(self, gate_quat: NDArray[np.floating]) -> None:
        self.gate_rpy = R.from_quat(gate_quat).as_euler("xyz", degrees=False)

    def _kafa1500_should_replan(
        self,
        now: float,
        frame: _KaFa1500ObservationFrame,
    ) -> bool:
        del now
        kafa1500_gate_delta = float(
            np.linalg.norm(
                self._kafa1500_reference_gate_pos[frame.target_gate]
                - frame.gate_pos[frame.target_gate]
            )
        )
        kafa1500_horizontal_distance = float(
            np.linalg.norm(frame.gate_pos[frame.target_gate, :2] - frame.pos[:2])
        )
        kafa1500_gate_shifted_nearby = (
            kafa1500_gate_delta > 0.005 and kafa1500_horizontal_distance < 0.7
        )
        return (
            self._kafa1500_needs_initial_plan
            or self._kafa1500_active_leg != frame.target_gate
            or kafa1500_gate_shifted_nearby
        )

    def _kafa1500_refresh_reference(
        self,
        t: float,
        target_gate: int,
        gate_pos: NDArray[np.floating],
    ) -> None:
        if self._kafa1500_active_leg != target_gate:
            self._kafa1500_leg_start_t[target_gate] = t
            self._kafa1500_active_leg = target_gate
        self._kafa1500_reference, self._kafa1500_reference_t_end = (
            self._kafa1500_build_reference_with_fallback(target_gate, gate_pos)
        )

    def _kafa1500_build_reference_with_fallback(
        self,
        target_gate: int,
        gate_pos: NDArray[np.floating],
    ) -> tuple[CubicSpline, float]:
        kafa1500_common_args = (
            target_gate,
            gate_pos,
            self.gate_rpy,
            self._kafa1500_obstacles,
            float(self._kafa1500_leg_start_t[target_gate]),
            float(self.leg_times[target_gate]),
        )
        try:
            return _kafa1500_build_reference_curve(
                *kafa1500_common_args,
                route_overrides=self._kafa1500_route_overrides,
                speed_profile=self.leg_speed_profiles[target_gate],
                route_tuning=self.route_tuning,
            )
        except TypeError as exc:
            if not self._kafa1500_is_legacy_reference_builder(exc):
                raise
            return _kafa1500_build_reference_curve(*kafa1500_common_args)

    def _kafa1500_is_legacy_reference_builder(self, exc: TypeError) -> bool:
        kafa1500_message = str(exc)
        return any(
            kafa1500_key in kafa1500_message
            for kafa1500_key in ("route_overrides", "speed_profile", "route_tuning")
        )

    def _kafa1500_require_reference(self) -> CubicSpline:
        if self._kafa1500_reference is None:
            raise RuntimeError("QualificationController used before planning a reference")
        return self._kafa1500_reference

    def _kafa1500_select_section_pid(self, target_gate: int) -> None:
        kafa1500_gains = self.pid_gains_by_section[target_gate]
        if self.position_pid.gains is kafa1500_gains:
            return
        self.position_pid.set_gains(kafa1500_gains)
