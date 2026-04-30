from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.kafa1500_v5.attitude import TrackingSample, attitude_action
from lsy_drone_racing.control.kafa1500_v5.feedback import CascadedPid
from lsy_drone_racing.control.kafa1500_v5.geometry import gate_rpy_from_quat
from lsy_drone_racing.control.kafa1500_v5.settings import ControllerSettings
from lsy_drone_racing.control.kafa1500_v5.state import parse_observation
from lsy_drone_racing.control.kafa1500_v5.trajectory import ReferenceManager, ReferencePlan

if TYPE_CHECKING:
    from numpy.typing import NDArray


class KaFa1500V5(Controller):
    """Seviye-2 eleme parkuru için modüler kademeli PID attitude denetleyicisi."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Zamanlamayı, fiziksel parametreleri, geri besleme durumunu ve planlayıcı durumunu başlatır."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("KaFa_1500_v5 requires env.control_mode = 'attitude'.")
        self._settings = ControllerSettings()
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(params["mass"])
        self._feedback = CascadedPid(self._settings.feedback)
        self._references = ReferenceManager(
            self._settings.planner,
            self._settings.runtime.replan_near_gate_xy_m,
            self._settings.runtime.replan_gate_delta_m,
        )
        self._tick = 0
        self._finished = False
        self._last_target = -2
        self._last_action = self._hover_action()
        self._last_time = 0.0
        self._last_sample: TrackingSample | None = None

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """[roll, pitch, yaw, thrust] sırasıyla bir attitude kip eylemi hesaplar."""
        frame = parse_observation(obs)
        now = min(self._tick / self._freq, self._settings.runtime.timeout_s)
        self._last_time = now

        if frame.target_gate == -1 or now >= self._settings.runtime.timeout_s:
            self._finished = True
            return self._last_action.copy()

        if frame.target_gate != self._last_target:
            self._feedback.set_sector(frame.target_gate, preserve_integrals=True)
            self._last_target = frame.target_gate

        gate_rpy = gate_rpy_from_quat(frame.gate_quat)
        plan = self._references.ensure_plan(frame, gate_rpy)
        t_eval = min(max(now, plan.t_start), plan.t_end)
        action, sample = attitude_action(
            plan.curve,
            t_eval,
            frame.pos,
            frame.vel,
            frame.quat,
            self._feedback,
            self._dt,
            self._mass,
            self._settings.runtime.gravity,
            self._settings.command,
        )
        self._last_action = action
        self._last_sample = sample
        return action.copy()

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Bir ortam adımından sonra denetleyici saatini ilerletir."""
        self._tick += 1
        return self._finished

    def reset(self) -> None:
        """Bölüme özgü denetleyici durumunu sıfırlar."""
        self._tick = 0
        self._finished = False
        self._last_target = -2
        self._feedback.reset()
        self._references.reset()
        self._last_action = self._hover_action()
        self._last_time = 0.0
        self._last_sample = None

    def episode_callback(self) -> None:
        """Bir bölüm tamamlandıktan sonra durumu sıfırlar."""
        self.reset()

    def episode_reset(self) -> None:
        """Sonraki bölüm başlamadan önce durumu sıfırlar."""
        self.reset()

    def render_callback(self, sim: object) -> None:
        """Görselleştirme kasıtlı olarak isteğe bağlıdır; burada denetim durumu değiştirilmez."""

    def diagnostic(self) -> dict[str, float | int | str | None]:
        """Hata ayıklama ve günlükleme için kısa bir durum özeti döndürür."""
        plan: ReferencePlan | None = self._references.plan
        return {
            "controller_phase": "FINISHED" if self._finished else "TRACKING",
            "active_target_gate": None if plan is None else plan.sector,
            "controller_time": self._last_time,
            "reference_end_time": None if plan is None else plan.t_end,
            "planning_mode": None if plan is None else plan.mode,
            "pid_sector": self._feedback.sector,
        }

    def _hover_action(self) -> NDArray[np.float32]:
        thrust = np.clip(
            self._mass * self._settings.runtime.gravity,
            self._settings.command.thrust_min,
            self._settings.command.thrust_max,
        )
        return np.array([0.0, 0.0, 0.0, thrust], dtype=np.float32)
