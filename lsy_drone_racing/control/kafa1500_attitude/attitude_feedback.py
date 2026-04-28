"""Closed-loop attitude feedback control."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.kafa1500_attitude.utils import slew_angle, wrap_angle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_attitude.config import FeedbackConfig
    from lsy_drone_racing.control.kafa1500_attitude.types import Observation, Reference, Vec3


class AttitudeFeedback:
    """Convert position/velocity errors into `[roll, pitch, yaw, thrust]` commands."""

    def __init__(self, race_config: dict, config: FeedbackConfig, freq: float, yaw: float):
        """Load drone parameters and initialize filters."""
        self._config = config
        self._dt = 1.0 / float(freq)
        self._params = load_params(race_config.sim.physics, race_config.sim.drone_model)
        self._mass = max(43e-3, float(self._params["mass"]))
        self._thrust_min = float(self._params["thrust_min"] * 4.0)
        self._thrust_max = float(self._params["thrust_max"] * 4.0)
        self._cx_integral = np.zeros(3, dtype=np.float32)
        self._cv_integral = np.zeros(3, dtype=np.float32)
        self._last_e_vel = np.zeros(3, dtype=np.float32)
        self._has_last_e_vel = False
        self._yaw_integral = 0.0
        self._last_yaw_error = 0.0
        self._last_rotation_error = np.zeros(3, dtype=np.float32)
        self._debug: dict[str, float] = {}
        self._last_action = np.array(
            [0.0, 0.0, yaw, self._mass * self._config.gravity], dtype=np.float32
        )

    def reset(self, yaw: float) -> None:
        """Reset integrator and output filters."""
        self._cx_integral[:] = 0.0
        self._cv_integral[:] = 0.0
        self._last_e_vel[:] = 0.0
        self._has_last_e_vel = False
        self._yaw_integral = 0.0
        self._last_yaw_error = 0.0
        self._last_rotation_error[:] = 0.0
        self._debug.clear()
        self._last_action[:] = np.array(
            [0.0, 0.0, yaw, self._mass * self._config.gravity], dtype=np.float32
        )

    def command(self, obs: Observation, reference: Reference) -> NDArray[np.float32]:
        """Compute the handout-style attitude-mode command."""
        p_act = obs["pos"].astype(np.float32)
        v_act = obs["vel"].astype(np.float32)
        q_act = obs["quat"]
        rotation = R.from_quat(q_act)
        r_act_body_to_world = rotation.as_matrix().astype(np.float32)
        _, _, yaw_act = rotation.as_euler("xyz", degrees=False)

        # Frame convention: observations and references are world/inertial-frame vectors.
        # The quaternion is interpreted by scipy as body-to-world rotation.  The
        # returned action is [roll, pitch, yaw, collective thrust], with angles in
        # radians and positive thrust along the current body z-axis.
        p_des = reference.position.astype(np.float32)
        zero = np.zeros(3, dtype=np.float32)
        v_path_ref = np.asarray(getattr(reference, "velocity", zero), dtype=np.float32)
        roll_ref = float(reference.roll)
        pitch_ref = float(reference.pitch)
        yaw_ref = float(reference.yaw)

        # Cascaded translational control:
        # CX position PID turns position error into a corrective velocity.  The
        # path velocity, when present, is kept as feedforward velocity, so v_ref
        # is feedforward motion plus CX correction rather than a pure PID output.
        e_pos = p_des - p_act
        e_pos_dot = v_path_ref - v_act
        self._cx_integral = np.clip(
            self._cx_integral + e_pos * self._dt,
            -self._config.cx_integral_limit,
            self._config.cx_integral_limit,
        ).astype(np.float32)
        # Integrators are limited but still accumulate under output saturation.
        # Full anti-windup can be added later if saturation becomes a problem.
        cx_kp = self._axis_gains(self._config.cx_kp_xy, self._config.cx_kp_z)
        cx_ki = self._axis_gains(self._config.cx_ki_xy, self._config.cx_ki_z)
        cx_kd = self._axis_gains(self._config.cx_kd_xy, self._config.cx_kd_z)
        v_ref = (
            v_path_ref + cx_kp * e_pos + cx_ki * self._cx_integral + cx_kd * e_pos_dot
        ).astype(np.float32)
        v_ref = self._clip_horizontal_and_vertical(
            v_ref,
            self._config.max_v_ref_xy,
            self._config.max_v_ref_z,
        )

        # CV velocity PID: velocity error becomes desired world-frame acceleration.
        e_vel = v_ref - v_act
        if self._has_last_e_vel:
            e_vel_dot = (e_vel - self._last_e_vel) / self._dt
        else:
            e_vel_dot = np.zeros_like(e_vel, dtype=np.float32)
            self._has_last_e_vel = True
        self._last_e_vel = e_vel.astype(np.float32)
        self._cv_integral = np.clip(
            self._cv_integral + e_vel * self._dt,
            -self._config.cv_integral_limit,
            self._config.cv_integral_limit,
        ).astype(np.float32)
        cv_kp = self._axis_gains(self._config.cv_kp_xy, self._config.cv_kp_z)
        cv_ki = self._axis_gains(self._config.cv_ki_xy, self._config.cv_ki_z)
        cv_kd = self._axis_gains(self._config.cv_kd_xy, self._config.cv_kd_z)
        a_des = (cv_kp * e_vel + cv_ki * self._cv_integral + cv_kd * e_vel_dot).astype(
            np.float32
        )
        a_des = self._clip_horizontal_and_vertical(
            a_des,
            self._config.max_acc_xy,
            self._config.max_acc_z,
        )

        # Keep the working force convention: acceleration command is in world
        # frame, and positive z gravity compensation produces hover thrust.
        f_des = (self._mass * a_des).astype(np.float32)
        f_des[2] += self._mass * self._config.gravity * self._config.hover_thrust_scale

        r_des_body_to_world = self._desired_rotation(
            f_des,
            yaw_ref,
            roll_ref,
            pitch_ref,
        )
        euler_des = R.from_matrix(r_des_body_to_world).as_euler("xyz", degrees=False).astype(
            np.float32
        )
        euler_des[:2] = np.clip(euler_des[:2], -self._config.max_tilt, self._config.max_tilt)

        # The handout rotation error is useful for diagnostics or later small
        # attitude corrections.  We compute it but do not command body torques,
        # because this environment accepts attitude setpoints directly.
        self._last_rotation_error = self._rotation_error(
            r_des_body_to_world,
            r_act_body_to_world,
        ).astype(np.float32)

        # Yaw PID.  The output is treated as a correction around current yaw,
        # so gains must be tuned carefully to avoid overshoot.  wrap_angle keeps
        # the error on the shortest path across +/- pi.
        yaw_error = wrap_angle(yaw_ref - float(yaw_act))
        yaw_error_derivative = wrap_angle(yaw_error - self._last_yaw_error) / self._dt
        self._last_yaw_error = yaw_error
        self._yaw_integral = float(
            np.clip(
                self._yaw_integral + yaw_error * self._dt,
                -self._config.yaw_integral_limit,
                self._config.yaw_integral_limit,
            )
        )
        yaw_correction = (
            self._config.yaw_kp * yaw_error
            + self._config.yaw_ki * self._yaw_integral
            + self._config.yaw_kd * yaw_error_derivative
        )
        euler_des[2] = wrap_angle(float(yaw_act) + yaw_correction)

        thrust = self._force_to_thrust(f_des, r_act_body_to_world)
        self._debug = {
            "commanded_roll": float(euler_des[0]),
            "commanded_pitch": float(euler_des[1]),
            "yaw_error": float(yaw_error),
            "thrust": float(thrust),
            "position_error_norm": float(np.linalg.norm(e_pos)),
            "v_ref_norm": float(np.linalg.norm(v_ref)),
            "e_vel_norm": float(np.linalg.norm(e_vel)),
            "a_des_norm": float(np.linalg.norm(a_des)),
        }
        return self._smooth_and_clip(euler_des, thrust)

    @staticmethod
    def _axis_gains(xy_gain: float, z_gain: float) -> NDArray[np.float32]:
        return np.array([xy_gain, xy_gain, z_gain], dtype=np.float32)

    @staticmethod
    def _clip_horizontal_and_vertical(
        vec: Vec3,
        max_xy: float,
        max_z: float,
    ) -> NDArray[np.float32]:
        clipped = vec.astype(np.float32).copy()
        xy_norm = float(np.linalg.norm(clipped[:2]))
        if xy_norm > max_xy and xy_norm > 1e-6:
            clipped[:2] *= max_xy / xy_norm
        clipped[2] = float(np.clip(clipped[2], -max_z, max_z))
        return clipped.astype(np.float32)

    def _desired_rotation(
        self,
        force: Vec3,
        yaw: float,
        fallback_roll: float,
        fallback_pitch: float,
    ) -> NDArray[np.float32]:
        # R_des is body-to-world: each column is one desired body axis expressed
        # in world coordinates.  z_des follows the desired force direction; x/y
        # are chosen from the requested yaw heading as in the handout.  roll_ref
        # and pitch_ref are fallback values only if the force direction is
        # undefined; normal roll/pitch commands come from the force direction.
        force_norm = float(np.linalg.norm(force))
        if force_norm < 1e-6:
            return R.from_euler(
                "xyz",
                [fallback_roll, fallback_pitch, yaw],
                degrees=False,
            ).as_matrix().astype(np.float32)

        z_axis = (force / force_norm).astype(np.float32)
        x_heading = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        y_axis = np.cross(z_axis, x_heading)
        if float(np.linalg.norm(y_axis)) < 1e-6:
            y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
        y_axis /= float(np.linalg.norm(y_axis)) + 1e-6
        x_axis = np.cross(y_axis, z_axis)
        return np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)

    def _force_to_thrust(
        self,
        force: Vec3,
        r_act_body_to_world: NDArray[np.float32],
    ) -> float:
        # Positive collective thrust is along the actual body z-axis.  Since
        # f_des includes +mg compensation in world z, dot(f_des, body_z_actual)
        # gives the commanded positive thrust for hover when the drone is upright.
        # Using actual body z keeps thrust consistent with current attitude.
        body_z_actual = r_act_body_to_world[:, 2].astype(np.float32)
        return float(np.clip(float(force @ body_z_actual), self._thrust_min, self._thrust_max))

    @staticmethod
    def _rotation_error(
        r_des_body_to_world: NDArray[np.float32],
        r_act_body_to_world: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        error_matrix = 0.5 * (
            r_des_body_to_world.T @ r_act_body_to_world
            - r_act_body_to_world.T @ r_des_body_to_world
        )
        return np.array(
            [error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]],
            dtype=np.float32,
        )

    def _smooth_and_clip(self, euler: NDArray[np.float32], thrust: float) -> NDArray[np.float32]:
        action = self._last_action.copy()
        alpha = self._config.attitude_smoothing
        action[:2] = ((1.0 - alpha) * action[:2] + alpha * euler[:2]).astype(np.float32)
        action[2] = slew_angle(action[2], float(euler[2]), self._config.max_yaw_rate_step)
        beta = self._config.thrust_smoothing
        action[3] = float((1.0 - beta) * action[3] + beta * thrust)
        action[:2] = np.clip(action[:2], -self._config.max_tilt, self._config.max_tilt)
        action[2] = wrap_angle(float(action[2]))
        action[3] = float(np.clip(action[3], self._thrust_min, self._thrust_max))
        self._last_action = action.astype(np.float32)
        return self._last_action.copy()
