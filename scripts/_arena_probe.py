"""TEMPORARY probe: bounding box of the flown trajectory vs the v11.2 arena onsets.

Flies a controller on a fixed-seed level2 sequence and reports, per run, the min/max world x and
y the drone actually reaches -- so we can see whether the natural racing line enters v11.2's
keep-in ramp band (onset x ~ +/-2.15, y ~ +/-1.15 at the defaults) or stays clear of it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)

# v11.2 defaults (KaFa_v11_2.cockpit): onset = sim_wall -/+ inset -/+ ramp
INSET, RAMP = 0.10, 0.30
ONSET = {"x": 2.5 - INSET - RAMP, "y": 1.5 - INSET - RAMP}  # 2.10, 1.10


def probe(controller: str = "KaFa_1500_v11.py", seed: int = 42, n_runs: int = 5) -> None:
    """Fly and report the trajectory bounding box vs the arena onsets."""
    cfg = load_config(Path(__file__).parents[1] / "config" / "level2.toml")
    cfg.sim.render = False
    cpath = Path(__file__).parents[1] / "lsy_drone_racing" / "control" / controller
    controller_cls = load_controller(cpath)
    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
    )
    env = JaxToNumpy(env)
    print(f"controller={controller} seed={seed}  onset |x|>{ONSET['x']} |y|>{ONSET['y']}")
    for run in range(n_runs):
        obs, info = env.reset()
        ctrl = controller_cls(obs, info, cfg)
        lo = np.array([np.inf, np.inf])
        hi = np.array([-np.inf, -np.inf])
        ref_lo = np.array([np.inf, np.inf])
        ref_hi = np.array([-np.inf, -np.inf])
        while True:
            action = ctrl.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            p = np.asarray(obs["pos"])[:2]
            lo, hi = np.minimum(lo, p), np.maximum(hi, p)
            path = getattr(ctrl, "_path", None)
            if path is not None and getattr(path, "_pts", None) is not None:
                pts = np.asarray(path._pts)[:, :2]
                ref_lo = np.minimum(ref_lo, pts.min(0))
                ref_hi = np.maximum(ref_hi, pts.max(0))
            if terminated or truncated or ctrl.step_callback(
                action, obs, reward, terminated, truncated, info
            ):
                break
        ctrl.episode_callback()
        ctrl.episode_reset()
        in_band_y = max(hi[1] - ONSET["y"], -ONSET["y"] - lo[1])
        flag = " <-- ENTERS BAND" if in_band_y > 0 else ""
        print(
            f"run {run}: drone y[{lo[1]:+.2f},{hi[1]:+.2f}] x[{lo[0]:+.2f},{hi[0]:+.2f}]"
            f" | REF y[{ref_lo[1]:+.2f},{ref_hi[1]:+.2f}] x[{ref_lo[0]:+.2f},{ref_hi[0]:+.2f}]"
            f"{flag}"
        )
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(probe, serialize=lambda _: None)
