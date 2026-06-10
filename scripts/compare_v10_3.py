"""Paired-seed evaluation harness for comparing KaFa controllers (development tool).

The official scripts/evaluate.py draws random tracks (env.seed = -1), so two 20-run calls see
different randomisations and small controller differences drown in track-draw noise. This
harness seeds the environment ONCE with a fixed seed, so every controller evaluated with the
same (seed, n_runs) flies the exact same sequence of randomised tracks -- a paired comparison.

Run sequentially per controller (the acados C-code generation into c_generated_code/ is not
safe to run concurrently from two processes):

    pixi run python scripts/compare_v10_3.py --controller=KaFa_1500_v10_1.py --seed=42
    pixi run python scripts/compare_v10_3.py --controller=KaFa_1500_v10_3.py --seed=42
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


def evaluate(
    controller: str,
    config: str = "level2.toml",
    n_runs: int = 20,
    seed: int = 42,
    control_mode: str | None = None,
) -> None:
    """Fly ``controller`` over ``n_runs`` episodes on a fixed-seed track sequence.

    ``control_mode`` overrides the config's env.control_mode (the KaFa v9+ controllers need
    "attitude"; level0/level1 ship with "state").
    """
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    if control_mode is not None:
        cfg.env.control_mode = control_mode
    controller_path = Path(__file__).parents[1] / "lsy_drone_racing" / "control" / controller
    controller_cls = load_controller(controller_path)
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

    times: list[float | None] = []
    for run in range(n_runs):
        obs, info = env.reset()
        ctrl = controller_cls(obs, info, cfg)
        i = 0
        while True:
            curr_time = i / cfg.env.freq  # same flight-time bookkeeping as scripts/sim.py
            action = ctrl.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            finished = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
            if terminated or truncated or finished:
                break
            i += 1
        ctrl.episode_callback()
        ctrl.episode_reset()
        if obs["target_gate"] == -1:
            times.append(curr_time)
            print(f"run {run:2d}: FINISH  {curr_time:6.2f} s", flush=True)
        else:
            times.append(None)
            print(f"run {run:2d}: FAIL    {curr_time:6.2f} s at gate {int(obs['target_gate'])}",
                  flush=True)
    env.close()

    ok = [t for t in times if t is not None]
    print(f"\n=== {controller} | {config} | seed {seed} ===")
    if ok:
        print(
            f"finish {len(ok)}/{n_runs} ({100 * len(ok) / n_runs:.0f}%) | "
            f"avg {np.mean(ok):.3f} s | best {np.min(ok):.3f} s | worst {np.max(ok):.3f} s"
        )
    else:
        print(f"finish 0/{n_runs}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(evaluate, serialize=lambda _: None)
