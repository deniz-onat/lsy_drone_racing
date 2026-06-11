"""Paired-seed evaluation harness for the KaFa v10.5 / v10.51 line (development tool).

Extends scripts/compare_v10_3.py with two additions, both needed by the v10.5/v10.51 plan:

1. ANCHOR TELEMETRY. If the controller exposes ``anchor_telemetry()`` (v10.5+), its per-episode
   progress-anchor diagnostics are collected: the worst single-step anchor jump, the count of
   jumps > 1 m (the fold-teleport signature -- must be ZERO on the sharp slalom), and the
   band-edge rate (drives the PROJ_BAND_M sweep). Aggregated across all episodes at the end.

2. MASS / GAIN TELEMETRY. If the controller exposes ``estimator_telemetry()`` (v10.51), the
   final thrust-gain estimate k_hat per episode is collected and its per-episode spread reported.
   (The env's randomised drone mass is NOT host-readable -- the randomiser writes it into the
   device-side dynamics but ``sim.data.params.mass`` re-syncs to nominal, a JAX lazy/sync artifact
   -- so there is no clean ground-truth k to diff against. The k_hat SPREAD across episodes is the
   evidence the filter tracks per-episode dynamics; the KF's actual value is decided by the paired
   finish/time delta vs the KF-off controller, which is the whole point of the paired protocol.)

Same fixed-seed paired protocol as compare_v10_3.py: the env is seeded ONCE, so every controller
run with the same (seed, n_runs) flies the exact same randomised track sequence. Run sequentially
per controller (acados C-code generation into c_generated_code/ is not concurrency-safe):

    pixi run python scripts/compare_v10_5.py --controller=KaFa_1500_v10_4.py --seed=42
    pixi run python scripts/compare_v10_5.py --controller=KaFa_1500_v10_5.py --seed=42
    pixi run python scripts/compare_v10_5.py --controller=KaFa_1500_v10_5.py \
        --config=level2_sharp_slalom.toml --seed=42
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
    anchor: list[dict] = []
    est: list[dict] = []
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
        # Pull telemetry BEFORE the episode callbacks reset the controller.
        a_tel = ctrl.anchor_telemetry() if hasattr(ctrl, "anchor_telemetry") else None
        e_tel = ctrl.estimator_telemetry() if hasattr(ctrl, "estimator_telemetry") else None
        ctrl.episode_callback()
        ctrl.episode_reset()

        finished_run = obs["target_gate"] == -1
        times.append(curr_time if finished_run else None)
        msg = (
            f"run {run:2d}: {'FINISH' if finished_run else 'FAIL  '}  {curr_time:6.2f} s"
            + ("" if finished_run else f" at gate {int(obs['target_gate'])}")
        )
        if a_tel is not None:
            anchor.append(a_tel)
            msg += (
                f" | anchor max {a_tel['max_jump_m']:.2f} m, >1m {a_tel['n_jumps_gt_1m']},"
                f" edge {100 * a_tel['band_edge_rate']:.0f}%"
            )
        if e_tel is not None:
            est.append(e_tel)
            msg += f" | k_hat {e_tel['k_hat']:.3f}"
        print(msg, flush=True)
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
    if anchor:
        max_jump = max(a["max_jump_m"] for a in anchor)
        tot_gt1 = sum(a["n_jumps_gt_1m"] for a in anchor)
        edge = float(np.mean([a["band_edge_rate"] for a in anchor]))
        print(
            f"anchor: worst single-step jump {max_jump:.3f} m | episodes with a >1 m jump "
            f"{sum(1 for a in anchor if a['n_jumps_gt_1m'] > 0)}/{len(anchor)} "
            f"({tot_gt1} steps total) | mean band-edge rate {100 * edge:.1f}%"
        )
    if est:
        ks = np.array([e["k_hat"] for e in est])
        print(
            f"estimator: k_hat mean {ks.mean():.3f} | spread [{ks.min():.3f}, {ks.max():.3f}] "
            f"(std {ks.std():.3f}) over {len(ks)} episodes"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(evaluate, serialize=lambda _: None)
