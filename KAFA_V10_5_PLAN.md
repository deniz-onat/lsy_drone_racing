# Plan: KaFa v10.5 (v10.2 anchor × v10.4 launch/continuity) and v10.51 (+ Kalman filters)

## 0. Why this merge is real (lineage facts)

The v10.x tree is not linear. v10.2 and v10.3 are **sibling branches off v10.1**:

```
v10.1 ──> v10.2   dynamics-aware progress anchor (sharp-slalom gate-skip fix)
      └─> v10.3   replan continuity (rebase)  ──> v10.4  fast launch + honest cold start
                                                          + reactive gate caps  (8.01 s flagship)
```

Consequences:

- v10.4 still anchors progress with the **global** geometric projection
  (`KaFa_1500_v10_4.py:126`, `self._path.project(frame.pos, self._s)`) — the exact mechanism
  v10.2 proved can teleport the anchor ~1–2 m across a path fold on sharp slaloms and skip the
  gate. v10.4's 9/10 on the sharp-slalom edge track came from launch fixes; the fold-teleport
  failure mode is structurally still present mid-race.
- v10.2 has none of v10.4's launch speed (mini-takeoff, hot ramp, honest cold start), none of
  v10.3's replan continuity (it cold-resets the solver at every reveal replan), and no reactive
  gate caps.

**v10.5 = v10.4's everything + v10.2's predicted-progress anchor.** The two mechanisms touch
disjoint code: the anchor changes one line of `_track_action` plus two tiny accessors; all of
v10.4's launch/replan machinery is untouched. The OCP is byte-for-byte identical in all
versions, so the compiled acados solver is shared with v10.4 (same horizon 18 → same
dimensions → reuse the `kafa_v10_4` codegen namespace; no new `_build_*` needed).

**v10.51 = v10.5 + online estimation.** Not measurement filtering — sim obs are noise-free
(verified in `envs/race_core.py`: exact pos/vel/quat, no sensor noise). The KF targets the one
uncertainty the architecture currently ignores: level2 randomizes drone mass ±5 g on a ~30 g
airframe (±15–18% — `config/level2.toml:118-122`) and inertia, while the controller maps MPCC
accelerations to thrust with the **nominal** `self._mass`. Every commanded acceleration is
systematically scaled by `m_nom/m_true`; at gate speed this is overshoot/undershoot the
contouring weights currently have to absorb.

Expectation setting: the v10.4 cockpit ledger proved the ±0.15 m reveal-correction ceiling
binds at every gate and is not beatable by cockpit knobs. v10.51's KF does **not** attack that
ceiling — it attacks the mass/inertia axis (cleaner thrust mapping → less overshoot at a given
speed → maybe a slightly higher robust W_LAG/ramp). Sub-7 remains a v11 problem.

---

## 1. v10.5 — merge plan

### 1.1 Files (thin-subclass convention, mirroring v10.2/v10.4)

```
lsy_drone_racing/control/KaFa_v10_5/
    __init__.py
    cockpit.py       # re-export v10.4 knobs (*-import) + PROJ_BAND_M
    settings.py      # MPCCSettings(v10.4) + proj_band_m; ControllerSettings binding
    arc_path.py      # GateArcPath(KaFa_v10_4.arc_path.GateArcPath) + project_near()  [~5 lines, copy v10.2]
    mpcc.py          # MPCC(KaFa_v10_4.mpcc.MPCC) + predicted_progress()              [~5 lines, copy v10.2]
lsy_drone_racing/control/KaFa_1500_v10_5.py   # KaFa1500V105(KaFa1500V104)
```

`predicted_progress()` reads `self._x_sol[6, 1]` — present and maintained by both the v10.4
honest cold start (sets `_x_sol` on success) and the v10.3 `rebase` (rewrites row 6). No solver
change.

### 1.2 The merged `_track_action`

v10.4's flow verbatim (nominal-gate snapshot, reactive caps, rebase-vs-set_path branch, ramp),
with exactly one substitution — the anchor line `self._s = self._path.project(frame.pos, self._s)`
becomes v10.2's anchor:

```python
th_pred = self._mpcc.predicted_progress()
if th_pred is None:          # first solve of a plan (honest cold start pending): geometric
    self._s = self._path.project(frame.pos, self._s)
else:                        # dynamics-feasible anchor, geometric correction within the band
    self._s = self._path.project_near(frame.pos, th_pred, self._proj_band)
```

Interaction analysis (why this composes cleanly):

1. **Cold start:** after `set_path`, `_x_sol is None` → `predicted_progress()` returns None →
   geometric fallback. Identical to v10.2's handling; the v10.4 honest cold start then seeds an
   anchored solution, so from step 2 the predicted-progress anchor is live.
2. **Rebase (the new case neither parent had):** `rebase` keeps `_x_sol` and re-anchors its
   progress row onto the new path, so `predicted_progress()` is already in **new-path arc
   coordinates** — the anchor works across replans with no special case. This is strictly
   better than v10.2, which lost the prediction (and the fold protection) for one step at every
   replan.
3. **Known residual risk — the rebase's own projection:** `rebase(path, s0)` receives
   `s0 = path.project(frame.pos, 0.0)` (global search) and forward-projects the horizon. A
   reveal replan **while inside a fold** could in principle anchor `s0` on the far leg. Old and
   new paths coincide to ≤~0.2 m, so the fix (if needed) is to compute
   `s0 = path.project_near(frame.pos, th_pred_old, proj_band)` when a prediction exists. Do
   **not** ship this preemptively — instrument first (see 1.4); v10.2's ledger showed every
   speculative anchor constraint regressed.

### 1.3 Tuning: PROJ_BAND_M does not transfer blindly

v10.2 measured 0.6 m at v10.1's ~8.3 s pace: just above the legitimate per-step fold advance
(~0.7 m/step) and below the fold self-approach gap (~1 m). v10.4 flies faster, and the
legitimate per-step advance scales with speed. Plan: start at 0.6, log the max single-step
anchor motion on the slalom, and sweep {0.6, 0.7, 0.8} only if the telemetry shows clamping
(anchor pinned at band edge in runs that finish). Above ~1.0 the teleport returns — hard upper
bound from v10.2's ledger.

### 1.4 Instrumentation (build before tuning)

- Per-step anchor jump `|s_t - s_{t-1}|`: max + histogram, logged per episode. Success signature
  from v10.2: max jump ~0.7 m (vs ~2.0 m for a teleport).
- Flag for "anchor at band edge" rate, to drive the PROJ_BAND_M sweep.
- Keep the cyan predicted-horizon overlay (inherited from v10.3's `render_callback`).

### 1.5 Evaluation (paired-seed, same protocol as the v10.4 ledger)

All under `pixi run` (acados env). Extend `scripts/compare_v10_3.py` into
`scripts/compare_v10_5.py` (paired track sequences, same seeds for both controllers).

| Track | Runs | Baseline | Accept v10.5 if |
|---|---|---|---|
| level2 | seeds 42/7/123 × 20 | v10.4: 52/60 @ 8.01 s | ≥ 52/60, time within paired noise of 8.01 s |
| level2_sharp_slalom | 3 seeds × 20 | v10.4: 9/10 (1 seed) | ≥ v10.4 paired; **zero anchor jumps > 1 m** |
| level2_sharp_hairpin, level2_sharp_boxloop, level2_inout_* | 1 seed × 10 each | measure v10.4 first | no regression |
| stress_synth_01–03 | 1 seed × 10 each | measure v10.4 first | no regression |

Gate for proceeding to v10.51: level2 row passes AND slalom anchor telemetry is clean.

---

## 2. v10.51 — v10.5 + Kalman filters

Two estimators, shipped independently switchable so the eval attributes gains correctly.

### 2.1 Thrust-gain KF (primary — targets the mass randomization)

Physics: we command `thrust = m_nom * (a_cmd + g·ez)`; the airframe realizes
`a_real = (m_nom/m_true)·(a_cmd + g·ez) − g·ez`. The mismatch is **multiplicative in
(a_cmd + g·ez)**, not additive — so estimate the scalar gain `k = m_nom/m_true` (k ∈ ~[0.85, 1.18]
under ±5 g), not a bias vector.

- **State:** scalar `k̂`, random walk: `k̂_{t+1} = k̂_t + w`, `Q = q` (small).
- **Measurement (each NAVIGATE tick):** `a_meas = (vel_t − vel_{t−1}) / dt` (sim vel is exact,
  so the finite difference is clean — dt = 1/env.freq). Model:
  `a_meas + g·ez = k · (a_cmd_prev + g·ez) + v`, a 3-rows-per-tick linear measurement of scalar
  k with `H_t = a_cmd_prev + g·ez`. Standard scalar-state KF update; with constant Q/R this has
  a cheap closed form (no matrix library needed).
- **Apply:** `thrust_vector = (m_nom / k̂) * (accel + g·ez)` in `_track_action` — i.e. divide
  the mass by the estimated gain. The OCP itself is untouched: `a_max` is baked into the
  compiled constraint set at codegen (`a_max**2 − Σthrust²` in `con_h`), so we deliberately do
  **not** retune constraints online — nominal `a_max` stays as a conservative envelope (worst
  case `m_true > m_nom` means we have ~15% less authority than the OCP believes; the tilt cap
  at 0.45 leaves margin; verify in the heavy-mass eval row).
- **Safeguards:** init `k̂ = 1`, clamp to [0.8, 1.2]; **freeze updates** (i) during TAKEOFF and
  for the first ~5 NAVIGATE ticks (hand-off transient), (ii) on any tick where the attitude
  command saturated (a_cmd was not what flew), (iii) across the replan tick (vel finite-diff
  spans the rebase — actually vel is physical, this is fine; freeze only (i) and (ii)).
- **Convergence target:** time constant ~0.3–0.5 s via Q/R choice, so k̂ is converged during
  the 2.4 s launch ramp — before the first high-speed gate. Validate against `m_true` read from
  the sim (validation only, never control).

### 2.2 Latency-compensation predict step (secondary, flag-gated)

One KF predict step used as delay compensation: before solving, propagate
`pos ← pos + vel·dt + ½·â·dt²`, `vel ← vel + â·dt` with `â = k̂·(a_cmd_prev + g·ez) − g·ez`,
where dt is the obs→actuation delay (one control tick in sim). At ~3 m/s and 50 Hz this is
~6 cm — the same order as the gate corridors, so it may matter, but it interacts with the
contouring tuning. Ship OFF by default; measure in a separate eval column. Drop it if the
paired runs don't show a gain (the v10.x ledgers are full of "obviously good" knobs that
landed on the frontier, not above it).

Explicit non-goals (from the earlier Kalman analysis): no filtering of pos/vel (noise-free in
sim; on hardware the Crazyflie already runs its onboard EKF — `real_race_env.py` sets
`stabilizer.estimator = 2` — and the ROS `drone_estimators` stack supplies vel); no gate-pose
filtering (reveals are one-shot exact snaps at sensor range, nothing to filter — the reactive
caps are the right tool and stay).

### 2.3 Files

```
lsy_drone_racing/control/KaFa_v10_51/
    __init__.py
    cockpit.py       # re-export v10.5 knobs + KF_Q, KF_R, KF_CLAMP, KF_FREEZE_TICKS,
                     # KF_ENABLED, LATENCY_COMP_ENABLED
    settings.py      # EstimatorSettings dataclass + ControllerSettings binding
    estimator.py     # ThrustGainKF: update(vel, dt, a_cmd_prev) -> k̂ ; predict_state(...)
lsy_drone_racing/control/KaFa_1500_v10_51.py  # KaFa1500V1051(KaFa1500V105):
                     # _track_action = update KF -> (optional state predict) -> v10.5 flow
                     # with thrust mapping scaled by 1/k̂; reset() clears the KF
```

`estimator.py` is pure numpy, unit-testable in isolation (feed synthetic (a_cmd, a_real) pairs
with known k, assert convergence + clamp + freeze behavior) — write that test first.

### 2.4 Evaluation

| Config | Purpose | Accept if |
|---|---|---|
| level2, seeds 42/7/123 × 20, KF on vs v10.5 | no-harm at nominal | finish ≥ v10.5, time within noise; k̂ traces ≈ m_nom/m_true ±3% by t=2 s |
| `stress_mass.toml` (new: level2 with drone_mass ±0.008, all else equal) | amplified signal | finish strictly > v10.5 on paired seeds |
| heavy-mass tail check (force m_true = m_nom+0.005 via fixed seed pick) | authority margin (2.1 caveat) | no new crash class at gates 2/3 |
| level2 + latency comp on (separate column) | 2.2 go/no-go | keep only if finish/time improves paired |

Telemetry: log k̂(t) per episode + the true sampled mass; plot estimate error. If k̂ converges
correctly but finish doesn't move, the mass axis wasn't binding at this pace — record that in
the cockpit ledger (it then becomes the justification to re-probe a warmer W_LAG/ramp with the
KF on, which is the actual payoff hypothesis).

---

## 3. Order of work

1. **v10.5 scaffolding + merge** (arc_path/mpcc accessors are verbatim v10.2 copies; the
   `_track_action` override is the only thinking). Smoke-run level0/level2 single episodes.
2. **Anchor telemetry + slalom validation**; PROJ_BAND_M sweep only if clamping shows.
3. **Full v10.5 paired eval** (1.5 table). Freeze v10.5. Ledger the numbers in
   `KaFa_v10_5/cockpit.py` (house style).
4. **ThrustGainKF + unit tests**, then v10.51 wiring with KF on / latency comp off.
5. **v10.51 eval** (2.4 table), incl. the new `config/stress_mass.toml`.
6. **Latency-comp column**; keep or condemn, ledger either way.
7. If KF passes: one bounded re-probe of {W_LAG, RAMP_START/RAMP_S} with KF on — the ledgered
   frontier was measured with the mass error uncorrected, so the robust knee may have moved.

Risks worth naming: (a) the anchor + hot launch may interact on the slalom *during* the ramp
(v10.2 never flew this fast this early) — the band sweep covers it; (b) finite-diff accel is
one tick stale, which biases k̂ slightly during jerks — the freeze rules + slow time constant
cover it; (c) per the ledger, every speed knob so far landed ON the frontier — v10.51's win
condition is moving the frontier via model fidelity, and step 7 is where that either shows or
doesn't.
