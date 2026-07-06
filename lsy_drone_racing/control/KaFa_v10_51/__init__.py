"""Gate-aware time-optimal MPCC package for KaFa_1500_v10_51 (v10.5 + online mass estimation).

v10.51 is v10.5 with a thrust-gain Kalman filter. The racing pipeline -- the OCP, the gate-aware
contouring, the launch, the replan-continuity rebase, the reactive caps, and the v10.2/v10.5
dynamics-aware anchor -- is all v10.5's and unchanged; the compiled acados solver is shared. The
only addition is online estimation of the scalar gain k = m_nom/m_true (the mass-randomisation
mismatch the MPCC->thrust mapping otherwise ignores) and a thrust mapping that divides the mass by
the estimate. Two estimators are shipped independently switchable so an eval can attribute gains:

1. Thrust-gain KF (PRIMARY, on by default): estimates k from the finite-difference acceleration
   and the actually-applied specific force, and the controller maps thrust with m_nom/k_hat.
2. Latency-compensation predict step (SECONDARY, off by default): one KF predict step as
   obs->actuation delay compensation before each solve.

The KF does NOT attack the +/-0.15 m reveal-correction ceiling (proven binding and not beatable by
cockpit knobs in the v10.4 ledger) -- it attacks the mass axis (cleaner thrust mapping -> less
overshoot at a given speed). See KaFa_1500_v10_51 and KaFa_v10_51.estimator for the physics and
the closed-loop-correctness argument. REQUIRES the acados environment -- run under ``pixi run``.
"""
