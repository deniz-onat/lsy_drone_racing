"""Gate-aware time-optimal MPCC package for KaFa_1500_v10_2.

v10.2 is v10.1 made *robust on sharp slaloms*. It keeps everything that makes v10.1 work -- v8's
gate-aware planner, the vertical takeoff, the real-time acados SQP-RTI time-optimal MPCC, the
-mu*vth progress reward, the friction-circle curvature cap, and the per-stage gate-aware contouring
weight -- and changes ONE thing: how the MPCC's progress state is anchored to the drone.

Why: on a sharp slalom (e.g. config/level2_sharp_slalom.toml), where the drone enters and leaves a
gate from the same side, the planned path folds back on itself into a cusp at the gate. v10/v10.1
anchor progress at the GLOBAL nearest path point over a 2.0 m forward arc window
(KaFa_v10.arc_path.ArcPath.project). When the drone is near that fold, a far-along-the-arc leg of
the cusp is spatially closer than the gate apex, so the search snaps the anchor ~1-2 m forward
across the fold in one step -- progress is hard-anchored PAST the gate and the drone skips it.

Constraining the forward search does not work: at the fold the legitimate foot-point genuinely
advances ~0.7 m/step as the drone rounds the doubled-over path, so any cap tight enough to block the
skip also clamps the legitimate motion and the controller stalls/destabilises. v10.2 instead anchors
progress to the SOLVER'S OWN predicted progress (KaFa_v10_2.mpcc), which is dynamics-feasible and so
cannot teleport, and lets a geometric search correct it only within +/- PROJ_BAND_M
(KaFa_v10_2.arc_path.GateArcPath.project_near). The far fold leg lies outside the band and can never
be selected; the gate-apex motion lies inside it, so tracking is unchanged. The worst single-step
anchor jump drops from ~2.0 m to ~0.7 m (the skip is gone) at finish within run-to-run noise of
v10.1, with no catastrophic regression at any seed. Straights, ordinary gates, and the v10.1 speed
budget are untouched; only the progress anchor changes.
"""
