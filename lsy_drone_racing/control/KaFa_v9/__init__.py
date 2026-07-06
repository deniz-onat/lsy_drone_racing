"""MPCC drone racing controller package for KaFa_1500_v9.

v9 keeps the parts of KaFa v8 that work well (the global, gate-aware, obstacle/funnel-routed
spline, used here purely as path geometry, and the dedicated vertical takeoff) and replaces
v8's hand-tuned speed caps and cascaded-PID tracker with a model-predictive contouring
controller (KaFa_v9.mpcc). The MPCC flies as fast as the drone's thrust/tilt limits allow
along whatever path it's given, so speed generalises to arbitrary track geometry instead of
being a tuned constant. Only the MPCC knobs live in KaFa_v9.cockpit; the path and takeoff
tuning is inherited from KaFa_v8.
"""
