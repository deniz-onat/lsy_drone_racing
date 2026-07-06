"""Self-contained controller package for KaFa_1500_v12_s.

v12_s is a standalone clone of the KaFa_1500_v12 package (the v11 tunnel-constrained MPCC flying
v10.6's guarded-smoothed, parity-capped reference) with one addition: a level-3 gate-search sweep
(``search.py``) flown between takeoff and navigation. All code, parameters, and settings it needs
live in this package; it imports nothing from any other ``KaFa_*`` version and keeps working if
every other controller is deleted.

The navigation stack is a faithful consolidation of v11_1's transitive dependency tree
(kafa1500_v6 -> KaFa_v8 -> KaFa_v9_1 -> KaFa_v10.x -> KaFa_v11/-v11_1); its control output is
unchanged from v12. Only the takeoff->navigate handoff now routes through the search phase.
"""
