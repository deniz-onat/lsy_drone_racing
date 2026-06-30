"""Self-contained controller package for KaFa_1500_v12.

v12 is a clean, standalone clone of KaFa_1500_v11_1: the v11 tunnel-constrained MPCC flying
v10.6's guarded-smoothed, parity-capped reference. All code, parameters, and settings it needs
live in this package; it imports nothing from any other ``KaFa_*`` version and keeps working if
every other controller is deleted.

This is a faithful, behaviour-preserving consolidation of v11_1's transitive dependency tree
(kafa1500_v6 -> KaFa_v8 -> KaFa_v9_1 -> KaFa_v10.x -> KaFa_v11/-v11_1). The flattened modules
were verified to produce bit-identical control output to KaFa_1500_v11_1.
"""
