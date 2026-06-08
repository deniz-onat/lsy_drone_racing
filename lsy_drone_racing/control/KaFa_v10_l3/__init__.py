"""Level-3 search-then-navigate package for KaFa_1500_v10_l3.

Adds a SEARCH phase in front of v10's TAKEOFF -> NAVIGATE flow. On level3 the gates and
obstacles are randomised across the whole arena and their true positions are only revealed
within the drone's sensor range (horizontally), so the drone must explore first. SEARCH flies
an expanding circular sweep (Archimedean spiral) over the arena -- driven by the SAME v10
time-optimal MPCC used for navigation, just handed a spiral path instead of the race path --
until every gate has been seen; then it hands off to v10's normal NAVIGATE. Detected gates and
obstacles are drawn in the sim. Search geometry/speed knobs live in KaFa_v10_l3.cockpit; the
spiral path builder is KaFa_v10_l3.search.
"""
