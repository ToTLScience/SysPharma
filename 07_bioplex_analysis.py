"""
07_bioplex_analysis.py
======================
Count shared first-degree BioPlex 3.0 HCT116 interaction partners
between pairs of nominated target proteins.  Used in Fig. 5B.

The BioPlex 3.0 network was downloaded from:
  https://bioplex.hms.harvard.edu
File: BioPlex_HCT116_Network_5_5K_Dec_2019.tsv
Columns: GeneA, GeneB, UniprotA, UniprotB, pInt, pNI, pW

Input  (data/)
------
- BioPlex_HCT116_Network_5_5K_Dec_2019.tsv

Output (results/)
-------
- bioplex_shared_partners.csv  : n × n matrix of shared partner counts
- bioplex_partner_details.csv  : identity of shared partners for each pair
"""

import pandas as pd
import os
from itertools import combinations

DATA = "data"
OUT  = "results"

TARGETS = ["BCL2L2", "MCL1", "GPX4", "CLPX", "LONP1"]  # BCL2L2 = BCL-W

# ── Load BioPlex network ──────────────────────────────────────────────────────
print("Loading BioPlex 3.0 HCT116 …")
bp = pd.read_csv(
    os.path.join(DATA, "BioPlex_HCT116_Network_5_5K_Dec_2019.tsv"),
    sep="\t",
    usecols=["GeneA", "GeneB"]
)
print(f"  {len(bp):,} interactions loaded")

# Build adjacency: gene → set of partners
adj = {}
for _, row in bp.iterrows():
    adj.setdefault(row["GeneA"], set()).add(row["GeneB"])
    adj.setdefault(row["GeneB"], set()).add(row["GeneA"])

# ── Count shared partners for each target pair ────────────────────────────────
records   = []
det_rows  = []

for t1, t2 in combinations(TARGETS, 2):
    p1 = adj.get(t1, set())
    p2 = adj.get(t2, set())
    shared = (p1 & p2) - set(TARGETS)   # exclude the targets themselves
    n = len(shared)
    records.append({"target_1": t1, "target_2": t2, "n_shared_partners": n})
    for partner in sorted(shared):
        det_rows.append({"target_1": t1, "target_2": t2, "shared_partner": partner})
    print(f"  {t1} / {t2}: {n} shared partner(s)"
          + (f" — {sorted(shared)}" if shared else ""))

pd.DataFrame(records).to_csv(
    os.path.join(OUT, "bioplex_shared_partners.csv"), index=False)

if det_rows:
    pd.DataFrame(det_rows).to_csv(
        os.path.join(OUT, "bioplex_partner_details.csv"), index=False)
else:
    print("  (No shared partners found across any pair)")

print(f"\nOutputs written to {OUT}/")
