"""
05_hallmark_scoring.py
======================
Score all 42 MSigDB Hallmark pathway gene sets using the same six
dimensions and composite formula, normalised within the drug-proteome
universe (n = 9,723 genes).  Used in Fig. S1A.

Input  (data/, results/)
------
- h.all.v2023.1.Hs.symbols.gmt   MSigDB Hallmark gene sets (GMT format)
- dimensions_all_proteome.csv     from 01_prepare_dimensions.py

Output (results/)
-------
- hallmark_scores_pathway_norm.csv
"""

import numpy as np
import pandas as pd
import os

DATA = "data"
OUT  = "results"
GMT  = os.path.join(DATA, "h.all.v2023.1.Hs.symbols.gmt")

def parse_gmt(path):
    """Parse a GMT file; return {pathway_name: [gene, ...]}."""
    pathways = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            name  = parts[0]
            genes = parts[2:]
            pathways[name] = genes
    return pathways

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading Hallmark gene sets …")
hallmarks = parse_gmt(GMT)
print(f"  {len(hallmarks)} Hallmark gene sets")

print("Loading dimension table …")
df = pd.read_csv(os.path.join(OUT, "dimensions_all_proteome.csv"), index_col=0)
df = df.dropna(subset=["D", "ZC"]).copy()

MITO_ZC_MEAN = df.loc[df["is_mito"], "ZC"].mean()
df["ZC_dev"] = (df["ZC"] - MITO_ZC_MEAN).abs()

CR_THRESHOLD = 4.0

# ── Score each Hallmark pathway ───────────────────────────────────────────────
records = []
for pw_name, pw_genes in sorted(hallmarks.items()):
    sub = df[df.index.isin(pw_genes)]
    if len(sub) < 3:
        continue
    records.append({
        "pathway":  pw_name,
        "n_genes":  len(sub),
        "E_raw":    sub["E"].median(),
        "D_raw":    sub["D"].mean(),
        "R_raw":    (sub["CR"] >= CR_THRESHOLD).mean(),
        "F_raw":    sub["F"].mean(),
        "ZC_raw":   abs(sub["ZC"].mean() - MITO_ZC_MEAN),
    })

hw_df = pd.DataFrame(records).set_index("pathway")

def minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=s.index)

WEIGHTS = {"E": 0.30, "D": 0.25, "R": 0.15, "F": 0.05, "ZC": 0.05}

hw_df["E_n"]  = minmax(hw_df["E_raw"])
hw_df["D_n"]  = minmax(hw_df["D_raw"])
hw_df["R_n"]  = minmax(hw_df["R_raw"])
hw_df["F_n"]  = minmax(hw_df["F_raw"])
hw_df["ZC_n"] = minmax(hw_df["ZC_raw"])

# Note: L dimension not included for Hallmark scoring (cysteine ligandability
# is less meaningful for non-organelle-restricted gene sets of mixed size)
hw_df["score"] = (
    WEIGHTS["E"]  * hw_df["E_n"] +
    WEIGHTS["D"]  * hw_df["D_n"] +
    WEIGHTS["R"]  * hw_df["R_n"] +
    WEIGHTS["F"]  * hw_df["F_n"] +
    WEIGHTS["ZC"] * hw_df["ZC_n"]
)
hw_df["rank"] = hw_df["score"].rank(ascending=False, method="first").astype(int)
hw_df = hw_df.sort_values("rank").reset_index()

hw_df.to_csv(os.path.join(OUT, "hallmark_scores_pathway_norm.csv"), index=False)
print(f"Top 5 Hallmark pathways:")
print(hw_df[["pathway","score","rank"]].head(5).to_string(index=False))
print(f"\nOutput: {OUT}/hallmark_scores_pathway_norm.csv")
