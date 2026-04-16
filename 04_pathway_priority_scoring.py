"""
04_pathway_priority_scoring.py
==============================
Score the 38 MitoCarta level-2 pathway gene sets using pathway-level
summary statistics, then apply the same composite weighting formula.

Pathway-level dimension definitions
-------------------------------------
  E   = median |Chronos| × (1-PE)² across pathway members
  D   = mean  |log2FC|             across pathway members
  L   = fraction of pathway genes with ≥1 CRA site (DrugMap)
  R   = fraction of pathway genes with max CR ≥ 4 (SLC-ABPP)
  F   = fraction of pathway genes detected in half-life dataset
  ZC  = |pathway_mean_ZC - mitochondrial_proteome_mean_ZC|

Scoring formula (identical weights to gene-level):
  S = 0.30×E + 0.25×D + 0.20×L + 0.15×R + 0.05×F + 0.05×ZC

Each pathway-level dimension is min-max normalised to [0,1] across
the 38 pathways before weighting.

Inputs  (results/, data/)
------
- dimensions_all_proteome.csv    from 01_prepare_dimensions.py
- Human.MitoCarta3.0.xls         MitoPathways level-2 annotations

Output (results/)
-------
- pathway_priority_final.csv
"""

import numpy as np
import pandas as pd
import xlrd
import os

DATA = "data"
OUT  = "results"

CR_THRESHOLD = 4.0   # SLC-ABPP CR threshold for R dimension

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dimension table …")
df = pd.read_csv(os.path.join(OUT, "dimensions_all_proteome.csv"), index_col=0)
df = df[df["is_mito"] & df["D"].notna()].copy()

MITO_ZC_MEAN = df["ZC"].mean()

# ── Parse MitoPathways level-2 ────────────────────────────────────────────────
print("Parsing MitoPathways …")
wb = xlrd.open_workbook(os.path.join(DATA, "Human.MitoCarta3.0.xls"))
ws = wb.sheet_by_name("A Human MitoCarta3.0")
headers   = [ws.cell_value(0, c) for c in range(ws.ncols)]
sym_col   = headers.index("Symbol")
path_col  = headers.index("MitoCarta3.0_MitoPathways")

gene_to_pathways = {}
for r in range(1, ws.nrows):
    gene  = ws.cell_value(r, sym_col)
    paths = ws.cell_value(r, path_col)
    if not gene or not paths:
        continue
    gene_to_pathways[gene.strip()] = [p.strip() for p in str(paths).split(">")]

def extract_level2(pathway_string):
    """Return the level-2 component from a MitoPathways string.
    Format: 'OXPHOS > OXPHOS subunits' — level-1 > level-2.
    Some entries have deeper nesting; take the second element only.
    """
    parts = [p.strip() for p in pathway_string.split(">")]
    return parts[1] if len(parts) >= 2 else None

# Build level-2 pathway → gene mapping
pathway_genes = {}
for gene, path_list in gene_to_pathways.items():
    for path_str in path_list:
        lv2 = extract_level2(path_str)
        if lv2:
            pathway_genes.setdefault(lv2, set()).add(gene)

# Filter to pathways with ≥3 scored genes
MIN_GENES = 3
pathway_genes = {
    pw: genes & set(df.index)
    for pw, genes in pathway_genes.items()
    if len(genes & set(df.index)) >= MIN_GENES
}
print(f"  {len(pathway_genes)} level-2 pathways with ≥{MIN_GENES} scored genes")

# ── Compute pathway-level dimension scores ────────────────────────────────────
records = []
for pw, genes in sorted(pathway_genes.items()):
    sub = df.loc[list(genes)]
    n   = len(sub)

    E_pw  = sub["E"].median()
    D_pw  = sub["D"].mean()
    L_pw  = (sub["CRA"] >= 1).mean()          # fraction with ≥1 CRA site
    R_pw  = (sub["CR"] >= CR_THRESHOLD).mean() # fraction with CR ≥ 4
    F_pw  = sub["F"].mean()                    # fraction with measured t½
    zc_mean = sub["ZC"].mean()
    ZC_pw = abs(zc_mean - MITO_ZC_MEAN)

    records.append({
        "pathway":          pw,
        "n_genes":          n,
        "E_raw":            E_pw,
        "D_raw":            D_pw,
        "L_raw":            L_pw,
        "R_raw":            R_pw,
        "F_raw":            F_pw,
        "ZC_mean":          zc_mean,
        "ZC_deviation_raw": ZC_pw,
    })

pw_df = pd.DataFrame(records).set_index("pathway")

# ── Min-max normalise and score ───────────────────────────────────────────────
def minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=s.index)

WEIGHTS = {"E": 0.30, "D": 0.25, "L": 0.20, "R": 0.15, "F": 0.05, "ZC": 0.05}

pw_df["E_norm"]  = minmax(pw_df["E_raw"])
pw_df["D_norm"]  = minmax(pw_df["D_raw"])
pw_df["L_norm"]  = minmax(pw_df["L_raw"])
pw_df["R_norm"]  = minmax(pw_df["R_raw"])
pw_df["F_norm"]  = minmax(pw_df["F_raw"])
pw_df["ZC_deviation_norm"] = minmax(pw_df["ZC_deviation_raw"])

pw_df["score"] = (
    WEIGHTS["E"]  * pw_df["E_norm"] +
    WEIGHTS["D"]  * pw_df["D_norm"] +
    WEIGHTS["L"]  * pw_df["L_norm"] +
    WEIGHTS["R"]  * pw_df["R_norm"] +
    WEIGHTS["F"]  * pw_df["F_norm"] +
    WEIGHTS["ZC"] * pw_df["ZC_deviation_norm"]
)
pw_df["rank"] = pw_df["score"].rank(ascending=False, method="first").astype(int)
pw_df = pw_df.sort_values("rank").reset_index()

pw_df.to_csv(os.path.join(OUT, "pathway_priority_final.csv"), index=False)
print(f"\nTop 5 pathways:")
print(pw_df[["pathway", "score", "rank"]].head(5).to_string(index=False))
print(f"\nOutput written to {OUT}/pathway_priority_final.csv")
