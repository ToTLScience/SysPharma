"""
08_jacques_proteomics.py
========================
Reanalyse published quantitative proteomics data from Jacques et al. 2020
(Genetics, DOI:10.1534/genetics.119.302851) examining ONC212 treatment
in WT versus CLPP-knockdown NALM6 pre-B leukemia cells.

The data (Supplementary Table S2) report protein-level log2 fold-changes
for ONC212 treatment (compound/DMSO) in both conditions.  We extract:
  - WT log2FC
  - CLPP-KD log2FC
for all detected proteins, and annotate targets of interest.

These values are used in Fig. 3H (scatter plot showing CLPP-dependence
of ONC212 proteome response).

Input  (data/)
------
- Table_S2_Jacques_et_al_accepted.xlsx
  Sheet: 'NALM6-log2-fold-change-values'
  Key columns: 'Gene symbol', 'WT log2 fold-change', 'CLPP sh  log2 fold-change'

Output (results/)
-------
- jacques_log2fc_nalm6.csv
"""

import pandas as pd
import numpy as np
import os

DATA = "data"
OUT  = "results"

TARGETS_OF_INTEREST = {
    "CLPX":    "CLPX (ClpXP chaperone)",
    "LONP1":   "LONP1 (matrix protease)",
    "CLPP":    "CLPP (ClpXP protease core)",
    "ATP5IF1": "ATP5IF1 (Complex V inhibitor; ATP synthase IF1)",
    "NDUFAF2": "NDUFAF2 (Complex I assembly factor)",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading Jacques et al. Supplementary Table S2 …")
xl = pd.read_excel(
    os.path.join(DATA, "Table_S2_Jacques_et_al_accepted.xlsx"),
    sheet_name="NALM6-log2-fold-change-values",
    usecols=["Gene symbol",
             "WT log2 fold-change",
             "CLPP sh  log2 fold-change"]
)
xl.columns = ["gene", "log2fc_WT", "log2fc_CLPPsh"]
xl = xl.dropna(subset=["gene"]).copy()
xl["gene"] = xl["gene"].astype(str).str.strip()

print(f"  {len(xl):,} proteins in dataset")

# ── Annotate targets of interest ──────────────────────────────────────────────
xl["target_label"] = xl["gene"].map(TARGETS_OF_INTEREST)

# ── Summary statistics reported in the manuscript ─────────────────────────────
# OXPHOS subunits (from MitoCarta Pathway = "OXPHOS subunits" or "Complex I/III/IV/V")
# We approximate by a curated list of common OXPHOS subunit symbols
OXPHOS_KEYWORDS = ["ATP5", "NDUF", "SDHA", "SDHB", "CYC", "COX", "UQCR", "MT-ND",
                   "MT-CO", "MT-ATP", "MT-CYB"]
is_oxphos = xl["gene"].apply(
    lambda g: any(g.startswith(k) for k in OXPHOS_KEYWORDS)
)
oxphos_sub = xl[is_oxphos]
print(f"\n  OXPHOS subunits detected: {len(oxphos_sub)}")
print(f"    Mean WT log2FC:    {oxphos_sub['log2fc_WT'].mean():.3f}")
print(f"    Mean CLPPsh log2FC:{oxphos_sub['log2fc_CLPPsh'].mean():.3f}")

# Target-specific values
print("\n  Key targets:")
for gene in TARGETS_OF_INTEREST:
    row = xl[xl["gene"] == gene]
    if len(row) > 0:
        wt  = row["log2fc_WT"].values[0]
        kd  = row["log2fc_CLPPsh"].values[0]
        print(f"    {gene}: WT={wt:.3f}, CLPPsh={kd:.3f}")
    else:
        print(f"    {gene}: not detected")

xl.to_csv(os.path.join(OUT, "jacques_log2fc_nalm6.csv"), index=False)
print(f"\nOutput: {OUT}/jacques_log2fc_nalm6.csv")
