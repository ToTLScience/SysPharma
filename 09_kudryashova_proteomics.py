"""
09_kudryashova_proteomics.py
============================
Reanalyse label-free quantification (LFQ) proteomics data from
Kudryashova et al. 2023 (Data MDPI, DOI:10.3390/data8070119) examining
GPX4 inhibition by ML210 in Pfa1 Gpx4-/- mouse embryonic fibroblasts.

The dataset provides MaxLFQ intensities for proteins at 24 h and 48 h
after ML210 treatment versus vehicle control.  Log2FC is computed as:
  log2FC = log2(treated_mean / control_mean)
with half-minimum imputation for missing values in conditions where
≥2 replicates are detected.

Mouse gene symbols are used as-is; a note is included that ortholog
mapping to human was performed by gene symbol matching.

These values are used in Fig. 4F.

Input  (data/)
------
- kudryashova_log2fc.csv   Pre-processed log2FC table
  columns: Gene, Protein ID, log2fc_24h, log2fc_48h, mean_A, mean_C, mean_B
  (A = control replicates; C = 24h ML210; B = 48h ML210 — as deposited)

Output (results/)
-------
- kudryashova_ml210_processed.csv
"""

import pandas as pd
import numpy as np
import os

DATA = "data"
OUT  = "results"

FERROPTOSIS_GENES = [
    "Gpx4", "Acsl4", "Hmox1", "Fth1", "Tfrc", "Cth",
    "Fsp1",  # AIFM2 in human
    "Dhodh", "Vdac2"
]

HUMAN_ORTHOLOGS = {  # mouse → human gene symbol
    "Gpx4":  "GPX4",
    "Acsl4": "ACSL4",
    "Hmox1": "HMOX1",
    "Fth1":  "FTH1",
    "Tfrc":  "TFRC",
    "Cth":   "CTH",
    "Fsp1":  "AIFM2",
    "Dhodh": "DHODH",
    "Vdac2": "VDAC2",
    "Akap1": "AKAP1",
    "Prkag1":"PRKAG1",
}

# ── Load pre-processed log2FC table ──────────────────────────────────────────
print("Loading Kudryashova et al. log2FC table …")
df = pd.read_csv(os.path.join(DATA, "kudryashova_log2fc.csv"))
df = df.dropna(subset=["Gene"]).copy()
df.columns = df.columns.str.strip()
print(f"  {len(df):,} proteins in dataset")

# Rename columns to match expected format
# Columns: Gene, Protein ID, log2fc_24h, log2fc_48h, mean_A, mean_C, mean_B
df = df.rename(columns={
    "Gene":       "mouse_gene",
    "Protein ID": "protein_id",
})

# ── Annotate ferroptosis genes ────────────────────────────────────────────────
df["is_ferroptosis"] = df["mouse_gene"].isin(FERROPTOSIS_GENES)
df["human_ortholog"] = df["mouse_gene"].map(HUMAN_ORTHOLOGS)

# Proteins with both timepoints quantified
df_both = df.dropna(subset=["log2fc_24h", "log2fc_48h"]).copy()
print(f"  {len(df_both):,} proteins with both 24h and 48h log2FC")

# ── Summary of key proteins ───────────────────────────────────────────────────
print("\n  Ferroptosis gene responses:")
ferr_sub = df_both[df_both["is_ferroptosis"]].copy()
for _, row in ferr_sub.iterrows():
    human = row["human_ortholog"] if pd.notna(row["human_ortholog"]) else ""
    print(f"    {row['mouse_gene']:<10} ({human:<8}) 24h={row['log2fc_24h']:.3f}  48h={row['log2fc_48h']:.3f}")

# Top upregulated at both timepoints
df_both["mean_fc"] = (df_both["log2fc_24h"] + df_both["log2fc_48h"]) / 2
top_up = df_both.nlargest(10, "mean_fc")[
    ["mouse_gene", "human_ortholog", "log2fc_24h", "log2fc_48h", "mean_fc"]
]
print(f"\n  Top 10 consistently upregulated proteins:")
print(top_up.to_string(index=False))

df_both.to_csv(os.path.join(OUT, "kudryashova_ml210_processed.csv"), index=False)
print(f"\nOutput: {OUT}/kudryashova_ml210_processed.csv")
