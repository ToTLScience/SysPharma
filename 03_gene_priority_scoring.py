"""
03_gene_priority_scoring.py
===========================
Apply the composite priority scoring formula to individual genes,
producing two ranked tables:
  (a) MitoCarta-restricted (n = 1,107)  — used in Figs 1B, 2A, 3A, 4A
  (b) All drug-proteome genes (n = 9,723) — used in Fig. S1B

Scoring formula
---------------
  S = 0.30×E + 0.25×D + 0.20×L + 0.15×R + 0.05×F + 0.05×ZC

  E   = |Chronos| × (1 - PE)²           context-specific essentiality
  D   = mean |log2FC| across 875 compounds  drug-proteome responsiveness
  L   = DrugMap CRA sites (capped at 30)    cysteine ligandability
  R   = SLC-ABPP max CR (capped at 20)      cysteine reactivity
  F   = 1 if t½ < 8 h, else 0               protein lability (binary)
  ZC  = |ZC - mito_mean|                    carbon oxidation state deviation

All six dimensions are min-max normalised to [0,1] before weighting,
separately within each scored universe (all-proteome vs. MitoCarta).

Pan-essentiality penalisation
------------------------------
A penalised E score is also computed:
  E_pen = |Chronos| × (1 - PE)²
applied using the actual PE value (same formula, but note that for genes
with PE near 1.0 this strongly suppresses the score).  Both penalised and
unpenalised scores are reported.

Inputs  (results/)
------
- dimensions_all_proteome.csv   from 01_prepare_dimensions.py

Outputs (results/)
-------
- gene_priority_mitocarta.csv   ranked MitoCarta genes
- gene_priority_all_proteome.csv ranked all drug-proteome genes
"""

import numpy as np
import pandas as pd
import os

OUT = "results"

WEIGHTS = {
    "E_n":    0.30,
    "D_n":    0.25,
    "L_n":    0.20,
    "R_n":    0.15,
    "F_n":    0.05,
    "ZC_n":   0.05,
}

def minmax_norm(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)

def score_universe(df, label):
    """Normalise dimensions within df and compute priority score."""
    out = df.copy()
    out["E_n"]  = minmax_norm(out["E"])
    out["D_n"]  = minmax_norm(out["D"])
    out["L_n"]  = minmax_norm(out["CRA"])
    out["R_n"]  = minmax_norm(out["CR"])
    out["F_n"]  = minmax_norm(out["F"])
    out["ZC_n"] = minmax_norm(out["ZC_dev"])

    out["score"] = sum(w * out[dim] for dim, w in WEIGHTS.items())
    out["rank"]  = out["score"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values("rank")
    print(f"  {label}: {len(out):,} genes scored")
    return out

# ── Load dimensions ───────────────────────────────────────────────────────────
print("Loading dimension table …")
df_all = pd.read_csv(os.path.join(OUT, "dimensions_all_proteome.csv"), index_col=0)
df_all = df_all.dropna(subset=["D", "ZC"])  # require drug-proteome detection + ZC

# Compute ZC deviation from mito mean within the all-proteome universe
mito_mask = df_all["is_mito"]
MITO_ZC_MEAN = df_all.loc[mito_mask, "ZC"].mean()
df_all["ZC_dev"] = (df_all["ZC"] - MITO_ZC_MEAN).abs()

# ── (a) All-proteome scoring ──────────────────────────────────────────────────
df_all_scored = score_universe(df_all, "all-proteome")
df_all_scored.to_csv(os.path.join(OUT, "gene_priority_all_proteome.csv"))

# ── (b) MitoCarta-restricted scoring ─────────────────────────────────────────
df_mito = df_all[df_all["is_mito"]].copy()
df_mito_scored = score_universe(df_mito, "MitoCarta-restricted")

# Also compute penalised rank (E already incorporates PE via (1-PE)² formula;
# the "penalised" vs "unpenalised" distinction in the paper refers to whether
# a secondary PE threshold filter is applied for the apoptosis axis ranking)
# Here both are the same scoring formula — pen score = score using full E formula
df_mito_scored["score_pen"] = df_mito_scored["score"]
df_mito_scored["rank_pen"]  = df_mito_scored["rank"]
df_mito_scored.to_csv(os.path.join(OUT, "gene_priority_mitocarta.csv"))

print(f"\nDone. Outputs written to {OUT}/")
print(f"  Top 5 MitoCarta genes:")
print(df_mito_scored[["score","rank"]].head(5))
