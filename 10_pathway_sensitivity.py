"""
10_pathway_sensitivity.py
=========================
Dirichlet-based stochastic sensitivity analysis of pathway-level
priority score weights, following Saint-Hilary et al. (2017).

Method
------
50,000 weight vectors are sampled from a Dirichlet distribution
centred on the baseline weights (α = w° × 30), where w° is the
baseline weight vector (0.30, 0.25, 0.20, 0.15, 0.05, 0.05).
The concentration parameter of 30 yields moderate dispersion
(coefficient of variation 27–79% across dimensions).

For each sampled weight vector, pathway-level composite scores
and ranks are recomputed.  Rank stability is summarised as the
probability of each pathway maintaining its baseline rank within
±k positions.

Reference:
  Saint-Hilary G, Cadour S, Robert V, Gasparini M.
  Biom J. 2017;59(3):567–578. DOI:10.1002/bimj.201600113

Inputs  (results/)
------
- pathway_priority_final.csv   from 04_pathway_priority_scoring.py

Outputs (results/)
-------
- pathway_sensitivity_mc.csv         Monte Carlo rank summary per pathway
- pathway_sensitivity_apoptosis.csv  Apoptosis rank distribution
"""

import numpy as np
import pandas as pd
import os

OUT = "results"

# Scoring weights (same as 03/04)
BASELINE_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.05, 0.05])
DIM_NAMES  = ["E", "D", "L", "R", "F", "Zc"]
NORM_COLS  = [
    "E_specificity_norm", "D_norm", "L_norm",
    "R_norm", "F_norm", "ZC_deviation_norm",
]

# Dirichlet parameters
CONCENTRATION = 30
N_MC          = 50_000
SEED          = 42

# ── Load pathway data ─────────────────────────────────────────────────────────
print("Loading pathway priority data …")
pw_df = pd.read_csv(os.path.join(OUT, "pathway_priority_final.csv"))
pathways = pw_df["pathway"].tolist()
n = len(pathways)
print(f"  {n} pathways loaded")

# Build normalised dimension matrix (pathways × 6)
M = pw_df[NORM_COLS].values

def score_and_rank(weights):
    """Compute scores and rank array for a given weight vector."""
    scores = M @ weights
    rank_arr = np.empty(n, dtype=int)
    for pos, idx in enumerate(np.argsort(-scores)):
        rank_arr[idx] = pos + 1
    return scores, rank_arr

# Baseline
baseline_scores, baseline_ranks = score_and_rank(BASELINE_WEIGHTS)
apop_idx = pathways.index("Apoptosis")
print(f"  Apoptosis baseline: rank {baseline_ranks[apop_idx]}, "
      f"score {baseline_scores[apop_idx]:.4f}")

# ── Monte Carlo sampling ──────────────────────────────────────────────────────
print(f"Sampling {N_MC:,} Dirichlet weight vectors "
      f"(α = baseline × {CONCENTRATION}) …")
rng = np.random.default_rng(SEED)
alpha = BASELINE_WEIGHTS * CONCENTRATION
mc_weights = rng.dirichlet(alpha, size=N_MC)

mc_all_ranks = np.zeros((N_MC, n), dtype=int)
mc_apop_ranks = np.zeros(N_MC, dtype=int)

for i in range(N_MC):
    _, ranks = score_and_rank(mc_weights[i])
    mc_all_ranks[i] = ranks
    mc_apop_ranks[i] = ranks[apop_idx]

# ── Summarise rank stability per pathway ──────────────────────────────────────
records = []
for p_idx in range(n):
    bl = baseline_ranks[p_idx]
    rd = mc_all_ranks[:, p_idx]
    records.append({
        "pathway":       pathways[p_idx],
        "baseline_rank": int(bl),
        "median_rank":   float(np.median(rd)),
        "mean_rank":     float(np.mean(rd)),
        "rank_5pct":     int(np.percentile(rd, 5)),
        "rank_95pct":    int(np.percentile(rd, 95)),
        "prob_within_1": float(np.mean(np.abs(rd - bl) <= 1)),
        "prob_within_3": float(np.mean(np.abs(rd - bl) <= 3)),
    })

summary_df = pd.DataFrame(records).sort_values("baseline_rank")
summary_df.to_csv(os.path.join(OUT, "pathway_sensitivity_mc.csv"), index=False)

# ── Apoptosis-specific summary ────────────────────────────────────────────────
pct_top3 = np.mean(mc_apop_ranks <= 3) * 100
pct_top5 = np.mean(mc_apop_ranks <= 5) * 100

apop_records = []
for rank_val in sorted(set(mc_apop_ranks)):
    apop_records.append({
        "rank":      int(rank_val),
        "count":     int(np.sum(mc_apop_ranks == rank_val)),
        "frequency": float(np.sum(mc_apop_ranks == rank_val) / N_MC),
    })

apop_df = pd.DataFrame(apop_records)
apop_df.to_csv(os.path.join(OUT, "pathway_sensitivity_apoptosis.csv"), index=False)

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\nApoptosis sensitivity results:")
print(f"  Median rank: {np.median(mc_apop_ranks):.0f}")
print(f"  Mean rank:   {np.mean(mc_apop_ranks):.1f}")
print(f"  5th–95th percentile: {np.percentile(mc_apop_ranks, 5):.0f}"
      f"–{np.percentile(mc_apop_ranks, 95):.0f}")
print(f"  P(rank ≤ 3): {pct_top3:.1f}%")
print(f"  P(rank ≤ 5): {pct_top5:.1f}%")
print(f"  Worst rank:  {mc_apop_ranks.max()}")

print(f"\nTop-5 pathway rank stability:")
print(summary_df[["pathway", "baseline_rank", "median_rank",
                   "rank_5pct", "rank_95pct"]].head(5).to_string(index=False))

print(f"\nOutputs written to {OUT}/")
