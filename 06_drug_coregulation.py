"""
06_drug_coregulation.py
=======================
Compute pairwise Pearson correlations between the drug-proteome response
vectors (875 compounds) for the five primary nominated targets
(BCL-W, MCL1, GPX4, CLPX, LONP1) and test significance by permutation.

Method
------
For each pair of targets (i, j):
  r_obs = Pearson(log2FC_i, log2FC_j)  across 875 compounds

Permutation test (n = 10,000):
  Randomly shuffle compound order for all targets simultaneously.
  Compute mean |r| of all pairwise correlations under each shuffle.
  Empirical p-value = fraction of permutations where |mean_r| >= |obs_mean_r|.

Input  (data/)
------
- mitchell_log2fc.csv : protein × compound log2FC matrix (Mitchell 2023)
                        rows = gene symbols, columns = compound names

Output (results/)
-------
- drug_coregulation_pearson.csv   : pairwise r and p-values
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

DATA = "data"
OUT  = "results"

TARGETS = ["BCL2L2", "MCL1", "GPX4", "CLPX", "LONP1"]  # BCL2L2 = BCL-W
N_PERM  = 10_000
RNG     = np.random.default_rng(42)

# ── Load drug-proteome matrix ─────────────────────────────────────────────────
print("Loading drug-proteome atlas …")
df_drug = pd.read_csv(os.path.join(DATA, "mitchell_log2fc.csv"), index_col=0)
print(f"  {df_drug.shape[0]:,} proteins × {df_drug.shape[1]:,} compounds")

# Extract target vectors
missing = [t for t in TARGETS if t not in df_drug.index]
if missing:
    print(f"  WARNING: targets not found in atlas: {missing}")

target_df = df_drug.loc[[t for t in TARGETS if t in df_drug.index]].T
target_df = target_df.dropna()  # compounds where all targets are measured
print(f"  {len(target_df):,} compounds with complete data for all targets")

# ── Observed pairwise correlations ───────────────────────────────────────────
mat = target_df.values  # shape: (n_compounds, n_targets)
n_t = len(target_df.columns)
labels = list(target_df.columns)

records = []
for i in range(n_t):
    for j in range(i+1, n_t):
        r, _ = stats.pearsonr(mat[:, i], mat[:, j])
        records.append({"target_1": labels[i], "target_2": labels[j], "pearson_r": r})

obs_df = pd.DataFrame(records)
obs_mean_r = obs_df["pearson_r"].abs().mean()

# ── Permutation test ──────────────────────────────────────────────────────────
print(f"Running {N_PERM:,} permutations …")
perm_mean_rs = np.empty(N_PERM)
for k in range(N_PERM):
    perm_mat = mat.copy()
    idx = RNG.permutation(len(mat))
    perm_mat = perm_mat[idx]  # shuffle compound order
    rs = []
    for i in range(n_t):
        for j in range(i+1, n_t):
            r_p, _ = stats.pearsonr(perm_mat[:, i], perm_mat[:, j])
            rs.append(abs(r_p))
    perm_mean_rs[k] = np.mean(rs)

# Per-pair p-value: fraction of permutations with |r_perm| >= |r_obs|
perm_rs_by_pair = {}
for k in range(N_PERM):
    idx = RNG.permutation(len(mat))
    perm_mat = mat[idx]
    for ii, (i, j) in enumerate(
        (a, b) for a in range(n_t) for b in range(a+1, n_t)
    ):
        key = (labels[i], labels[j])
        r_p, _ = stats.pearsonr(perm_mat[:, i], perm_mat[:, j])
        perm_rs_by_pair.setdefault(key, []).append(abs(r_p))

pval_global = np.mean(np.abs(perm_mean_rs) >= np.abs(obs_mean_r))
print(f"  Global mean |r| = {obs_mean_r:.4f}, permutation p = {pval_global:.4f}")

for rec in records:
    key = (rec["target_1"], rec["target_2"])
    if key in perm_rs_by_pair:
        pvals_k = np.array(perm_rs_by_pair[key])
        rec["p_value"] = np.mean(pvals_k >= abs(rec["pearson_r"]))
    else:
        rec["p_value"] = np.nan
    rec["significant"] = rec["p_value"] < 0.05

result_df = pd.DataFrame(records)
result_df.to_csv(os.path.join(OUT, "drug_coregulation_pearson.csv"), index=False)
print(f"\nPairwise correlations:")
print(result_df.to_string(index=False))
print(f"\nOutput: {OUT}/drug_coregulation_pearson.csv")
