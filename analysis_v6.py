#!/usr/bin/env python3
"""
Systems Pharmacoproteomics of Mitochondrial Drug Targets in Colorectal Cancer
Master analysis script (v6)

Author: Tsz-Leung To (tto@alum.mit.edu)
Repository: https://github.com/ToTLScience/SysPharma

This script reproduces all gene-level and pathway-level priority scores,
data tables, and figures reported in the manuscript.

Requirements:
  pip install pandas numpy scipy matplotlib seaborn xlrd openpyxl

Input data files (place in data/ directory):
  - crispr_hct116_full.csv          DepMap 23Q4 CRISPR fitness (HCT116)
  - gene_priority_all_proteome.csv  Drug-proteome D values (Mitchell 2023)
  - drugmap_gene_sites.csv          DrugMap CRA site counts (Takahashi 2024)
  - slcabpp_gene_summary.csv        SLC-ABPP max CR (Kuljanin 2021)
  - halflife_hct116.csv             Protein half-lives (Li 2021)
  - protein_zc_table.csv            Carbon oxidation state (UniProt/LaRowe)
  - Human.MitoCarta3.0.xls          MitoCarta 3.0 (Rath 2021)
  - h_all_v2026_1_Hs_symbols.gmt   MSigDB Hallmark gene sets
  - kudryashova_log2fc.csv          ML210 proteomics (Kudryashova 2023)
  - Table_S2_Jacques_et_al_accepted.xlsx  ONC212 proteomics (Jacques 2020)

Usage:
  python analysis_v6.py --data-dir data/ --output-dir results/
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

# ============================================================
# CONFIGURATION
# ============================================================

# Scoring weights (locked)
WEIGHTS = {"E": 0.30, "D": 0.30, "L": 0.15, "R": 0.15, "F": 0.05, "Zc": 0.05}

# Caps
CRA_CAP = 30
CR_CAP = 20
HALFLIFE_THRESHOLD = 8.0  # hours

# Sensitivity analysis
N_DIRICHLET = 50000
DIRICHLET_CONCENTRATION = 30

# Six nominated targets
TARGETS = ["BCL2L2", "MCL1", "GPX4", "CLPX", "LONP1", "CLPP"]
TARGET_LABELS = {"BCL2L2": "BCL-W", "MCL1": "MCL1", "GPX4": "GPX4",
                 "CLPX": "CLPX", "LONP1": "LONP1", "CLPP": "CLPP"}


def load_data(data_dir):
    """Load all input datasets."""
    print("Loading data...")
    
    # MitoCarta 3.0
    mc = pd.read_excel(os.path.join(data_dir, "Human.MitoCarta3.0.xls"),
                       sheet_name="A Human MitoCarta3.0")
    mito_genes = set(mc["Symbol"].dropna().unique())
    
    # CRISPR fitness
    crispr = pd.read_csv(os.path.join(data_dir, "crispr_hct116_full.csv"))
    
    # Drug-proteome D values
    old = pd.read_csv(os.path.join(data_dir, "gene_priority_all_proteome.csv"))
    d_vals = old[["gene", "D"]].copy()
    
    # DrugMap CRA sites
    drugmap = pd.read_csv(os.path.join(data_dir, "drugmap_gene_sites.csv"))
    
    # SLC-ABPP max CR
    slcabpp = pd.read_csv(os.path.join(data_dir, "slcabpp_gene_summary.csv"))
    slcabpp = slcabpp[["gene", "max_cr"]].rename(columns={"max_cr": "CR"})
    
    # Protein half-lives (deduplicate: take minimum per gene)
    hl = pd.read_csv(os.path.join(data_dir, "halflife_hct116.csv"))
    hl_dedup = hl.groupby("gene")["halflife_h"].min().reset_index()
    
    # Carbon oxidation state
    zc = pd.read_csv(os.path.join(data_dir, "protein_zc_table.csv"))
    zc = zc.rename(columns={"Gene": "gene"})
    zc_dedup = zc.drop_duplicates(subset="gene", keep="first")
    
    # Hallmark gene sets
    hallmarks = {}
    gmt_path = os.path.join(data_dir, "h_all_v2026_1_Hs_symbols.gmt")
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            name = parts[0].replace("HALLMARK_", "").replace("_", " ").title()
            hallmarks[name] = parts[2:]
    
    return {
        "mito_genes": mito_genes,
        "crispr": crispr,
        "d_vals": d_vals,
        "drugmap": drugmap,
        "slcabpp": slcabpp,
        "hl_dedup": hl_dedup,
        "zc_dedup": zc_dedup,
        "hallmarks": hallmarks,
        "hl_raw": hl,
    }


def compute_gene_scores(data):
    """Compute proteome-wide gene-level priority scores."""
    print("\n=== Gene-level scoring (n = 9,723) ===")
    
    # Build master table from drug-proteome genes
    genes = data["d_vals"][["gene", "D"]].copy()
    
    # Merge dimensions
    genes = genes.merge(data["crispr"][["gene", "chronos_hct116", "pan_essentiality"]],
                        on="gene", how="left")
    genes = genes.merge(data["drugmap"].rename(columns={"n_cra_sites": "CRA"}),
                        on="gene", how="left")
    genes["CRA"] = genes["CRA"].fillna(0)
    genes = genes.merge(data["slcabpp"], on="gene", how="left")
    genes["CR"] = genes["CR"].fillna(0)
    genes = genes.merge(data["hl_dedup"], on="gene", how="left")
    genes = genes.merge(data["zc_dedup"][["gene", "ZC"]], on="gene", how="left")
    genes["is_mito"] = genes["gene"].isin(data["mito_genes"])
    
    print(f"  Total genes: {len(genes)}, mitochondrial: {genes['is_mito'].sum()}")
    
    # E = |Chronos| × (1 − PE) [linear penalty]
    genes["E"] = genes["chronos_hct116"].abs() * (1 - genes["pan_essentiality"])
    genes["E"] = genes["E"].fillna(0)
    
    # L = CRA capped
    genes["L"] = genes["CRA"].clip(upper=CRA_CAP)
    
    # R = CR capped
    genes["R"] = genes["CR"].clip(upper=CR_CAP)
    
    # F = 1 if measured half-life < threshold, else 0
    genes["F"] = 0.0
    genes.loc[genes["halflife_h"].notna() & (genes["halflife_h"] < HALFLIFE_THRESHOLD), "F"] = 1.0
    
    # Zc_dev = compartment-specific absolute deviation
    mito_zc = genes.loc[genes["is_mito"] & genes["ZC"].notna(), "ZC"]
    nonmito_zc = genes.loc[~genes["is_mito"] & genes["ZC"].notna(), "ZC"]
    mito_mean = mito_zc.mean()
    nonmito_mean = nonmito_zc.mean()
    print(f"  Mito Zc mean: {mito_mean:.4f} (n={len(mito_zc)})")
    print(f"  Non-mito Zc mean: {nonmito_mean:.4f} (n={len(nonmito_zc)})")
    
    genes["Zc_dev"] = np.nan
    genes.loc[genes["is_mito"], "Zc_dev"] = (genes.loc[genes["is_mito"], "ZC"] - mito_mean).abs()
    genes.loc[~genes["is_mito"], "Zc_dev"] = (genes.loc[~genes["is_mito"], "ZC"] - nonmito_mean).abs()
    
    # Min-max normalize
    dims = {"E": "E", "D": "D", "L": "L", "R": "R", "F": "F", "Zc": "Zc_dev"}
    for dim_name, col in dims.items():
        vmin, vmax = genes[col].min(), genes[col].max()
        genes[f"{dim_name}_n"] = (genes[col] - vmin) / (vmax - vmin) if vmax > vmin else 0
        genes[f"{dim_name}_n"] = genes[f"{dim_name}_n"].fillna(0)
    
    # Composite score
    genes["score"] = sum(WEIGHTS[d] * genes[f"{d}_n"] for d in WEIGHTS)
    genes["rank"] = genes["score"].rank(ascending=False, method="min").astype(int)
    
    # Print targets
    print("\n  Six targets:")
    t = genes[genes["gene"].isin(TARGETS)].sort_values("rank")
    for _, r in t.iterrows():
        label = TARGET_LABELS.get(r["gene"], r["gene"])
        print(f"    {label:8s}  score={r['score']:.3f}  rank={int(r['rank']):5d}")
    
    return genes


def compute_pathway_scores(genes, hallmarks):
    """Compute Hallmark pathway-level priority scores."""
    print("\n=== Pathway-level scoring (50 Hallmark gene sets) ===")
    
    scored_genes = set(genes["gene"])
    rows = []
    
    for pw_name, pw_genes in hallmarks.items():
        scorable = [g for g in pw_genes if g in scored_genes]
        n = len(scorable)
        if n < 3:
            continue
        
        pw = genes[genes["gene"].isin(scorable)]
        
        pw_E = pw["E"].mean()
        pw_D = pw["D"].mean()
        pw_L = (pw["CRA"] >= 1).sum() / n
        
        detected_slc = pw[pw["CR"] > 0]
        pw_R = (detected_slc["CR"] >= 4).sum() / len(detected_slc) if len(detected_slc) > 0 else 0.0
        pw_F = pw["F"].sum() / n
        pw_Zc = pw["Zc_dev"].mean() if pw["Zc_dev"].notna().any() else 0.0
        
        rows.append({
            "pathway": pw_name, "n_genes": n,
            "E_raw": pw_E, "D_raw": pw_D, "L_raw": pw_L,
            "R_raw": pw_R, "F_raw": pw_F, "Zc_raw": pw_Zc
        })
    
    pw_df = pd.DataFrame(rows)
    
    # Normalize and score
    for dim, col in [("E","E_raw"),("D","D_raw"),("L","L_raw"),("R","R_raw"),("F","F_raw"),("Zc","Zc_raw")]:
        vmin, vmax = pw_df[col].min(), pw_df[col].max()
        pw_df[f"{dim}_n"] = (pw_df[col] - vmin) / (vmax - vmin) if vmax > vmin else 0
    
    pw_df["score"] = sum(WEIGHTS[d] * pw_df[f"{d}_n"] for d in WEIGHTS)
    pw_df["rank"] = pw_df["score"].rank(ascending=False, method="min").astype(int)
    pw_df = pw_df.sort_values("rank")
    
    print(f"  Scored {len(pw_df)} pathways")
    for _, r in pw_df.head(10).iterrows():
        print(f"    {int(r['rank']):3d}  {r['pathway']:40s}  score={r['score']:.3f}")
    
    return pw_df


def sensitivity_analysis(pw_df):
    """Dirichlet Monte Carlo sensitivity analysis of pathway rankings."""
    print(f"\n=== Sensitivity analysis ({N_DIRICHLET:,d} iterations) ===")
    
    w0 = np.array([WEIGHTS[d] for d in ["E","D","L","R","F","Zc"]])
    alpha = w0 * DIRICHLET_CONCENTRATION
    
    raw_cols = ["E_raw","D_raw","L_raw","R_raw","F_raw","Zc_raw"]
    raw_vals = pw_df[raw_cols].values
    
    n_pw = len(pw_df)
    rank_history = np.zeros((N_DIRICHLET, n_pw), dtype=int)
    
    np.random.seed(42)
    for i in range(N_DIRICHLET):
        w = np.random.dirichlet(alpha)
        norm = np.zeros_like(raw_vals)
        for d in range(6):
            vmin, vmax = raw_vals[:, d].min(), raw_vals[:, d].max()
            if vmax > vmin:
                norm[:, d] = (raw_vals[:, d] - vmin) / (vmax - vmin)
        scores = norm @ w
        rank_history[i] = (-scores).argsort().argsort() + 1
    
    # Report Apoptosis stability
    pw_reset = pw_df.reset_index(drop=True)
    apop_pos = pw_reset[pw_reset["pathway"] == "Apoptosis"].index[0]
    apop_ranks = rank_history[:, apop_pos]
    
    print(f"  Apoptosis baseline rank: {pw_reset.loc[apop_pos, 'rank']}")
    print(f"  Median rank: {np.median(apop_ranks):.0f}")
    print(f"  P(rank ≤ 5): {(apop_ranks <= 5).mean()*100:.1f}%")
    print(f"  5th–95th percentile: {np.percentile(apop_ranks, 5):.0f}–{np.percentile(apop_ranks, 95):.0f}")
    
    return rank_history


def main():
    parser = argparse.ArgumentParser(description="v6 priority scoring analysis")
    parser.add_argument("--data-dir", default="data/", help="Input data directory")
    parser.add_argument("--output-dir", default="results/", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load
    data = load_data(args.data_dir)
    
    # Gene scores
    genes = compute_gene_scores(data)
    genes.to_csv(os.path.join(args.output_dir, "gene_priority_v6_proteome.csv"), index=False)
    
    # Pathway scores
    pw = compute_pathway_scores(genes, data["hallmarks"])
    pw.to_csv(os.path.join(args.output_dir, "hallmark_pathway_scores_v6.csv"), index=False)
    
    # Apoptosis Hallmark gene ranking
    apop_genes = data["hallmarks"].get("Apoptosis", [])
    apop = genes[genes["gene"].isin(apop_genes)].sort_values("score", ascending=False)
    apop["apop_rank"] = range(1, len(apop) + 1)
    apop.to_csv(os.path.join(args.output_dir, "apoptosis_hallmark_genes_v6.csv"), index=False)
    
    # Sensitivity analysis
    rank_history = sensitivity_analysis(pw)
    
    # Supplementary Table S1
    supp = genes[genes["gene"].isin(TARGETS)].sort_values("rank")
    supp_out = supp[["gene","score","rank","chronos_hct116","pan_essentiality",
                     "E","D","CRA","CR","halflife_h","F","ZC","Zc_dev"]].copy()
    supp_out.to_csv(os.path.join(args.output_dir, "table_S1_targets.csv"), index=False)
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
