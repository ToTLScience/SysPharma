"""
01_prepare_dimensions.py
========================
Assemble the six raw scoring dimensions for every gene in the
drug-proteome atlas and the MitoCarta 3.0 proteome.

Inputs  (data/)
-------
- mitchell_log2fc.csv          : protein × compound log2FC matrix (Mitchell 2023)
                                 rows = proteins, columns = compounds, first col = gene
- crispr_hct116_full.csv       : DepMap 23Q4 Chronos scores + pan-essentiality
                                 columns: gene, chronos_hct116, pan_essentiality
- drugmap_gene_sites.csv       : DrugMap CRA site counts per gene
                                 columns: gene, n_cra_sites
- slcabpp_gene_summary.csv     : SLC-ABPP per-gene max competitive ratio
                                 columns: gene, n_sites, max_cr, n_engaged
- halflife_hct116.csv          : protein half-lives in HCT116 (Li 2021)
                                 columns: gene, uniprot, halflife_h, r2
- protein_zc_table.csv         : per-protein ZC values (from 02_compute_zc.py)
                                 columns: Gene, UniprotID, seq_length, ZC
- Human.MitoCarta3.0.xls       : MitoCarta 3.0 gene list and pathway annotations

Outputs (results/)
-------
- dimensions_all_proteome.csv  : one row per drug-proteome gene, raw dimension values
- dimensions_mitocarta.csv     : MitoCarta-restricted subset
"""

import numpy as np
import pandas as pd
import xlrd
import os

DATA  = "data"
OUT   = "results"
os.makedirs(OUT, exist_ok=True)

HCT116_DEPMAP_ID = "ACH-000971"
CRA_CAP  = 30       # L dimension cap (DrugMap CRA sites)
CR_CAP   = 20.0     # R dimension cap (SLC-ABPP max competitive ratio)
HL_THRESHOLD = 8.0  # hours — proteins below this are classified as labile (F=1)

# ── 1. Drug-proteome response (D) ────────────────────────────────────────────
print("Loading drug-proteome atlas …")
df_drug = pd.read_csv(os.path.join(DATA, "mitchell_log2fc.csv"), index_col=0)
# Rows = genes, columns = compounds
D = df_drug.abs().mean(axis=1).rename("D")
print(f"  {len(D):,} proteins × {df_drug.shape[1]:,} compounds")

# ── 2. Context-specific essentiality (E) ─────────────────────────────────────
print("Loading CRISPR data …")
crispr = pd.read_csv(
    os.path.join(DATA, "crispr_hct116_full.csv"),
    usecols=["gene", "chronos_hct116", "pan_essentiality"]
).set_index("gene")
crispr["E_raw"] = crispr["chronos_hct116"].abs() * (1 - crispr["pan_essentiality"]) ** 2

# ── 3. Cysteine ligandability (L) — DrugMap CRA sites ────────────────────────
print("Loading DrugMap …")
drugmap = (
    pd.read_csv(os.path.join(DATA, "drugmap_gene_sites.csv"))
    .set_index("gene")["n_cra_sites"]
    .rename("L_raw")
)
drugmap_capped = drugmap.clip(upper=CRA_CAP).rename("L")

# ── 4. Cysteine reactivity (R) — SLC-ABPP max CR ─────────────────────────────
print("Loading SLC-ABPP …")
slcabpp = (
    pd.read_csv(os.path.join(DATA, "slcabpp_gene_summary.csv"))
    .set_index("gene")["max_cr"]
    .rename("R_raw")
)
slcabpp_capped = slcabpp.clip(upper=CR_CAP).rename("R")

# ── 5. Protein lability (F) — half-life < 8 h ────────────────────────────────
print("Loading half-life data …")
hl = pd.read_csv(os.path.join(DATA, "halflife_hct116.csv"), usecols=["gene", "halflife_h"])
hl = hl.groupby("gene")["halflife_h"].min()  # take shortest if multiple isoforms
F  = (hl < HL_THRESHOLD).astype(float).rename("F")

# ── 6. ZC deviation — from 02_compute_zc.py output ──────────────────────────
print("Loading ZC table …")
zc = (
    pd.read_csv(os.path.join(DATA, "protein_zc_table.csv"))
    .rename(columns={"Gene": "gene"})
    .set_index("gene")["ZC"]
)

# ── 7. MitoCarta gene list ────────────────────────────────────────────────────
print("Loading MitoCarta 3.0 …")
wb  = xlrd.open_workbook(os.path.join(DATA, "Human.MitoCarta3.0.xls"))
ws  = wb.sheet_by_name("A Human MitoCarta3.0")
mc_headers = [ws.cell_value(0, c) for c in range(ws.ncols)]
mc_genes   = [ws.cell_value(r, mc_headers.index("Symbol")).strip()
              for r in range(1, ws.nrows)
              if ws.cell_value(r, mc_headers.index("Symbol"))]
mito_genes = set(mc_genes)
print(f"  {len(mito_genes):,} MitoCarta genes")

# Compute mitochondrial proteome mean ZC (restricted to MitoCarta × drug-proteome)
mito_zc_vals = zc[zc.index.isin(mito_genes) & zc.index.isin(D.index)]
MITO_ZC_MEAN = mito_zc_vals.mean()
print(f"  Mitochondrial proteome mean ZC = {MITO_ZC_MEAN:.6f}")

# ── 8. Assemble master table ──────────────────────────────────────────────────
all_genes = D.index  # anchored to drug-proteome atlas coverage

df = pd.DataFrame(index=all_genes)
df["D"]       = D
df["chronos"]  = crispr["chronos_hct116"].reindex(all_genes)
df["pe"]       = crispr["pan_essentiality"].reindex(all_genes)
df["E"]        = crispr["E_raw"].reindex(all_genes).fillna(0.0)
df["CRA"]      = drugmap_capped.reindex(all_genes).fillna(0.0)
df["CR"]       = slcabpp_capped.reindex(all_genes).fillna(0.0)
df["hl"]       = hl.reindex(all_genes)
df["F"]        = F.reindex(all_genes).fillna(0.0)
df["ZC"]       = zc.reindex(all_genes)
df["ZC_dev"]   = (df["ZC"] - MITO_ZC_MEAN).abs()
df["is_mito"]  = df.index.isin(mito_genes)

# Drop genes with no ZC (sequences not found) — handled in 03 / 04
df.to_csv(os.path.join(OUT, "dimensions_all_proteome.csv"))

# MitoCarta-restricted subset (n = 1,107 after exclusions)
df_mito = df[df["is_mito"] & df["D"].notna() & df["E"].notna()].copy()
df_mito.to_csv(os.path.join(OUT, "dimensions_mitocarta.csv"))

print(f"\nDone. All-proteome: {len(df):,} genes | MitoCarta: {len(df_mito):,} genes")
print(f"Outputs written to {OUT}/")
