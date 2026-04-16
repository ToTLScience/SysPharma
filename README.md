# SysPharma — Systems Pharmacoproteomics of Mitochondrial Drug Targets in Colorectal Cancer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Analysis code and figure generation for:

> **A Pharmacoproteomic Scoring Framework for Mitochondrial Drug Target Prioritization in Colorectal Cancer**
>
> Tsz-Leung To · *Molecular & Cellular Proteomics* (2026)

## Pipeline
data/ files → analysis_v6.py → results/ CSVs → generate_figures.py → figures/ PDFs

## Overview

This repository implements a six-dimension composite priority scoring framework that integrates:

| Dimension | Weight | Source | Description |
|-----------|--------|--------|-------------|
| **E** — Essentiality | 0.30 | DepMap 23Q4 | \|Chronos\| × (1 − PE), linear pan-essentiality penalty |
| **D** — Drug response | 0.30 | Mitchell 2023 | Mean \|log₂FC\| across 875 compounds in HCT116 |
| **L** — Ligandability | 0.15 | DrugMap (Takahashi 2024) | Cysteine-reactive adduct (CRA) site count, capped at 30 |
| **R** — Reactivity | 0.15 | SLC-ABPP (Kuljanin 2021) | Max competitive ratio, capped at 20 |
| **F** — Lability | 0.05 | Li 2021 | Binary: 1 if measured half-life < 8 h, else 0 |
| **Z<sub>C</sub>** — Oxidation state | 0.05 | UniProt / LaRowe 2011 | \|Z<sub>C</sub> − compartment mean\|, MitoCarta-classified |

The framework scores **9,723 proteins** detected in the HCT116 drug-proteome atlas and ranks **50 MSigDB Hallmark gene sets** at the pathway level. Three therapeutic axes are nominated and characterized:

- **BH3 Apoptosis** — BCL-W (rank #151) and MCL1 (rank #903)
- **Mito-protease** — CLPX (#3,100), LONP1 (#1,023), CLPP (#9,340)
- **Ferroptosis** — GPX4 (#1,019)

## Repository structure

```
SysPharma/
├── README.md
├── requirements.txt
├── data/                          # Input data (see Data section below)
│   ├── crispr_hct116_full.csv
│   ├── gene_priority_all_proteome.csv
│   ├── drugmap_gene_sites.csv
│   ├── slcabpp_gene_summary.csv
│   ├── halflife_hct116.csv
│   ├── protein_zc_table.csv
│   ├── Human.MitoCarta3.0.xls
│   ├── h_all_v2026_1_Hs_symbols.gmt
│   ├── kudryashova_log2fc.csv
│   └── Table_S2_Jacques_et_al_accepted.xlsx
├── scripts/
│   ├── analysis_v6.py             # Master scoring script
│   ├── figures_main.py            # Main figures (Fig 1–5)
│   └── figures_supplementary.py   # Supplementary figures (Fig S1–S2)
└── results/                       # Generated outputs
    ├── gene_priority_v6_proteome.csv
    ├── hallmark_pathway_scores_v6.csv
    ├── apoptosis_hallmark_genes_v6.csv
    ├── table_S1_targets.csv
    └── figures/
```

## Quick start

```bash
# Clone the repository
git clone https://github.com/ToTLScience/SysPharma.git
cd SysPharma

# Install dependencies
pip install -r requirements.txt

# Run the scoring analysis
python scripts/analysis_v6.py --data-dir data/ --output-dir results/

# Generate figures
python scripts/figures_main.py --data-dir data/ --results-dir results/ --output-dir results/figures/
python scripts/figures_supplementary.py --data-dir data/ --results-dir results/ --output-dir results/figures/
```

## Input data

All input datasets are publicly available. Place them in the `data/` directory.

| File | Source | Accession / URL |
|------|--------|-----------------|
| `crispr_hct116_full.csv` | DepMap 23Q4 | [depmap.org/portal](https://depmap.org/portal/) |
| `gene_priority_all_proteome.csv` | Mitchell et al. 2023 | MassIVE [MSV000089282](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000089282) |
| `drugmap_gene_sites.csv` | Takahashi et al. 2024 | Supplementary Table S3 |
| `slcabpp_gene_summary.csv` | Kuljanin et al. 2021 | ProteomeXchange [PXD016553](http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD016553) |
| `halflife_hct116.csv` | Li et al. 2021 | ProteomeXchange [PXD024513](http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD024513) |
| `protein_zc_table.csv` | UniProt / LaRowe & Van Cappellen 2011 | Computed from [UniProtKB](https://www.uniprot.org/) |
| `Human.MitoCarta3.0.xls` | Rath et al. 2021 | [broadinstitute.org](https://personal.broadinstitute.org/scalvo/MitoCarta3.0/) |
| `h_all_v2026_1_Hs_symbols.gmt` | MSigDB Hallmark gene sets | [gsea-msigdb.org](https://www.gsea-msigdb.org/gsea/msigdb/human/collection/H) |
| `kudryashova_log2fc.csv` | Kudryashova et al. 2023 | PRIDE [PXD041327](https://www.ebi.ac.uk/pride/archive/projects/PXD041327) |
| `Table_S2_Jacques_et_al_accepted.xlsx` | Jacques et al. 2020 | figshare [11873841](https://doi.org/10.25386/genetics.11873841) |

## Scoring methodology

### Gene-level (n = 9,723)

Each protein detected in the drug-proteome atlas is scored:

```
S = 0.30 × E_norm + 0.30 × D_norm + 0.15 × L_norm + 0.15 × R_norm + 0.05 × F_norm + 0.05 × Zc_norm
```

All dimensions are min-max normalized to [0, 1] across the full scored set before weighting.

**Key design decisions:**

- **E uses a linear (1 − PE) penalty**, not quadratic. This preserves signal from moderately pan-essential targets like MCL1 (PE = 0.64) that are clinically relevant, while still suppressing housekeeping genes.
- **F is binary with a threshold of 8 h.** Proteins absent from the half-life dataset are treated as stable (F = 0), not as labile.
- **Z<sub>C</sub> deviation is compartment-specific.** Mitochondrial proteins (classified by MitoCarta 3.0) are referenced against the mitochondrial proteome mean (−0.151); non-mitochondrial proteins against the non-mitochondrial mean (−0.116).

### Pathway-level (n = 50 Hallmark gene sets)

| Dimension | Pathway-level statistic |
|-----------|------------------------|
| E | Mean of per-gene E across scorable members |
| D | Mean of per-gene D across scorable members |
| L | Fraction of scorable genes with ≥ 1 CRA site |
| R | Fraction of SLC-ABPP-detected genes with CR ≥ 4 |
| F | Fraction of scorable genes with half-life < 8 h |
| Z<sub>C</sub> | Mean of per-gene Z<sub>C</sub> deviation |

Pathway dimensions are min-max normalized across the 50 gene sets before applying the same weighting formula.

### Sensitivity analysis

Robustness of pathway rankings to weight choice is assessed by sampling 50,000 weight vectors from a Dirichlet distribution (concentration = 30) centered on the baseline weights.

## Key results

**Hallmark pathway ranking (top 10 of 50):**

| Rank | Pathway | Score |
|------|---------|-------|
| 1 | Cholesterol Homeostasis | 0.597 |
| 2 | Coagulation | 0.550 |
| 3 | Myogenesis | 0.540 |
| 4 | TGF-β Signaling | 0.534 |
| 5 | G2M Checkpoint | 0.518 |
| 6 | TNF-α Signaling via NF-κB | 0.511 |
| 7 | Notch Signaling | 0.507 |
| 8 | Androgen Response | 0.504 |
| 9 | **Apoptosis** | **0.492** |
| 10 | Epithelial Mesenchymal Transition | 0.491 |

**Six nominated targets:**

| Gene | Target | Score | Rank (of 9,723) |
|------|--------|-------|-----------------|
| BCL2L2 | BCL-W | 0.203 | #151 |
| MCL1 | MCL1 | 0.119 | #903 |
| GPX4 | GPX4 | 0.114 | #1,019 |
| LONP1 | LONP1 | 0.114 | #1,023 |
| CLPX | CLPX | 0.070 | #3,100 |
| CLPP | CLPP | 0.023 | #9,340 |

## Citation

```bibtex
@article{To2026SysPharma,
  author  = {To, Tsz-Leung},
  title   = {A Pharmacoproteomic Scoring Framework for Mitochondrial Drug
             Target Prioritization in Colorectal Cancer},
  journal = {Molecular \& Cellular Proteomics},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

Tsz-Leung To — [tto@alum.mit.edu](mailto:tto@alum.mit.edu)
