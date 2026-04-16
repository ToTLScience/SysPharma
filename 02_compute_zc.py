"""
02_compute_zc.py
================
Compute per-protein carbon oxidation state (ZC) from UniProt canonical
FASTA sequences following the formalism of LaRowe & Van Cappellen (2011)
as applied to proteomes in Flamholz et al. (2025).

    ZC = Σ(nᵢ × oxᵢ) / nC

where oxᵢ is the formal oxidation state of the i-th carbon atom and nC
is the total carbon count of the protein sequence.

Oxidation states are derived from each amino acid's elemental formula:
formal C oxidation state = (2×nO + nN - nH + charge×nS_adj - ...) / nC
using the standard CHNOS atomic composition and charge balance rule.

Reference:
  LaRowe DE, Van Cappellen P. Geochim Cosmochim Acta. 2011;75(6):1520-1543.
  DOI:10.1016/j.gca.2011.01.020

Input  (data/)
------
- uniprot_sequences.fasta : UniProt canonical FASTA (downloaded via UniProtKB API)

Output (results/ and data/)
-------
- protein_zc_table.csv : columns Gene, UniprotID, seq_length, ZC
  (also written to data/ so 01_prepare_dimensions.py can read it)
"""

import os
import csv
import re
from collections import Counter

DATA = "data"
OUT  = "results"
os.makedirs(OUT, exist_ok=True)

# Amino acid elemental compositions (C, H, N, O, S counts as residue, i.e. minus water)
# Source: standard biochemistry tables; residue = amino acid - H2O
AA_FORMULA = {
    #        C   H   N   O   S
    "A": (   3,  5,  1,  1,  0),
    "R": (   6,  12, 4,  1,  0),
    "N": (   4,  6,  2,  2,  0),
    "D": (   4,  5,  1,  3,  0),
    "C": (   3,  5,  1,  1,  1),
    "E": (   5,  7,  1,  3,  0),
    "Q": (   5,  8,  2,  2,  0),
    "G": (   2,  3,  1,  1,  0),
    "H": (   6,  7,  3,  1,  0),
    "I": (   6,  11, 1,  1,  0),
    "L": (   6,  11, 1,  1,  0),
    "K": (   6,  12, 2,  1,  0),
    "M": (   5,  9,  1,  1,  1),
    "F": (   9,  9,  1,  1,  0),
    "P": (   5,  7,  1,  1,  0),
    "S": (   3,  5,  1,  2,  0),
    "T": (   4,  7,  1,  2,  0),
    "W": (  11,  10, 2,  1,  0),
    "Y": (   9,  9,  1,  2,  0),
    "V": (   5,  9,  1,  1,  0),
}

# ZC per residue = (2*nO + nN - nH) / nC  [LaRowe & Van Cappellen 2011, Eq. 4]
# (sulfur treated as neutral; charge on terminal groups cancels across long chains)
def zc_residue(aa):
    if aa not in AA_FORMULA:
        return None, 0
    C, H, N, O, S = AA_FORMULA[aa]
    if C == 0:
        return None, 0
    # Formal ZC: charge balance requires 4*C + H - 2*O - 3*N - 2*S = 0 for neutral
    # Solving: ZC = (2*O + 3*N + 2*S - H) / C  — but LaRowe uses simpler H-only version
    # Full formula including N and O contributions:
    zc = (2 * O + 3 * N + 2 * S - H) / C
    return zc, C


def protein_zc(sequence):
    """Compute ZC for a protein sequence string."""
    total_zc_nC = 0.0
    total_nC    = 0
    for aa in sequence.upper():
        zc, nC = zc_residue(aa)
        if zc is not None:
            total_zc_nC += zc * nC
            total_nC    += nC
    if total_nC == 0:
        return None
    return total_zc_nC / total_nC


def parse_fasta(fasta_path):
    """Parse UniProt FASTA; yield (uniprot_id, gene_name, sequence)."""
    uniprot_re = re.compile(r">[a-z]{2}\|([A-Z0-9]+)\|(\S+)")
    gene_re    = re.compile(r"GN=(\S+)")
    current_id, current_gene, current_seq = None, None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id:
                    yield current_id, current_gene, "".join(current_seq)
                m_id   = uniprot_re.match(line)
                m_gene = gene_re.search(line)
                current_id   = m_id.group(1)   if m_id   else line[1:].split()[0]
                current_gene = m_gene.group(1)  if m_gene else current_id
                current_seq  = []
            else:
                current_seq.append(line)
    if current_id:
        yield current_id, current_gene, "".join(current_seq)


# ── Main ──────────────────────────────────────────────────────────────────────
fasta_path = os.path.join(DATA, "uniprot_sequences.fasta")
print(f"Reading sequences from {fasta_path} …")

rows = []
seen_genes = {}  # keep longest sequence per gene symbol
for uid, gene, seq in parse_fasta(fasta_path):
    zc_val = protein_zc(seq)
    if zc_val is None:
        continue
    if gene not in seen_genes or len(seq) > seen_genes[gene][2]:
        seen_genes[gene] = (uid, gene, len(seq), zc_val)

rows = list(seen_genes.values())
print(f"  Computed ZC for {len(rows):,} unique gene symbols")

out_path = os.path.join(OUT, "protein_zc_table.csv")
with open(out_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Gene", "UniprotID", "seq_length", "ZC"])
    for uid, gene, length, zc_val in sorted(rows, key=lambda x: x[1]):
        w.writerow([gene, uid, length, f"{zc_val:.8f}"])

# Also copy to data/ for use by 01_prepare_dimensions.py
import shutil
shutil.copy(out_path, os.path.join(DATA, "protein_zc_table.csv"))

print(f"ZC table written to {out_path}")
