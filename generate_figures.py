#!/usr/bin/env python3
"""
generate_figures.py — Generate all manuscript figures (Fig 1–5, Fig S1–S2).

Systems Pharmacoproteomics of Drug Targets in Colorectal Cancer
Author: Tsz-Leung To (tto@alum.mit.edu)
Repository: https://github.com/ToTLScience/SysPharma

Usage:
    python generate_figures.py \\
        --data-dir data/ \\
        --results-dir results/ \\
        --output-dir results/figures/

Requires: pandas, numpy, scipy, matplotlib, xlrd, openpyxl
Font: Liberation Sans (metric-equivalent to Arial)
Style: panel headings 12pt bold; all other text 10–12pt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.stats import gaussian_kde

# =====================================================================
# SHARED STYLE
# =====================================================================
FONT = 'Liberation Sans'

def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT, 'Arial', 'DejaVu Sans'],
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,  'ytick.major.width': 0.6,
        'xtick.major.size': 3.5,   'ytick.major.size': 3.5,
        'xtick.labelsize': 10,     'ytick.labelsize': 10,
        'axes.labelsize': 11,
    })

def panel_label(ax, letter, title):
    """12pt bold panel heading."""
    ax.set_title(f'{letter}   {title}', fontsize=12, fontweight='bold',
                 fontfamily=FONT, loc='left', pad=8)

def cleanup_spines(axes_array):
    """Remove top and right spines from all axes."""
    for row in (axes_array if axes_array.ndim == 2 else [axes_array]):
        for a in (row if hasattr(row, '__iter__') else [row]):
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)

# ── Color palette ─────────────────────────────────────────────────────
RED       = '#CC3333'
DKRED     = '#8B0000'
BLUE      = '#1565C0'
PURPLE    = '#7B1FA2'
LTBLUE    = '#64B5F6'
DK_GREEN  = '#2E7D32'
LT_GREEN  = '#81C784'
BAR_GREEN = '#558B2F'
BAR_UP    = '#B39DDB'
ORANGE    = '#E8881D'
RED_UP    = '#C62828'
GRAY_DOT  = '#CCCCCC'
GRAY_LINE = '#999999'
BAR_GRAY  = '#BDBDBD'
BAR_RED   = '#CC3333'
BAR_ORANGE= '#E8881D'

# Schema colors
SRC_COLORS = ['#2B4C6F', '#34587F', '#3D648E', '#476F9E', '#507BAD', '#5986BD']
DIM_BG     = '#E6ECF2'
DIM_BORDER = '#6A8EAF'
SCORE_BG   = '#C4C4C4'
SCORE_TEXT  = '#333333'
ARROW_CLR  = '#8A8A8A'


# =====================================================================
# FIGURE 1 — Scoring framework + Hallmark pathway ranking
# =====================================================================
def generate_fig1(pw, output_dir):
    fig = plt.figure(figsize=(11.5, 17.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.32, 0.68], hspace=0.07)

    # -- Panel A: Scoring schema --
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, 10); ax_a.set_ylim(-0.1, 7.4); ax_a.axis('off')

    ax_a.text(0.0, 7.3, 'A', fontsize=12, fontweight='bold', fontfamily=FONT, va='top')
    ax_a.text(0.35, 7.3, 'Scoring schema', fontsize=12, fontweight='bold',
              fontfamily=FONT, va='top')

    ax_a.text(1.55, 6.7, 'Data sources', ha='center', fontsize=10,
              fontweight='bold', color='#3A6EA5', fontfamily=FONT)
    ax_a.text(5.2, 6.7, 'Scoring dimensions', ha='center', fontsize=10,
              fontweight='bold', color='#333333', fontfamily=FONT)
    ax_a.text(8.6, 6.7, 'Priority score', ha='center', fontsize=10,
              fontweight='bold', color='#555555', fontfamily=FONT)

    src_labels = [
        "DepMap 23Q4\n[Tsherniak 2017]", "Drug proteome atlas\n[Mitchell 2023]",
        "DrugMap\n[Takahashi 2024]", "SLC-ABPP\n[Kuljanin 2021]",
        "Protein half-lives\n[Li 2021]", "UniProt sequences\n[LaRowe 2011]",
    ]
    dim_labels = [
        "E  Context-specific essentiality\n|Chronos| \u00d7 (1 \u2212 PE)",
        "D  Drug-proteome response\nmean |log\u2082FC|, 875 compounds",
        "L  Cysteine ligandability\nCRA site count (capped 30)",
        "R  Cysteine reactivity\nmax competitive ratio (capped 20)",
        "F  Protein lability\n1 if half-life < 8 h, else 0",
        "Zc  Oxidation state deviation\n|Zc \u2212 compartment mean|",
    ]
    y_positions = [5.85, 4.95, 4.05, 3.15, 2.25, 1.35]
    box_w, box_h, dim_w = 2.3, 0.68, 3.2

    for i, (txt, y) in enumerate(zip(src_labels, y_positions)):
        b = FancyBboxPatch((0.4, y - box_h/2), box_w, box_h,
                            boxstyle="round,pad=0.06", facecolor=SRC_COLORS[i],
                            edgecolor='white', linewidth=0.8)
        ax_a.add_patch(b)
        ax_a.text(0.4 + box_w/2, y, txt, ha='center', va='center',
                  fontsize=8.5, color='white', fontweight='bold',
                  fontfamily=FONT, linespacing=1.3)

    for i, (txt, y) in enumerate(zip(dim_labels, y_positions)):
        b = FancyBboxPatch((3.5, y - box_h/2), dim_w, box_h,
                            boxstyle="round,pad=0.06", facecolor=DIM_BG,
                            edgecolor=DIM_BORDER, linewidth=0.8)
        ax_a.add_patch(b)
        ax_a.text(3.5 + dim_w/2, y, txt, ha='center', va='center',
                  fontsize=8.5, color='#1a1a1a', fontfamily=FONT, linespacing=1.3)

    for y in y_positions:
        ax_a.annotate('', xy=(3.5, y), xytext=(2.7, y),
                      arrowprops=dict(arrowstyle='->', color=ARROW_CLR, lw=0.9,
                                      shrinkA=0, shrinkB=2))

    score_w, score_h = 2.1, 1.8
    sx, sy = 7.55, 4.5
    b = FancyBboxPatch((sx, sy - score_h/2), score_w, score_h,
                        boxstyle="round,pad=0.12", facecolor=SCORE_BG,
                        edgecolor='#999999', linewidth=1.0)
    ax_a.add_patch(b)
    cx = sx + score_w / 2
    for dy, txt in [(0.45, "S = 0.30\u00d7E  +  0.30\u00d7D"),
                     (0.0,  "  + 0.15\u00d7L  +  0.15\u00d7R"),
                     (-0.45, "  + 0.05\u00d7F  +  0.05\u00d7Zc")]:
        ax_a.text(cx, sy + dy, txt, ha='center', va='center', fontsize=9,
                  fontfamily='monospace', color=SCORE_TEXT)

    for y in y_positions:
        ax_a.annotate('', xy=(sx, sy), xytext=(3.5 + dim_w, y),
                      arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=0.5,
                                      connectionstyle='arc3,rad=0.05',
                                      shrinkA=0, shrinkB=4))

    out_w, out_h = 2.1, 1.15
    ox, oy = 7.55, 2.0
    b = FancyBboxPatch((ox, oy - out_h/2), out_w, out_h,
                        boxstyle="round,pad=0.12", facecolor=SCORE_BG,
                        edgecolor='#999999', linewidth=1.0)
    ax_a.add_patch(b)
    cx2 = ox + out_w / 2
    ax_a.text(cx2, oy + 0.2, "50 MSigDB Hallmark", ha='center', va='center',
              fontsize=9, fontweight='bold', fontfamily=FONT, color=SCORE_TEXT)
    ax_a.text(cx2, oy - 0.2, "pathways ranked", ha='center', va='center',
              fontsize=9, fontweight='bold', fontfamily=FONT, color=SCORE_TEXT)
    ax_a.annotate('', xy=(cx2, oy + out_h/2), xytext=(cx, sy - score_h/2),
                  arrowprops=dict(arrowstyle='->', color=ARROW_CLR, lw=1.2,
                                  shrinkA=2, shrinkB=2))

    # -- Panel B: Hallmark pathway ranking --
    pw_sorted = pw.sort_values("score", ascending=False).reset_index(drop=True)
    ax_b = fig.add_subplot(gs[1])
    ax_b.text(-0.02, 1.012, 'B', transform=ax_b.transAxes, fontsize=12,
              fontweight='bold', fontfamily=FONT, va='top')
    ax_b.text(0.025, 1.012,
              'MSigDB Hallmark pathway priority score  (50 pathways, HCT116)',
              transform=ax_b.transAxes, fontsize=12, fontweight='bold',
              fontfamily=FONT, va='top')

    n = len(pw_sorted)
    colors, label_colors = [], []
    for _, row in pw_sorted.iterrows():
        pname = row['pathway']
        if pname == 'Apoptosis':
            colors.append(BAR_RED); label_colors.append(BAR_RED)
        elif pname == 'Oxidative Phosphorylation':
            colors.append(BAR_ORANGE); label_colors.append(BAR_ORANGE)
        else:
            colors.append(BAR_GRAY); label_colors.append('#333333')

    ax_b.barh(range(n), pw_sorted['score'], height=0.72, color=colors, edgecolor='none')
    ax_b.set_yticks(range(n))
    ax_b.set_yticklabels(pw_sorted['pathway'], fontsize=9, fontfamily=FONT)
    ax_b.invert_yaxis()
    for i, lc in enumerate(label_colors):
        if lc != '#333333':
            ax_b.get_yticklabels()[i].set_color(lc)
            ax_b.get_yticklabels()[i].set_fontweight('bold')

    highlight_names = {'Apoptosis', 'Oxidative Phosphorylation'}
    for i, (_, row) in enumerate(pw_sorted.iterrows()):
        if i < 10 or row['pathway'] in highlight_names:
            ax_b.text(row['score'] + 0.004, i, f"{row['score']:.3f}",
                      va='center', fontsize=8, color='#444444', fontfamily=FONT)

    apop_row = pw_sorted[pw_sorted['pathway'] == 'Apoptosis'].iloc[0]
    apop_y = pw_sorted.index[pw_sorted['pathway'] == 'Apoptosis'][0]
    ax_b.text(apop_row['score'] - 0.01, apop_y, f"rank {int(apop_row['rank'])}",
              va='center', ha='right', fontsize=8, color='white',
              fontweight='bold', fontfamily=FONT)

    ax_b.set_xlabel('Priority score', fontsize=11, fontfamily=FONT)
    ax_b.set_xlim(0, pw_sorted['score'].max() * 1.14)
    ax_b.tick_params(axis='y', length=0); ax_b.tick_params(axis='x', labelsize=10)
    ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)

    fig.savefig(os.path.join(output_dir, 'Fig1_combined.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    print('  Fig 1 saved.')


# =====================================================================
# FIGURE 2 — BH3-apoptosis axis (BCL-W / MCL1)
# =====================================================================
def generate_fig2(genes, apop, hl_raw, zc_dedup, output_dir):
    n_apop = len(apop)
    fig, axes = plt.subplots(4, 2, figsize=(12, 18.5),
                             gridspec_kw={'hspace': 0.38, 'wspace': 0.35})

    # 2A — Proteome-wide score curve
    ax = axes[0, 0]; panel_label(ax, 'A', 'Gene-level priority score')
    gs = genes.sort_values('rank')
    ax.plot(gs['rank'], gs['score'], color='#AAAAAA', linewidth=1.0)
    for gene, label in [('BCL2L2', 'BCL-W'), ('MCL1', 'MCL1')]:
        r = genes[genes['gene'] == gene].iloc[0]
        ax.scatter(r['rank'], r['score'], c=RED, s=70, zorder=5,
                  edgecolors='white', linewidth=0.6)
        ax.annotate(f"{label}  (#{int(r['rank'])})", (r['rank'], r['score']),
                    textcoords='offset points', xytext=(12, 4),
                    fontsize=10, color=RED, fontweight='bold', fontfamily=FONT)
    ax.set_xlabel('Gene rank (of 9,723)'); ax.set_ylabel('Priority score')
    ax.set_xlim(0, 10000); ax.set_ylim(0, 0.42)

    # 2B — Top 25 Apoptosis Hallmark genes
    ax = axes[0, 1]; panel_label(ax, 'B', f'Apoptosis Hallmark \u2014 top 25 genes (of {n_apop})')
    top25 = apop.head(25).sort_values('score', ascending=True)
    labels_b = [('BCL-W' if g == 'BCL2L2' else g) for g in top25['gene']]
    colors_b = [RED if g in ('BCL2L2', 'MCL1') else BAR_GRAY for g in top25['gene']]
    ax.barh(range(len(top25)), top25['score'], color=colors_b, height=0.72, edgecolor='none')
    ax.set_yticks(range(len(top25)))
    ax.set_yticklabels(labels_b, fontsize=9, fontfamily=FONT)
    for i, g in enumerate(top25['gene']):
        if g in ('BCL2L2', 'MCL1'):
            ax.get_yticklabels()[i].set_color(RED)
            ax.get_yticklabels()[i].set_fontweight('bold')
    ax.set_xlabel('Priority score')

    # 2C — PE vs Chronos scatter
    ax = axes[1, 0]; panel_label(ax, 'C', 'Essentiality vs pan-essentiality')
    ax.scatter(genes['pan_essentiality'], genes['chronos_hct116'],
               c=GRAY_DOT, s=3, alpha=0.4, rasterized=True, zorder=1)
    apop_data = genes[genes['gene'].isin(apop['gene'])]
    ax.scatter(apop_data['pan_essentiality'], apop_data['chronos_hct116'],
               c='#FFAAAA', s=18, alpha=0.7, zorder=3, label='Apoptosis Hallmark')
    for gene, label, sz in [('BCL2L2', 'BCL-W', 90), ('MCL1', 'MCL1', 70)]:
        r = genes[genes['gene'] == gene].iloc[0]
        ax.scatter(r['pan_essentiality'], r['chronos_hct116'],
                  c=RED, s=sz, zorder=6, edgecolors='white', linewidth=0.8)
        offset = (10, 5) if gene == 'BCL2L2' else (10, 8)
        ax.annotate(label, (r['pan_essentiality'], r['chronos_hct116']),
                    textcoords='offset points', xytext=offset,
                    fontsize=10, color=RED, fontweight='bold', fontfamily=FONT)
    ax.axhline(-0.5, color=GRAY_LINE, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.axvline(0.2, color=GRAY_LINE, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('Pan-essentiality'); ax.set_ylabel('Chronos score (HCT116)')
    ax.text(0.02, -2.7, 'context-specific\nessential', fontsize=9,
            fontstyle='italic', color='gray', fontfamily=FONT)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8, markerscale=1.5)

    # 2D — BCL-W drug-proteome responses
    ax = axes[1, 1]; panel_label(ax, 'D', 'BCL-W drug-proteome responses')
    _drug_bar(ax, 'BCL-W log\u2082FC',
              ['AXL1717','JP1302','Nexturastat A','G-1','Vorinostat'],
              ['IGF1R','ADRA2C','HDAC6','GPER1','HDAC1;HDAC3'],
              [-0.9,-0.7,-0.6,-0.5,-0.4],
              ['4-Aminobenzo...','Pictilisib','Bromopyruvic...','OTSSP167','Torcetrapib'],
              ['MPO','PIK3CA','HK2','MELK','CETP'],
              [0.35,0.4,0.5,0.6,0.8], BLUE, RED)

    # 2E — MCL1 drug-proteome responses
    ax = axes[2, 0]; panel_label(ax, 'E', 'MCL1 drug-proteome responses')
    _drug_bar(ax, 'MCL1 log\u2082FC',
              ['OTSSP167','Flavopiridol','JP1302','PF-3758309','CBL0137'],
              ['MELK','CDK9','ADRA2C','PAK4','SSRP1'],
              [-7.0,-5.0,-3.5,-3.0,-2.5],
              ['Epoxomicin','ONX 0914','MG-132','TPPB','NMS-873'],
              ['PSMB5','PSMB8','PSMB5','PRKCA','VCP'],
              [2.5,3.0,3.5,4.0,6.5], BLUE, RED)

    # 2F — Cysteine druggability
    ax = axes[2, 1]; panel_label(ax, 'F', 'Cysteine druggability')
    ax.scatter(genes['L'].clip(upper=30), genes['R'].clip(upper=20),
               c=GRAY_DOT, s=6, alpha=0.25, rasterized=True, label='All detected proteins')
    for gene, label, cra, cr in [('BCL2L2','BCL-W',3,20.0), ('MCL1','MCL1',1,0)]:
        ax.scatter(cra, cr, c=RED, s=80, zorder=5, edgecolors='white', linewidth=0.8,
                  label='BCL-W / MCL1' if gene == 'BCL2L2' else None)
        cr_txt = str(cr) if cr > 0 else 'not detected'
        ax.annotate(f"{label}\n(CRA={cra}, CR={cr_txt})", (cra, cr),
                    textcoords='offset points', xytext=(12, -8 if cr > 10 else 8),
                    fontsize=9, color=RED, fontfamily=FONT)
    ax.set_xlabel('DrugMap cysteine-reactive sites (CRA)')
    ax.set_ylabel('SLC-ABPP max competitive ratio (CR)')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

    # 2G — Protein half-life KDE
    ax = axes[3, 0]; panel_label(ax, 'G', 'Protein half-life (HCT116)')
    hl_min = hl_raw.groupby('gene')['halflife_h'].min()
    vals = hl_min[hl_min <= 40].values
    kde = gaussian_kde(vals); x = np.linspace(0, 15, 300)
    ax.fill_between(x, kde(x), alpha=0.25, color='gray')
    ax.plot(x, kde(x), color='gray', linewidth=1.0)
    ax.axvline(0.78, color=RED, linewidth=1.5)
    ax.text(0.78 + 0.3, kde(np.array([0.78]))[0] * 0.7, 'MCL1\n0.78 h',
            fontsize=10, color=RED, fontfamily=FONT, fontweight='bold')
    ax.axvline(13.5, color=RED, linewidth=1.2, linestyle='--')
    ax.text(10.5, 0.015, 'BCL-W\n>8 h (stable)', fontsize=9,
            color=RED, fontfamily=FONT, fontstyle='italic')
    ax.text(0.97, 0.55, f'All measured proteins (n = {len(vals):,d})',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            color='gray', fontfamily=FONT)
    ax.set_xlabel('Protein half-life (h)'); ax.set_ylabel('Density'); ax.set_xlim(0, 15)

    # 2H — Carbon oxidation state KDE
    ax = axes[3, 1]; panel_label(ax, 'H', 'Carbon oxidation state (Zc)')
    _zc_kde(ax, genes, zc_dedup)
    ax.axvline(-0.095, color=RED, linewidth=1.2, linestyle='--')
    ax.text(0.05, 8.0, 'BCL-W\nZc = \u22120.095', fontsize=9, color=RED,
            fontfamily=FONT, fontweight='bold')
    ax.annotate('', xy=(-0.093, 7.0), xytext=(0.04, 7.5),
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
    ax.axvline(-0.118, color=RED, linewidth=1.2, linestyle='--')
    ax.text(-0.24, 4.0, 'MCL1\nZc = \u22120.118', fontsize=9, color=RED, fontfamily=FONT)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.8)

    cleanup_spines(axes)
    fig.savefig(os.path.join(output_dir, 'Fig2_BH3_axis.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print('  Fig 2 saved.')


# =====================================================================
# FIGURE 3 — Mito-protease axis (CLPX / LONP1 / CLPP)
# =====================================================================
def generate_fig3(genes, zc_dedup, jacques, mc, output_dir):
    oxphos_genes = set()
    for _, row in mc.iterrows():
        paths = str(row.get('MitoCarta3.0_MitoPathways', ''))
        if 'OXPHOS' in paths and 'subunit' in paths.lower():
            oxphos_genes.add(row['Symbol'])

    fig, axes = plt.subplots(4, 2, figsize=(12, 18.5),
                             gridspec_kw={'hspace': 0.38, 'wspace': 0.35})

    # 3A — Score curve
    ax = axes[0, 0]; panel_label(ax, 'A', 'Gene-level priority score')
    gs = genes.sort_values('rank')
    ax.plot(gs['rank'], gs['score'], color='#AAAAAA', linewidth=1.0)
    for gene, clr in [('LONP1', PURPLE), ('CLPX', BLUE), ('CLPP', LTBLUE)]:
        r = genes[genes['gene'] == gene].iloc[0]
        ax.scatter(r['rank'], r['score'], c=clr, s=70, zorder=5,
                  edgecolors='white', linewidth=0.6)
        ax.annotate(f"{gene}  (#{int(r['rank'])})", (r['rank'], r['score']),
                    textcoords='offset points', xytext=(12, 4),
                    fontsize=10, color=clr, fontweight='bold', fontfamily=FONT)
    ax.set_xlabel('Gene rank (of 9,723)'); ax.set_ylabel('Priority score')
    ax.set_xlim(0, 10000); ax.set_ylim(0, 0.42)

    # 3B — PE vs Chronos
    ax = axes[0, 1]; panel_label(ax, 'B', 'Essentiality vs pan-essentiality')
    ax.scatter(genes['pan_essentiality'], genes['chronos_hct116'],
               c=GRAY_DOT, s=3, alpha=0.4, rasterized=True)
    for gene, clr in [('CLPX', BLUE), ('LONP1', PURPLE), ('CLPP', LTBLUE)]:
        r = genes[genes['gene'] == gene].iloc[0]
        ax.scatter(r['pan_essentiality'], r['chronos_hct116'],
                  c=clr, s=80, zorder=6, edgecolors='white', linewidth=0.8, label=gene)
        ofs = {'CLPX': (-35, 10), 'LONP1': (-10, -18), 'CLPP': (10, 10)}
        ax.annotate(gene, (r['pan_essentiality'], r['chronos_hct116']),
                    textcoords='offset points', xytext=ofs[gene],
                    fontsize=10, color=clr, fontweight='bold', fontfamily=FONT)
    ax.axhline(-0.5, color=GRAY_LINE, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.axvline(0.2, color=GRAY_LINE, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('Pan-essentiality'); ax.set_ylabel('Chronos score (HCT116)')
    ax.text(0.02, -2.7, 'context-specific\nessential', fontsize=9,
            fontstyle='italic', color='gray', fontfamily=FONT)
    ax.legend(fontsize=9, loc='lower left', framealpha=0.8, markerscale=1.2)

    # 3C — CLPP drug responses
    ax = axes[1, 0]; panel_label(ax, 'C', 'CLPP drug-proteome responses')
    _drug_bar(ax, 'CLPP log\u2082FC',
              ['WH-4-023','Nexturastat A','EIPA','JQ-1','GBR 12909'],
              ['LCK','HDAC6','SLC9A1','BRD4','SLC6A3'],
              [-1.2,-0.9,-0.7,-0.6,-0.5],
              ['AZD7762','PF-3758309','Binospirone','ML-323','4-Nitropiazt...'],
              ['CHEK1','PAK4','HTR1A','USP1','TXNRD1'],
              [0.25,0.3,0.35,0.4,0.5], BLUE, BAR_UP)

    # 3D — CLPX drug responses
    ax = axes[1, 1]; panel_label(ax, 'D', 'CLPX drug-proteome responses')
    _drug_bar(ax, 'CLPX log\u2082FC',
              ['GBR 12909','MG-132','Epoxomicin','UNC 0638','ONX 0914'],
              ['SLC6A3','PSMB5','PSMB5','EHMT2','PSMB8'],
              [-2.0,-1.5,-1.0,-0.8,-0.6],
              ['ICG-001','AZ20','Brequinar','Nutlin3a','Bombesin'],
              ['CTNNB1','ATR','DHODH','MDM2','GRPR'],
              [0.35,0.4,0.5,0.6,0.8], BLUE, BAR_UP)

    # 3E — LONP1 drug responses
    ax = axes[2, 0]; panel_label(ax, 'E', 'LONP1 drug-proteome responses')
    _drug_bar(ax, 'LONP1 log\u2082FC',
              ['MLN4924','Epoxomicin','UNC 0638','RG2833','ONX 0914'],
              ['NAE1','PSMB5','EHMT2','HDAC3','PSMB8'],
              [-0.9,-0.7,-0.6,-0.5,-0.45],
              ['PF-431396','TAE 226','ICG-001','Silmitasertib','9-Aminocampt...'],
              ['PTK2','PTK2','CTNNB1','CSNK2A1','TOP1'],
              [0.3,0.35,0.4,0.5,0.6], PURPLE, BAR_UP)

    # 3F — Cysteine druggability
    ax = axes[2, 1]; panel_label(ax, 'F', 'Cysteine druggability')
    ax.scatter(genes['L'].clip(upper=30), genes['R'].clip(upper=20),
               c=GRAY_DOT, s=6, alpha=0.25, rasterized=True, label='All detected proteins')
    ax.scatter(2, 2.57, c=BLUE, s=80, zorder=5, edgecolors='white', linewidth=0.8, label='CLPX')
    ax.annotate('CLPX\n(CRA=2, CR=2.57)', (2, 2.57), textcoords='offset points',
                xytext=(12, 12), fontsize=9, color=BLUE, fontfamily=FONT)
    ax.scatter(18, 2.03, c=PURPLE, s=80, zorder=5, edgecolors='white', linewidth=0.8, label='LONP1')
    ax.annotate('LONP1\n(CRA=18, CR=2.03)', (18, 2.03), textcoords='offset points',
                xytext=(5, 12), fontsize=9, color=PURPLE, fontfamily=FONT)
    ax.scatter(0, 0, facecolors='none', edgecolors=LTBLUE, s=60, zorder=5,
               linewidth=1.5, label='CLPP (not detected)')
    ax.set_xlabel('DrugMap cysteine-reactive sites (CRA)')
    ax.set_ylabel('SLC-ABPP max competitive ratio (CR)')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

    # 3G — Zc KDE
    ax = axes[3, 0]; panel_label(ax, 'G', 'Carbon oxidation state (Zc)')
    _zc_kde(ax, genes, zc_dedup)
    for zc_val, clr, ls, label in [(-0.163, LTBLUE, ':', 'CLPP'),
                                     (-0.162, PURPLE, '--', 'LONP1'),
                                     (-0.126, BLUE, '--', 'CLPX')]:
        ax.axvline(zc_val, color=clr, linewidth=1.2, linestyle=ls)
    ax.text(-0.30, 6.5, 'CLPP\nZc=\u22120.163', fontsize=9, color=LTBLUE, fontfamily=FONT)
    ax.text(-0.30, 4.5, 'LONP1\nZc=\u22120.162', fontsize=9, color=PURPLE, fontfamily=FONT)
    ax.text(-0.06, 6.5, 'CLPX\nZc=\u22120.126', fontsize=9, color=BLUE, fontfamily=FONT)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

    # 3H — ONC212 proteomics
    ax = axes[3, 1]; panel_label(ax, 'H', 'ONC212 proteomics (NALM6)')
    ax.scatter(jacques['wt_fc'], jacques['kd_fc'], c=GRAY_DOT, s=4, alpha=0.3, rasterized=True)
    ox_data = jacques[jacques['gene'].isin(oxphos_genes)]
    ax.scatter(ox_data['wt_fc'], ox_data['kd_fc'], c=ORANGE, s=18, alpha=0.6, zorder=3,
              label='OXPHOS subunits')
    for gene, clr, sz in [('CLPX', BLUE, 100), ('LONP1', PURPLE, 100), ('CLPP', LTBLUE, 100)]:
        row = jacques[jacques['gene'] == gene]
        if len(row):
            r = row.iloc[0]
            ax.scatter(r['wt_fc'], r['kd_fc'], c=clr, s=sz, zorder=6,
                      edgecolors='white', linewidth=1.5, label=gene)
            ofs = {'CLPX': (8, -15), 'LONP1': (8, 8), 'CLPP': (8, -12)}
            ax.annotate(gene, (r['wt_fc'], r['kd_fc']), textcoords='offset points',
                        xytext=ofs[gene], fontsize=10, color=clr, fontweight='bold',
                        fontfamily=FONT)
    for gene in ['ATP5IF1', 'NDUFAF2']:
        row = jacques[jacques['gene'] == gene]
        if len(row):
            r = row.iloc[0]
            ax.scatter(r['wt_fc'], r['kd_fc'], c=ORANGE, s=45, zorder=4,
                      edgecolors='black', linewidth=0.5)
            ax.annotate(gene, (r['wt_fc'], r['kd_fc']), textcoords='offset points',
                        xytext=(5, 5), fontsize=9, color='#E65100', fontstyle='italic',
                        fontfamily=FONT)
    ax.axhline(0, color='gray', linewidth=0.3); ax.axvline(0, color='gray', linewidth=0.3)
    lim = 9
    ax.plot([-lim, lim], [-lim, lim], '--', color='gray', linewidth=0.5, alpha=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('log\u2082FC (ONC212, WT NALM6)')
    ax.set_ylabel('log\u2082FC (ONC212, CLPP-KD NALM6)')
    ax.legend(fontsize=8, loc='lower right', framealpha=0.8, markerscale=0.8)

    cleanup_spines(axes)
    fig.savefig(os.path.join(output_dir, 'Fig3_protease_axis.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print('  Fig 3 saved.')


# =====================================================================
# FIGURE 4 — Ferroptosis axis (GPX4)
# =====================================================================
def generate_fig4(genes, zc_dedup, kud, output_dir):
    ferro_genes = ['GPX4', 'AIFM2', 'DHODH', 'COQ7', 'COQ10A', 'TXNRD2', 'VDAC2', 'IDH2']
    fig, axes = plt.subplots(3, 2, figsize=(12, 14),
                             gridspec_kw={'hspace': 0.38, 'wspace': 0.35})

    # 4A — Score curve
    ax = axes[0, 0]; panel_label(ax, 'A', 'Gene-level priority score')
    gs = genes.sort_values('rank')
    ax.plot(gs['rank'], gs['score'], color='#AAAAAA', linewidth=1.0)
    r = genes[genes['gene'] == 'GPX4'].iloc[0]
    ax.scatter(r['rank'], r['score'], c=DK_GREEN, s=70, zorder=5,
              edgecolors='white', linewidth=0.6)
    ax.annotate(f"GPX4  (#{int(r['rank'])})", (r['rank'], r['score']),
                textcoords='offset points', xytext=(12, 4),
                fontsize=10, color=DK_GREEN, fontweight='bold', fontfamily=FONT)
    ax.set_xlabel('Gene rank (of 9,723)'); ax.set_ylabel('Priority score')
    ax.set_xlim(0, 10000); ax.set_ylim(0, 0.42)

    # 4B — PE vs Chronos with ferroptosis genes
    ax = axes[0, 1]; panel_label(ax, 'B', 'Essentiality vs pan-essentiality')
    ax.scatter(genes['pan_essentiality'], genes['chronos_hct116'],
               c=GRAY_DOT, s=3, alpha=0.4, rasterized=True)
    for g in ferro_genes:
        if g == 'GPX4': continue
        row = genes[genes['gene'] == g]
        if len(row):
            r = row.iloc[0]
            ax.scatter(r['pan_essentiality'], r['chronos_hct116'],
                      c=LT_GREEN, s=40, zorder=4, edgecolors='white', linewidth=0.5)
            offsets = {'AIFM2': (15, 12), 'DHODH': (10, -15), 'COQ7': (12, 8),
                       'COQ10A': (10, 12), 'TXNRD2': (12, 8), 'VDAC2': (10, -14), 'IDH2': (-15, 10)}
            ax.annotate(g, (r['pan_essentiality'], r['chronos_hct116']),
                        textcoords='offset points', xytext=offsets.get(g, (10, 8)),
                        fontsize=8, color='#2E7D32', fontfamily=FONT,
                        arrowprops=dict(arrowstyle='-', color='#2E7D32', lw=0.6, shrinkB=2))
    r = genes[genes['gene'] == 'GPX4'].iloc[0]
    ax.scatter(r['pan_essentiality'], r['chronos_hct116'],
              c=DK_GREEN, s=90, zorder=6, edgecolors='white', linewidth=0.8)
    ax.annotate('GPX4', (r['pan_essentiality'], r['chronos_hct116']),
                textcoords='offset points', xytext=(10, 5),
                fontsize=10, color=DK_GREEN, fontweight='bold', fontfamily=FONT)
    ax.axhline(-0.5, color=GRAY_LINE, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.axvline(0.2, color=GRAY_LINE, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('Pan-essentiality'); ax.set_ylabel('Chronos score (HCT116)')
    ax.text(0.02, -2.7, 'context-specific\nessential', fontsize=9,
            fontstyle='italic', color='gray', fontfamily=FONT)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY_DOT, markersize=5, label='All proteome'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DK_GREEN, markersize=8, label='GPX4'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LT_GREEN, markersize=6, label='Ferroptosis pathway'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower left', framealpha=0.8)

    # 4C — GPX4 drug responses
    ax = axes[1, 0]; panel_label(ax, 'C', 'GPX4 drug-proteome responses')
    _drug_bar(ax, 'GPX4 log\u2082FC',
              ['Bafilomycin ...','NVP-BSK805','STK410283','Obatoclax','RN486'],
              ['ATP6V1H','JAK2','MAP3K9;TNIK','BCL2L1','BTK'],
              [-1.3,-1.0,-0.8,-0.7,-0.5],
              ['PHA 767491','AS 1517499','Flavopiridol','9-Aminocampt...','Dasatinib'],
              ['CDC7','STAT6','CDK9','TOP1','ABL1'],
              [0.4,0.5,0.6,0.8,1.0], BLUE, BAR_GREEN)

    # 4D — Cysteine druggability
    ax = axes[1, 1]; panel_label(ax, 'D', 'Cysteine druggability')
    ax.scatter(genes['L'].clip(upper=30), genes['R'].clip(upper=20),
               c=GRAY_DOT, s=6, alpha=0.25, rasterized=True, label='All detected proteins')
    for g in ferro_genes:
        if g == 'GPX4': continue
        row = genes[genes['gene'] == g]
        if len(row):
            r = row.iloc[0]
            if r['L'] > 0 or r['R'] > 0:
                ax.scatter(r['L'], r['R'], c=LT_GREEN, s=35, zorder=4, edgecolors='white', linewidth=0.5)
    ax.scatter(8, 7.24, c=DK_GREEN, s=80, zorder=5, edgecolors='white', linewidth=0.8, label='GPX4')
    ax.annotate('GPX4\n(CRA=8, CR=7.24)', (8, 7.24), textcoords='offset points',
                xytext=(12, 8), fontsize=9, color=DK_GREEN, fontfamily=FONT)
    ax.scatter([], [], c=LT_GREEN, s=35, label='Ferroptosis pathway')
    ax.set_xlabel('DrugMap cysteine-reactive sites (CRA)')
    ax.set_ylabel('SLC-ABPP max competitive ratio (CR)')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

    # 4E — Zc KDE
    ax = axes[2, 0]; panel_label(ax, 'E', 'Carbon oxidation state (Zc)')
    _zc_kde(ax, genes, zc_dedup)
    ax.axvline(-0.156, color=DK_GREEN, linewidth=1.5)
    ax.text(-0.07, 5.0, 'GPX4\nZc = \u22120.156', fontsize=10, color=DK_GREEN,
            fontfamily=FONT, fontweight='bold')
    ax.annotate('', xy=(-0.154, 4.5), xytext=(-0.08, 4.7),
                arrowprops=dict(arrowstyle='->', color=DK_GREEN, lw=0.8))
    ax.legend(fontsize=9, loc='upper left', framealpha=0.8)

    # 4F — ML210 proteomics
    ax = axes[2, 1]; panel_label(ax, 'F', 'GPX4 inhibition proteomics (ML210, MEFs)')
    kud_both = kud.dropna(subset=['log2fc_24h', 'log2fc_48h'])
    ax.scatter(kud_both['log2fc_24h'], kud_both['log2fc_48h'],
               c=GRAY_DOT, s=5, alpha=0.3, rasterized=True, label='All proteins')
    ferro_mouse = {'Hmox1': 'HMOX1', 'Fth1': 'FTH1', 'Acsl4': 'ACSL4', 'Tfrc': 'TFRC', 'Cth': 'CTH'}
    offsets_ml = {'HMOX1': (8, -10), 'FTH1': (5, 8), 'ACSL4': (5, -12), 'TFRC': (5, 8), 'CTH': (-10, -12)}
    for msym, hsym in ferro_mouse.items():
        row = kud_both[kud_both['Gene'] == msym]
        if len(row):
            r = row.iloc[0]
            ax.scatter(r['log2fc_24h'], r['log2fc_48h'], c=DK_GREEN, s=50, zorder=5,
                      edgecolors='white', linewidth=0.6)
            ax.annotate(hsym, (r['log2fc_24h'], r['log2fc_48h']),
                        textcoords='offset points', xytext=offsets_ml.get(hsym, (5, 5)),
                        fontsize=9, color=DK_GREEN, fontfamily=FONT)
    for msym, hsym in {'Akap1': 'AKAP1', 'Prkag1': 'PRKAG1'}.items():
        row = kud_both[kud_both['Gene'] == msym]
        if len(row):
            r = row.iloc[0]
            ax.scatter(r['log2fc_24h'], r['log2fc_48h'], c=RED_UP, s=50, zorder=5,
                      edgecolors='white', linewidth=0.6)
            ofs = (8, 5) if hsym == 'AKAP1' else (8, -8)
            ax.annotate(hsym, (r['log2fc_24h'], r['log2fc_48h']),
                        textcoords='offset points', xytext=ofs,
                        fontsize=9, color=RED_UP, fontstyle='italic', fontfamily=FONT)
    ax.axhline(0, color='gray', linewidth=0.3); ax.axvline(0, color='gray', linewidth=0.3)
    lim = 3.5
    ax.plot([-lim, lim], [-lim, lim], '--', color='gray', linewidth=0.5, alpha=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.text(-lim * 0.9, -lim * 0.85, 'GPX4 not detected\nin this dataset',
            fontsize=9, fontstyle='italic', color='gray', fontfamily=FONT)
    ax.scatter([], [], c=DK_GREEN, s=50, label='Ferroptosis / ROS genes')
    ax.scatter([], [], c=RED_UP, s=50, label='Top upregulated')
    ax.set_xlabel('log\u2082FC (24 h, ML210)'); ax.set_ylabel('log\u2082FC (48 h, ML210)')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.8, markerscale=0.8)

    cleanup_spines(axes)
    fig.savefig(os.path.join(output_dir, 'Fig4_ferroptosis_axis.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print('  Fig 4 saved.')


# =====================================================================
# FIGURE 5 — Cross-axis co-regulation + BioPlex
# =====================================================================
def generate_fig5(output_dir):
    labels = ['BCL-W', 'MCL1', 'GPX4', 'CLPX', 'LONP1']
    corr = np.array([
        [np.nan, -0.02, -0.09, 0.31, 0.13],
        [np.nan, np.nan, -0.29, -0.03, -0.43],
        [np.nan, np.nan, np.nan, -0.02, 0.24],
        [np.nan, np.nan, np.nan, np.nan, 0.08],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ])
    sig = np.array([[0,0,1,1,1],[0,0,1,0,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,0]])
    bioplex = np.array([
        [np.nan, 3, 0, 0, 0],
        [np.nan, np.nan, 0, 0, 0],
        [np.nan, np.nan, np.nan, 0, 0],
        [np.nan, np.nan, np.nan, np.nan, 1],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ])
    axis_colors = [RED, RED, DK_GREEN, BLUE, BLUE]
    group_info = [('BH3 Apoptosis', RED, 0, 1), ('Ferroptosis', DK_GREEN, 2, 2),
                  ('Mito-protease', BLUE, 3, 4)]
    n = 5

    fig = plt.figure(figsize=(16.5, 5.5))
    ax_a = fig.add_axes([0.03, 0.10, 0.33, 0.78])
    ax_cb = fig.add_axes([0.42, 0.10, 0.012, 0.78])
    ax_b = fig.add_axes([0.53, 0.10, 0.33, 0.78])

    def draw_upper_triangle(ax, matrix, cmap, vmin, vmax, fmt_func, cb_ax=None, cb_label=None):
        ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis(); ax.set_aspect('equal'); ax.axis('off')
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, labels[i], ha='center', va='center', fontsize=11,
                            fontweight='bold', fontfamily=FONT, color=axis_colors[i])
                elif j > i:
                    val = matrix[i, j]
                    if np.isnan(val): continue
                    color = sm.to_rgba(val)
                    rect = Rectangle((j-0.48, i-0.48), 0.96, 0.96, facecolor=color,
                                      edgecolor='white', linewidth=1.5)
                    ax.add_patch(rect)
                    brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                    ax.text(j, i, fmt_func(i, j, val), ha='center', va='center',
                            fontsize=11, color='white' if brightness < 0.55 else 'black',
                            fontfamily=FONT)
        for gname, gcolor, i_start, i_end in group_info:
            y_mid = (i_start + i_end) / 2
            bar_x = n - 0.25
            ax.plot([bar_x, bar_x], [i_start - 0.4, i_end + 0.4],
                    color=gcolor, linewidth=3, solid_capstyle='round')
            ax.text(bar_x + 0.3, y_mid, gname, fontsize=10, color=gcolor,
                    fontfamily=FONT, va='center', ha='left')
        if cb_ax:
            cb = plt.colorbar(sm, cax=cb_ax)
            cb.set_label(cb_label, fontsize=11, fontfamily=FONT)
            cb.ax.tick_params(labelsize=10)

    ax_a.text(0.0, 1.08, 'A', transform=ax_a.transAxes, fontsize=12,
              fontweight='bold', fontfamily=FONT, va='top')
    ax_a.text(0.06, 1.08, 'Drug-proteome co-regulation (875 compounds)',
              transform=ax_a.transAxes, fontsize=12, fontweight='bold', fontfamily=FONT, va='top')
    draw_upper_triangle(ax_a, corr, 'RdBu_r', -0.6, 0.6,
                        lambda i, j, v: f'{v:.2f}*' if sig[i, j] else f'{v:.2f}',
                        cb_ax=ax_cb, cb_label='Pearson r')

    ax_b.text(0.0, 1.08, 'B', transform=ax_b.transAxes, fontsize=12,
              fontweight='bold', fontfamily=FONT, va='top')
    ax_b.text(0.06, 1.08, 'Shared BioPlex 3.0 interaction partners (HCT116)',
              transform=ax_b.transAxes, fontsize=12, fontweight='bold', fontfamily=FONT, va='top')
    draw_upper_triangle(ax_b, bioplex, 'YlOrBr', 0, 5,
                        lambda i, j, v: str(int(v)))

    fig.savefig(os.path.join(output_dir, 'Fig5_cross_axis.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print('  Fig 5 saved.')


# =====================================================================
# FIGURE S1 — Dirichlet sensitivity analysis
# =====================================================================
def generate_figS1(pw, output_dir):
    pw_sorted = pw.sort_values("rank").reset_index(drop=True)
    w0 = np.array([0.30, 0.30, 0.15, 0.15, 0.05, 0.05])
    raw_cols = ["E_raw", "D_raw", "L_raw", "R_raw", "F_raw", "Zc_raw"]
    raw_vals = pw_sorted[raw_cols].values
    n_pw, N_ITER = len(pw_sorted), 50000

    rank_history = np.zeros((N_ITER, n_pw), dtype=int)
    np.random.seed(42)
    for i in range(N_ITER):
        w = np.random.dirichlet(w0 * 30)
        norm = np.zeros_like(raw_vals)
        for d in range(6):
            vmin, vmax = raw_vals[:, d].min(), raw_vals[:, d].max()
            if vmax > vmin: norm[:, d] = (raw_vals[:, d] - vmin) / (vmax - vmin)
        rank_history[i] = (-(norm @ w)).argsort().argsort() + 1

    apop_idx = pw_sorted[pw_sorted['pathway'] == 'Apoptosis'].index[0]
    apop_ranks = rank_history[:, apop_idx]
    baseline_rank = int(pw_sorted.loc[apop_idx, 'rank'])
    median_rank = int(np.median(apop_ranks))
    top10_pct = (apop_ranks <= 10).mean() * 100
    p5, p95 = int(np.percentile(apop_ranks, 5)), int(np.percentile(apop_ranks, 95))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'wspace': 0.35})

    # S1A
    ax = axes[0]
    panel_label(ax, 'A', f'Apoptosis rank distribution under Dirichlet perturbation (n = {N_ITER:,d})')
    ax.hist(apop_ranks, bins=np.arange(0.5, 51.5, 1), color=RED, alpha=0.65, edgecolor='white', linewidth=0.5)
    ax.axvline(baseline_rank, color='black', linestyle='--', linewidth=1.5,
               label=f'Baseline rank ({baseline_rank})')
    ax.axvline(median_rank, color='gray', linestyle=':', linewidth=1.0)
    ax.text(0.97, 0.95, f"Median rank: {median_rank}\nP(rank \u2264 10): {top10_pct:.1f}%\n5th\u201395th: {p5}\u2013{p95}",
            transform=ax.transAxes, fontsize=10, fontfamily=FONT, va='top', ha='right')
    ax.set_xlabel('Rank', fontsize=11); ax.set_ylabel('Frequency', fontsize=11)
    ax.set_xlim(0, 30)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # S1B
    ax = axes[1]
    panel_label(ax, 'B', 'Rank stability of top 15 Hallmark pathways')
    top15 = pw_sorted.head(15)
    for i, (_, row) in enumerate(top15.iterrows()):
        ranks = rank_history[:, row.name]
        clr = RED if row['pathway'] == 'Apoptosis' else '#666666'
        lw = 2.5 if row['pathway'] == 'Apoptosis' else 2.0
        ax.plot([np.percentile(ranks, 5), np.percentile(ranks, 95)], [i, i],
                color=clr, linewidth=lw, alpha=0.7, solid_capstyle='round')
        ax.scatter(int(row['rank']), i, color=clr, s=50, marker='s', zorder=5)
        ax.scatter(np.median(ranks), i, color=clr, s=25, marker='o', zorder=5)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(list(top15['pathway']), fontsize=10, fontfamily=FONT)
    for i, pn in enumerate(top15['pathway']):
        if pn == 'Apoptosis':
            ax.get_yticklabels()[i].set_color(RED)
            ax.get_yticklabels()[i].set_fontweight('bold')
    ax.set_xlabel('Rank (5th\u201395th percentile)', fontsize=11)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    leg_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#666666', markersize=7, label='Baseline rank'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=5, label='Median rank'),
        Line2D([0], [0], color='#666666', linewidth=2, label='5th\u201395th percentile'),
    ]
    ax.legend(handles=leg_handles, fontsize=10, loc='upper right', framealpha=0.8)

    fig.savefig(os.path.join(output_dir, 'FigS1_sensitivity.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    print('  Fig S1 saved.')


# =====================================================================
# FIGURE S2 — Protein half-life distribution
# =====================================================================
def generate_figS2(hl_raw, output_dir):
    hl_min = hl_raw.groupby('gene')['halflife_h'].min().reset_index()
    vals = hl_min['halflife_h'].dropna()
    vals_clipped = vals[vals <= 40].values

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_title('Protein half-life distribution (HCT116)', fontsize=12,
                 fontweight='bold', fontfamily=FONT, loc='left', pad=8)
    kde = gaussian_kde(vals_clipped); x = np.linspace(0, 38, 400)
    ax.fill_between(x, kde(x), alpha=0.25, color='gray')
    ax.plot(x, kde(x), color='gray', linewidth=1.0)
    ax.axvline(0.78, color=RED, linewidth=1.5)
    ax.text(0.78 + 0.5, kde(np.array([0.78]))[0] + 0.004, 'MCL1\n0.78 h',
            fontsize=10, color=RED, fontfamily=FONT, fontweight='bold')
    ax.axvline(26.3, color=BLUE, linewidth=1.5)
    ax.text(26.3 + 0.5, 0.006, 'CLPX\n26.3 h', fontsize=10, color=BLUE,
            fontfamily=FONT, fontweight='bold')
    for i, (name, clr) in enumerate([('BCL-W', RED), ('LONP1', PURPLE),
                                       ('CLPP', LTBLUE), ('GPX4', DK_GREEN)]):
        ax.text(33, 0.028 - i * 0.004, f'{name}  >8 h (stable)', fontsize=10,
                color=clr, fontfamily=FONT, fontstyle='italic', ha='left')
    ax.text(0.97, 0.95, f'All measured proteins (n = {len(vals):,d})',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            color='gray', fontfamily=FONT)
    ax.set_xlabel('Protein half-life (h)', fontsize=11); ax.set_ylabel('Density', fontsize=11)
    ax.set_xlim(0, 38)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(output_dir, 'FigS2_halflife.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    print('  Fig S2 saved.')


# =====================================================================
# HELPERS
# =====================================================================
def _drug_bar(ax, xlabel, drugs_dn, tgts_dn, fc_dn, drugs_up, tgts_up, fc_up,
              color_dn, color_up):
    """Draw a drug-response bar chart (down at bottom, up at top)."""
    all_d = drugs_dn + drugs_up
    all_f = fc_dn + fc_up
    all_t = tgts_dn + tgts_up
    bar_c = [color_dn if f < 0 else color_up for f in all_f]
    ax.barh(range(len(all_d)), all_f, color=bar_c, height=0.68, edgecolor='none')
    ax.set_yticks(range(len(all_d)))
    ax.set_yticklabels(all_d, fontsize=9, fontfamily=FONT)
    for j, (fc, tgt) in enumerate(zip(all_f, all_t)):
        pad = max(abs(fc) * 0.04, 0.03)
        xp = fc + (pad if fc > 0 else -pad)
        ax.text(xp, j, tgt, fontsize=8, va='center',
                ha='left' if fc > 0 else 'right', color='#555', fontfamily=FONT)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=11, fontfamily=FONT)


def _zc_kde(ax, genes, zc_dedup):
    """Draw mitochondrial / non-mitochondrial Zc KDE."""
    zc_m = zc_dedup.merge(genes[['gene', 'is_mito']], on='gene', how='inner')
    mito_zc = zc_m.loc[zc_m['is_mito'], 'ZC'].dropna()
    nonmito_zc = zc_m.loc[~zc_m['is_mito'], 'ZC'].dropna()
    xr = np.linspace(-0.45, 0.35, 300)
    kde_m = gaussian_kde(mito_zc); kde_nm = gaussian_kde(nonmito_zc)
    ax.fill_between(xr, kde_m(xr), alpha=0.30, color='#1565C0')
    ax.plot(xr, kde_m(xr), color='#1565C0', linewidth=1.0,
            label=f'Mitochondrial (n={len(mito_zc):,d})')
    ax.fill_between(xr, kde_nm(xr), alpha=0.15, color='gray')
    ax.plot(xr, kde_nm(xr), color='gray', linewidth=1.0,
            label=f'Non-mitochondrial (n={len(nonmito_zc):,d})')
    ax.set_xlabel('Carbon oxidation state (Zc)', fontsize=11, fontfamily=FONT)
    ax.set_ylabel('Density', fontsize=11, fontfamily=FONT)


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate all manuscript figures (Fig 1\u20135, Fig S1\u2013S2).')
    parser.add_argument('--data-dir', default='data/',
                        help='Directory containing input data files')
    parser.add_argument('--results-dir', default='results/',
                        help='Directory containing analysis output CSVs')
    parser.add_argument('--output-dir', default='results/figures/',
                        help='Directory for output PDFs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()

    print('Loading data...')
    genes = pd.read_csv(os.path.join(args.results_dir, 'gene_priority_v6_proteome.csv'))
    pw = pd.read_csv(os.path.join(args.results_dir, 'hallmark_pathway_scores_v6.csv'))
    apop = pd.read_csv(os.path.join(args.results_dir, 'apoptosis_hallmark_genes_v6.csv'))
    hl_raw = pd.read_csv(os.path.join(args.data_dir, 'halflife_hct116.csv'))
    zc_raw = pd.read_csv(os.path.join(args.data_dir, 'protein_zc_table.csv')).rename(columns={'Gene': 'gene'})
    zc_dedup = zc_raw.drop_duplicates(subset='gene', keep='first')
    kud = pd.read_csv(os.path.join(args.data_dir, 'kudryashova_log2fc.csv'))
    jacques = pd.read_excel(os.path.join(args.data_dir, 'Table_S2_Jacques_et_al_accepted.xlsx'),
                            engine='openpyxl')
    jacques = jacques.rename(columns={'Gene symbol': 'gene',
                                       'WT log2 fold-change': 'wt_fc',
                                       'CLPP sh  log2 fold-change': 'kd_fc'})
    mc_path = os.path.join(args.data_dir, 'Human.MitoCarta3.0.xls')
    if not os.path.exists(mc_path):
        mc_path = os.path.join(args.data_dir, 'Human_MitoCarta3_0.xls')
    mc = pd.read_excel(mc_path, sheet_name='A Human MitoCarta3.0')

    print('Generating figures...')
    generate_fig1(pw, args.output_dir)
    generate_fig2(genes, apop, hl_raw, zc_dedup, args.output_dir)
    generate_fig3(genes, zc_dedup, jacques, mc, args.output_dir)
    generate_fig4(genes, zc_dedup, kud, args.output_dir)
    generate_fig5(args.output_dir)
    generate_figS1(pw, args.output_dir)
    generate_figS2(hl_raw, args.output_dir)
    print(f'\nAll 7 figures saved to {args.output_dir}')


if __name__ == '__main__':
    main()
