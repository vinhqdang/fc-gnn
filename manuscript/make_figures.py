"""
Generate all high-resolution figures for the FC-GNN manuscript.
Output: manuscript/figures/*.png at 300 DPI
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

OUT = Path("manuscript/figures")
OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
FONT = 11
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': FONT,
    'axes.titlesize': FONT + 1,
    'axes.labelsize': FONT,
    'xtick.labelsize': FONT - 1,
    'ytick.labelsize': FONT - 1,
    'legend.fontsize': FONT - 1,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv("results/results_table.csv")
DATASETS = ['CIC-IDS-2017', 'UNSW-NB15', 'NF-BoT-IoT',
            'ISCX-Botnet', 'CTU-13', 'N-BaIoT', 'ToN-IoT']
DS_SHORT  = ['CIC', 'UNSW', 'NF-BoT', 'ISCX', 'CTU', 'N-BaI', 'ToN']

CP_MODELS   = ['fc_gnn', 'cfgnn', 'daps', 'rrgnn', 'snaps']
ALL_MODELS  = ['fc_gnn', 'cfgnn', 'daps', 'rrgnn', 'snaps', 'flgnn', 'fgat', 'gcn', 'sage']
MODEL_LABELS = {
    'fc_gnn': 'FC-GNN (ours)',
    'cfgnn':  'CF-GNN',
    'daps':   'DAPS',
    'rrgnn':  'RR-GNN',
    'snaps':  'SNAPS',
    'flgnn':  'FL-GNN',
    'fgat':   'FGAT',
    'gcn':    'GCN',
    'sage':   'GraphSAGE',
}

PALETTE = {
    'fc_gnn': '#E63946',   # red – ours
    'cfgnn':  '#457B9D',
    'daps':   '#2A9D8F',
    'rrgnn':  '#E9C46A',
    'snaps':  '#F4A261',
    'flgnn':  '#6A4C93',
    'fgat':   '#C77DFF',
    'gcn':    '#8D99AE',
    'sage':   '#ADB5BD',
}

def pivot(metric, models=ALL_MODELS):
    rows = []
    for ds in DATASETS:
        row = {}
        for m in models:
            v = df[(df['Dataset'] == ds) & (df['Model'] == m)][metric]
            row[m] = float(v.values[0]) if len(v) and not pd.isna(v.values[0]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=DATASETS)

# ════════════════════════════════════════════════════════════════════════════
# FIG 1 – Architecture diagram (matplotlib)
# ════════════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, label, sublabel='', color='#457B9D', alpha=0.18,
            fontsize=10, textcolor='#1d3557'):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=color, alpha=alpha,
                              edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0),
                label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=textcolor)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.28, sublabel,
                    ha='center', va='center', fontsize=fontsize-2, color=textcolor)

    def arrow(x1, y1, x2, y2, color='#457B9D'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8))

    # Title
    ax.text(7, 5.7, 'FC-GNN Architecture', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1d3557')

    # Input
    box(0.2, 2.2, 1.5, 1.6, 'Input Graph', r'$G=(V,E,X)$', '#8D99AE', 0.22, 9)
    arrow(1.7, 3.0, 2.2, 3.0)

    # FMPL layers
    fmpl_x = 2.2
    box(fmpl_x, 0.3, 3.8, 5.2, '', '', '#457B9D', 0.07)
    ax.text(fmpl_x + 1.9, 5.25, 'Fuzzy Message-Passing Layer  ×L',
            ha='center', fontsize=10, fontweight='bold', color='#1d3557')

    box(fmpl_x+0.15, 3.5, 3.5, 1.4, 'Membership Functions',
        r'$\mu_k(h) = \exp(-(h-c_k)^2 / 2\sigma_k^2)$',
        '#457B9D', 0.25, 8.5)
    box(fmpl_x+0.15, 1.9, 3.5, 1.4, 'T-norm Rule Firing',
        r'$\tau_k(v) = \prod_{u \in N(v)} \mu_k(h_u^{(\ell)})$',
        '#2A9D8F', 0.25, 8.5)
    box(fmpl_x+0.15, 0.4, 3.5, 1.3, 'Defuzzification',
        r'$h_v^{(\ell+1)} = \sum_k \tau_k w_k \,/\, \sum_k \tau_k$',
        '#E9C46A', 0.30, 8.5)

    # small arrows inside FMPL
    arrow(fmpl_x+1.9, 3.5, fmpl_x+1.9, 3.3)
    arrow(fmpl_x+1.9, 1.9, fmpl_x+1.9, 1.7)

    arrow(6.0, 3.0, 6.5, 3.0)

    # Output head
    box(6.5, 2.1, 1.8, 1.8, 'MLP Head',
        r'$\mu_v = \mathrm{softmax}(\mathrm{MLP}(h_v^L))$',
        '#E63946', 0.20, 8.5, '#7d1128')
    arrow(8.3, 3.0, 8.8, 3.0)

    # FMCP block
    box(8.8, 0.3, 4.9, 5.2, '', '', '#6A4C93', 0.07)
    ax.text(8.8+2.45, 5.25, 'Fuzzy Mondrian Conformal Prediction',
            ha='center', fontsize=10, fontweight='bold', color='#1d3557')

    box(8.95, 3.5, 4.6, 1.4, 'Community Partition',
        r'Louvain: $V_\mathrm{cal} \to \{Q_1, \ldots, Q_M\}$',
        '#6A4C93', 0.25, 8.5)
    box(8.95, 1.9, 4.6, 1.4, 'Sugeno Nonconformity Score',
        r'$s_v = 1 - S_{\lambda}(\mu_v,\, y_v)$',
        '#C77DFF', 0.25, 8.5)
    box(8.95, 0.4, 4.6, 1.3, 'Per-Community Quantile',
        r'$\hat{q}_{Q_m}$ = $(|Q_m|+1)(1-\alpha)$-th order stat.',
        '#E63946', 0.22, 8.5)

    arrow(8.8+2.3, 3.5, 8.8+2.3, 3.3)
    arrow(8.8+2.3, 1.9, 8.8+2.3, 1.7)

    # Output arrow + label
    ax.annotate('', xy=(13.8, 3.0), xytext=(13.7, 3.0),
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=2))
    ax.text(13.82, 3.0, r'$\mathcal{C}_\alpha(v)$', fontsize=12,
            va='center', color='#E63946', fontweight='bold')

    fig.savefig(OUT / 'architecture.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ architecture.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 2 – Coverage comparison (all CP models, all datasets)
# ════════════════════════════════════════════════════════════════════════════
def fig_coverage():
    cov = pivot('Coverage', CP_MODELS)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    x = np.arange(len(DATASETS))
    w = 0.14
    offsets = np.linspace(-(len(CP_MODELS)-1)/2, (len(CP_MODELS)-1)/2, len(CP_MODELS)) * w

    for i, m in enumerate(CP_MODELS):
        vals = cov[m].values
        bars = ax.bar(x + offsets[i], vals, width=w*0.9,
                      color=PALETTE[m], label=MODEL_LABELS[m],
                      edgecolor='white', linewidth=0.4,
                      zorder=3,
                      hatch='//' if m == 'fc_gnn' else '')
        # mark bars below target
        for j, v in enumerate(vals):
            if not np.isnan(v) and v < 0.90:
                ax.text(x[j] + offsets[i], v + 0.002, '✗',
                        ha='center', va='bottom', fontsize=7, color='#c62828')

    ax.axhline(0.90, color='#c62828', lw=1.5, ls='--', zorder=4,
               label='Target coverage (1−α=0.90)')
    ax.set_xticks(x)
    ax.set_xticklabels(DS_SHORT)
    ax.set_ylabel('Empirical Coverage')
    ax.set_ylim(0.68, 1.02)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.legend(ncol=3, loc='lower right', framealpha=0.9)
    ax.set_title('Empirical Coverage across Datasets (α = 0.10, target = 0.90)')
    ax.grid(axis='y', alpha=0.3, zorder=0)

    fig.savefig(OUT / 'coverage_comparison.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ coverage_comparison.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 3 – APSS comparison (CP models)
# ════════════════════════════════════════════════════════════════════════════
def fig_apss():
    apss = pivot('APSS', CP_MODELS)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    x = np.arange(len(DATASETS))
    w = 0.14
    offsets = np.linspace(-(len(CP_MODELS)-1)/2, (len(CP_MODELS)-1)/2,
                           len(CP_MODELS)) * w

    for i, m in enumerate(CP_MODELS):
        vals = apss[m].values
        ax.bar(x + offsets[i], vals, width=w*0.9,
               color=PALETTE[m], label=MODEL_LABELS[m],
               edgecolor='white', linewidth=0.4, zorder=3,
               hatch='//' if m == 'fc_gnn' else '')

    ax.axhline(1.0, color='gray', lw=1, ls=':', zorder=2, label='APSS = 1 (singleton)')
    ax.set_xticks(x)
    ax.set_xticklabels(DS_SHORT)
    ax.set_ylabel('Avg. Prediction Set Size (APSS)')
    ax.legend(ncol=3, loc='upper right', framealpha=0.9)
    ax.set_title('Average Prediction Set Size (APSS) — smaller is more efficient (α = 0.10)')
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # Annotate CTU-13 RR-GNN outlier
    rrgnn_ctu = apss.loc['CTU-13', 'rrgnn']
    if not np.isnan(rrgnn_ctu) and rrgnn_ctu > 3:
        ctu_idx = DATASETS.index('CTU-13')
        rr_off = offsets[CP_MODELS.index('rrgnn')]
        ax.annotate(f'{rrgnn_ctu:.2f}',
                    xy=(x[ctu_idx] + rr_off, rrgnn_ctu),
                    xytext=(x[ctu_idx] + rr_off + 0.3, rrgnn_ctu + 0.2),
                    arrowprops=dict(arrowstyle='->', color='#c62828'),
                    fontsize=8, color='#c62828')

    fig.savefig(OUT / 'apss_comparison.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ apss_comparison.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 4 – Macro-F1 comparison (all models)
# ════════════════════════════════════════════════════════════════════════════
def fig_f1():
    f1 = pivot('Macro-F1', ALL_MODELS)
    fig, ax = plt.subplots(figsize=(12, 4.5))

    x = np.arange(len(DATASETS))
    w = 0.085
    n = len(ALL_MODELS)
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * w

    for i, m in enumerate(ALL_MODELS):
        vals = f1[m].values
        ax.bar(x + offsets[i], vals, width=w*0.9,
               color=PALETTE[m], label=MODEL_LABELS[m],
               edgecolor='white', linewidth=0.3, zorder=3,
               hatch='//' if m == 'fc_gnn' else '')

    ax.set_xticks(x)
    ax.set_xticklabels(DS_SHORT)
    ax.set_ylabel('Macro-F1')
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=5, loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_title('Macro-F1 Comparison across All Models and Datasets')
    ax.grid(axis='y', alpha=0.3, zorder=0)

    fig.savefig(OUT / 'f1_comparison.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ f1_comparison.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 5 – Coverage vs APSS scatter (efficiency frontier)
# ════════════════════════════════════════════════════════════════════════════
def fig_efficiency_frontier():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    sample_ds = ['UNSW-NB15', 'CTU-13', 'ToN-IoT']

    for ax, ds in zip(axes, sample_ds):
        ds_df = df[(df['Dataset'] == ds) & df['Coverage'].notna() & df['APSS'].notna()]
        for _, row in ds_df.iterrows():
            m = row['Model']
            col = PALETTE.get(m, '#888')
            marker = '*' if m == 'fc_gnn' else 'o'
            ms = 180 if m == 'fc_gnn' else 80
            ax.scatter(row['APSS'], row['Coverage'],
                       c=col, s=ms, marker=marker, zorder=3,
                       edgecolors='white', linewidths=0.5)
            ax.annotate(MODEL_LABELS.get(m, m), (row['APSS'], row['Coverage']),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=7, color='#333')

        ax.axhline(0.90, color='#c62828', lw=1.2, ls='--', alpha=0.7)
        ax.set_xlabel('APSS (↓ better)')
        ax.set_ylabel('Empirical Coverage (↑ better)')
        ax.set_title(ds)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.70, 1.02)

    # legend
    handles = [plt.scatter([], [], c=PALETTE[m], s=80 if m != 'fc_gnn' else 180,
                           marker='*' if m == 'fc_gnn' else 'o',
                           label=MODEL_LABELS[m])
               for m in CP_MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02),
               fontsize=8, framealpha=0.9)
    fig.suptitle('Coverage–Efficiency Trade-off (α = 0.10, ★ = FC-GNN)',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()

    fig.savefig(OUT / 'efficiency_frontier.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ efficiency_frontier.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 6 – Coverage Gap (conditional coverage deviation)
# ════════════════════════════════════════════════════════════════════════════
def fig_coverage_gap():
    # Only Mondrian models have non-trivial coverage gap
    mondrian = ['fc_gnn', 'rrgnn']
    gap = pivot('Cov-Gap', mondrian)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(DATASETS))
    w = 0.28

    for i, m in enumerate(mondrian):
        offset = (i - 0.5) * w
        vals = gap[m].values
        ax.bar(x + offset, vals, width=w*0.9,
               color=PALETTE[m], label=MODEL_LABELS[m],
               edgecolor='white', linewidth=0.5, zorder=3,
               hatch='//' if m == 'fc_gnn' else '')

    ax.set_xticks(x)
    ax.set_xticklabels(DS_SHORT)
    ax.set_ylabel('Coverage Gap  $\\max_m|\\hat p_m - (1-\\alpha)|$')
    ax.set_ylim(0, 0.25)
    ax.legend()
    ax.set_title('Per-Community Coverage Gap (Mondrian CP models, lower is better)')
    ax.grid(axis='y', alpha=0.3, zorder=0)

    fig.savefig(OUT / 'coverage_gap.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ coverage_gap.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 7 – Interpretability metrics (FC-GNN only)
# ════════════════════════════════════════════════════════════════════════════
def fig_interpretability():
    fc = df[df['Model'] == 'fc_gnn'][['Dataset', 'Rule-Fidelity', 'Expl-Stability',
                                       'Macro-F1']].copy()
    fc = fc.set_index('Dataset').reindex(DATASETS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Rule Fidelity
    ax = axes[0]
    vals = fc['Rule-Fidelity'].values
    colors = [PALETTE['fc_gnn'] if v >= 0.3 else '#adb5bd' for v in vals]
    ax.barh(DS_SHORT, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(np.nanmean(vals), color='#c62828', lw=1.5, ls='--',
               label=f'Mean = {np.nanmean(vals):.3f}')
    ax.set_xlabel('Rule Fidelity')
    ax.set_title('Rule Fidelity\n(fraction of nodes where top-1 rule matches class)')
    ax.set_xlim(0, 0.65)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # Explanation Stability
    ax = axes[1]
    vals2 = fc['Expl-Stability'].values
    colors2 = [PALETTE['fc_gnn'] if v >= 0.5 else '#adb5bd' for v in vals2]
    ax.barh(DS_SHORT, vals2, color=colors2, edgecolor='white', linewidth=0.5)
    ax.axvline(np.nanmean(vals2), color='#c62828', lw=1.5, ls='--',
               label=f'Mean = {np.nanmean(vals2):.3f}')
    ax.set_xlabel('Explanation Stability (Jaccard)')
    ax.set_title('Explanation Stability\n(Jaccard sim. of fired rules for similar inputs)')
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    fig.suptitle('FC-GNN Interpretability Metrics', fontsize=12, fontweight='bold')
    fig.tight_layout()

    fig.savefig(OUT / 'interpretability.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ interpretability.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 8 – Radar / spider chart: overall comparison
# ════════════════════════════════════════════════════════════════════════════
def fig_radar():
    # Average across datasets where all models have data
    metrics_radar = ['Macro-F1', 'Coverage', 'SHP', 'Accuracy']
    metric_labels = ['Macro-F1', 'Coverage', 'SHP\n(Singleton Hit)', 'Accuracy']

    radar_models = ['fc_gnn', 'cfgnn', 'rrgnn', 'daps', 'snaps']
    n_metrics = len(metrics_radar)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for m in radar_models:
        m_df = df[(df['Model'] == m)]
        vals = []
        for metric in metrics_radar:
            v = m_df[metric].dropna()
            vals.append(float(v.mean()) if len(v) else 0.0)
        vals += vals[:1]
        lw = 2.5 if m == 'fc_gnn' else 1.4
        ls = '-' if m == 'fc_gnn' else '--'
        ax.plot(angles, vals, lw=lw, ls=ls, color=PALETTE[m],
                label=MODEL_LABELS[m])
        ax.fill(angles, vals, alpha=0.07 if m != 'fc_gnn' else 0.15,
                color=PALETTE[m])

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
    ax.grid(alpha=0.4)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=9)
    ax.set_title('Multi-Metric Comparison\n(average across 7 datasets)',
                 fontweight='bold', y=1.12, fontsize=11)

    fig.savefig(OUT / 'radar_comparison.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ radar_comparison.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 9 – Heatmap: Macro-F1 across datasets and models
# ════════════════════════════════════════════════════════════════════════════
def fig_heatmap_f1():
    mat = []
    for ds in DATASETS:
        row = []
        for m in ALL_MODELS:
            v = df[(df['Dataset'] == ds) & (df['Model'] == m)]['Macro-F1']
            row.append(float(v.values[0]) if len(v) and not pd.isna(v.values[0]) else np.nan)
        mat.append(row)
    mat = np.array(mat)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(ALL_MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS], rotation=30, ha='right')
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS)

    for i in range(len(DATASETS)):
        for j in range(len(ALL_MODELS)):
            v = mat[i, j]
            if not np.isnan(v):
                color = 'white' if v < 0.4 or v > 0.85 else 'black'
                weight = 'bold' if ALL_MODELS[j] == 'fc_gnn' else 'normal'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight=weight)

    # Highlight FC-GNN column
    fc_col = ALL_MODELS.index('fc_gnn')
    for i in range(len(DATASETS)):
        rect = plt.Rectangle((fc_col - 0.5, i - 0.5), 1, 1,
                              fill=False, edgecolor='#E63946', lw=2.0)
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label='Macro-F1')
    ax.set_title('Macro-F1 Heatmap (■ = FC-GNN column)', fontsize=12, fontweight='bold')
    fig.tight_layout()

    fig.savefig(OUT / 'heatmap_f1.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ heatmap_f1.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 10 – CP metrics summary table as figure
# ════════════════════════════════════════════════════════════════════════════
def fig_cp_summary():
    metrics = ['Coverage', 'APSS', 'SHP', 'Cov-Gap']
    m_labels = ['Coverage', 'APSS', 'SHP', 'Cov-Gap']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric, mlbl in zip(axes, metrics, m_labels):
        data = pivot(metric, CP_MODELS)
        x = np.arange(len(DATASETS))
        w = 0.14
        offsets = np.linspace(-(len(CP_MODELS)-1)/2,
                               (len(CP_MODELS)-1)/2,
                               len(CP_MODELS)) * w
        for i, m in enumerate(CP_MODELS):
            vals = data[m].values
            ax.bar(x + offsets[i], vals, width=w*0.9,
                   color=PALETTE[m], label=MODEL_LABELS[m],
                   edgecolor='white', linewidth=0.4, zorder=3,
                   hatch='//' if m == 'fc_gnn' else '')

        if metric == 'Coverage':
            ax.axhline(0.90, color='#c62828', lw=1.2, ls='--', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(DS_SHORT, fontsize=9)
        ax.set_ylabel(mlbl)
        ax.set_title(f'{mlbl} across datasets', fontsize=10)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        if ax == axes[0]:
            ax.legend(ncol=2, fontsize=8, framealpha=0.9)

    fig.suptitle('Conformal Prediction Quality Metrics (α = 0.10)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    fig.savefig(OUT / 'cp_metrics_summary.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  ✓ cp_metrics_summary.png")

# ════════════════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating manuscript figures...")
    fig_architecture()
    fig_coverage()
    fig_apss()
    fig_f1()
    fig_efficiency_frontier()
    fig_coverage_gap()
    fig_interpretability()
    fig_radar()
    fig_heatmap_f1()
    fig_cp_summary()
    print(f"\nAll figures saved to {OUT}/")
