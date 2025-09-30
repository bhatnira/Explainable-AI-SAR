#!/usr/bin/env python3
"""
Plot figures from fast_sweep results:
- Heatmaps (AUC, Collision, Objective) per (useFeatures,useChirality)
- Scatter AUC vs Collision with best config annotation
- Top-K faithfulness comparison (SHAP/LIME/ENSEMBLE)
Outputs saved under analysis/sar/figures
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root for absolute imports like analysis.sar.fig_style
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.sar.fig_style import apply_nature_style, colsize, save_figure

apply_nature_style()

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
FIG = HERE / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

# Load results
csv_path = RES / 'fast_sweep_results.csv'
if not csv_path.exists():
    raise SystemExit(f"Missing {csv_path}. Run sar-fast-sweep first.")

df = pd.read_csv(csv_path)

# Ensure types
for col in ['radius', 'nBits', 'useFeatures', 'useChirality']:
    df[col] = df[col].astype(int)

# Helper to draw heatmaps by (radius x nBits)
def heatmap_metric(metric: str, title_prefix: str):
    for f in sorted(df.useFeatures.unique()):
        for c in sorted(df.useChirality.unique()):
            d = df[(df.useFeatures == f) & (df.useChirality == c)].copy()
            pivot = d.pivot_table(index='radius', columns='nBits', values=metric, aggfunc='mean')
            fig, ax = plt.subplots(figsize=colsize(single=False))
            sns.heatmap(pivot, annot=False, cmap='viridis', ax=ax)
            ax.set_title(f"{title_prefix} (features={f}, chiral={c})")
            ax.set_ylabel('radius')
            ax.set_xlabel('nBits')
            out = FIG / f"{metric}_heatmap_feat{f}_chiral{c}.png"
            save_figure(fig, out, dpi=1200, formats=("tiff","png","pdf"))

# Draw heatmaps
heatmap_metric('auc', 'AUC by (radius, nBits)')
heatmap_metric('collision_rate', 'Collision rate by (radius, nBits)')
heatmap_metric('objective', 'Objective by (radius, nBits)')

# Scatter: AUC vs collision
fig, ax = plt.subplots(figsize=colsize(single=True))
scatter = ax.scatter(df['collision_rate'], df['auc'], c=df['nBits'], cmap='plasma', alpha=0.6, s=12)
cb = fig.colorbar(scatter, ax=ax)
cb.set_label('nBits')
ax.set_xlabel('Collision rate')
ax.set_ylabel('AUC')
ax.set_title('AUC vs Collision (color=nBits)')
save_figure(fig, FIG / 'auc_vs_collision_scatter.png', dpi=1200, formats=("tiff","png","pdf"))

# Annotate best from fast_sweep_best.json if present
best_path = RES / 'fast_sweep_best.json'
if best_path.exists():
    best = json.loads(best_path.read_text())
    # Highlight point
    b = df[(df.radius == best['radius']) & (df.nBits == best['nBits']) & (df.useFeatures == best['useFeatures']) & (df.useChirality == best['useChirality'])]
    if not b.empty:
        x = float(b['collision_rate'].iloc[0]); y = float(b['auc'].iloc[0])
        fig, ax = plt.subplots(figsize=colsize(single=True))
        scatter = ax.scatter(df['collision_rate'], df['auc'], c=df['nBits'], cmap='plasma', alpha=0.35, s=10)
        fig.colorbar(scatter, ax=ax, label='nBits')
        ax.scatter([x], [y], c='red', s=30, edgecolors='k', label='Best')
        ax.set_xlabel('Collision rate')
        ax.set_ylabel('AUC')
        ax.legend()
        ax.set_title('Best config highlighted')
        save_figure(fig, FIG / 'auc_vs_collision_best.png', dpi=1200, formats=("tiff","png","pdf"))

# Top-K faithfulness bars
k_path = RES / 'fast_sweep_topK_details.json'
if k_path.exists():
    top = json.loads(k_path.read_text())
    if isinstance(top, dict):
        # If somehow saved as dict, wrap
        top = [top]
    # Aggregate mean faithfulness per interpretation
    shap_vals = [t.get('faithfulness_SHAP', 0.0) for t in top]
    lime_vals = [t.get('faithfulness_LIME', 0.0) for t in top]
    ens_vals = [t.get('faithfulness_ENSEMBLE', 0.0) for t in top]
    means = {
        'SHAP': float(np.mean(shap_vals)) if shap_vals else 0.0,
        'LIME': float(np.mean(lime_vals)) if lime_vals else 0.0,
        'ENSEMBLE': float(np.mean(ens_vals)) if ens_vals else 0.0,
    }
    fig, ax = plt.subplots(figsize=colsize(single=True))
    sns.barplot(x=list(means.keys()), y=list(means.values()), palette=['#4ECDC4', '#FFBE0B', '#3A86FF'], ax=ax)
    ax.set_ylabel('Mean faithfulness (Top-K)')
    ax.set_title('Faithfulness across Top-K configs')
    save_figure(fig, FIG / 'topK_faithfulness_bar.png', dpi=1200, formats=("tiff","png","pdf"))

print('Figures written to', FIG)
