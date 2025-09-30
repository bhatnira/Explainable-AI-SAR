#!/usr/bin/env python3
"""
Performance vs Explanation Quality (SAR) Analysis
- Correlate model performance (AUC) with explanation quality metrics and interpretability proxies.
- Inputs: fast_sweep_results.csv, fast_sweep_topK_details.json
- Outputs: figures (1200 dpi) and a JSON summary of correlations
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys

# Add project root and apply Nature-style figures
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.sar.fig_style import apply_nature_style, save_figure
apply_nature_style()

HERE = Path(__file__).resolve().parent
RES_DIR = HERE / 'results'
FIG_DIR = HERE / 'figures'
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load sweep metrics
sweep_csv = RES_DIR / 'fast_sweep_results.csv'
topk_json = RES_DIR / 'fast_sweep_topK_details.json'

if not sweep_csv.exists():
    raise SystemExit(f"Missing sweep CSV: {sweep_csv}")

sweep = pd.read_csv(sweep_csv)
# Ensure boolean flags are consistent types for merging
for col in ['useFeatures', 'useChirality']:
    if col in sweep.columns:
        sweep[col] = sweep[col].astype(int)

faith_df = None
if topk_json.exists():
    with open(topk_json, 'r') as f:
        topk = json.load(f)
    faith_df = pd.DataFrame(topk)
    for col in ['useFeatures', 'useChirality']:
        if col in faith_df.columns:
            faith_df[col] = faith_df[col].astype(int)
    # Merge on the identifying keys
    merged = pd.merge(
        faith_df,
        sweep,
        on=['radius', 'nBits', 'useFeatures', 'useChirality', 'auc'],
        how='left',
        suffixes=("", "_sweep")
    )
else:
    merged = sweep.copy()

summary = {}

# AUC vs Collision rate (all configs)
if 'collision_rate' in sweep.columns:
    r_p, p_p = stats.pearsonr(sweep['auc'], sweep['collision_rate'])
    r_s, p_s = stats.spearmanr(sweep['auc'], sweep['collision_rate'])
    summary['auc_vs_collision'] = {
        'pearson_r': float(r_p), 'pearson_p': float(p_p),
        'spearman_r': float(r_s), 'spearman_p': float(p_s),
        'n': int(len(sweep))
    }

# AUC vs faithfulness metrics (subset with faithfulness)
if faith_df is not None and not faith_df.empty:
    def corr_block(df, x_col, y_col):
        r_p, p_p = stats.pearsonr(df[x_col], df[y_col])
        r_s, p_s = stats.spearmanr(df[x_col], df[y_col])
        return {
            'pearson_r': float(r_p), 'pearson_p': float(p_p),
            'spearman_r': float(r_s), 'spearman_p': float(p_s),
            'n': int(len(df))
        }
    summary['auc_vs_faithfulness_ENSEMBLE'] = corr_block(merged, 'auc', 'faithfulness_ENSEMBLE')
    summary['auc_vs_faithfulness_SHAP'] = corr_block(merged, 'auc', 'faithfulness_SHAP')
    summary['auc_vs_faithfulness_LIME'] = corr_block(merged, 'auc', 'faithfulness_LIME')

# Save summary JSON
with open(RES_DIR / 'perf_vs_explanation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Figures
sns.set_style('white')

# 1. AUC vs Collision (all)
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.scatter(sweep['collision_rate'], sweep['auc'], s=18, alpha=0.7, color='#4ECDC4', edgecolor='none')
# Trend line
x = sweep['collision_rate'].values.reshape(-1, 1)
y = sweep['auc'].values
try:
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x, y)
    xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    yy = reg.predict(xx)
    ax.plot(xx, yy, '--', color='gray', linewidth=1)
except Exception:
    pass
ax.set_xlabel('Collision rate (lower is better)')
ax.set_ylabel('AUC')
ax.set_title('Performance vs Collisions (all configs)')
save_figure(fig, FIG_DIR / 'auc_vs_collision_scatter.png', dpi=1200, formats=("tiff","png","pdf"))

# 2–3. AUC vs Faithfulness (subset)
if faith_df is not None and not faith_df.empty:
    for m, color in [('faithfulness_ENSEMBLE', '#FF6B6B'), ('faithfulness_SHAP', '#45B7D1')]:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        ax.scatter(merged[m], merged['auc'], s=24, alpha=0.8, color=color, edgecolor='none')
        # Trend
        x = merged[m].values.reshape(-1, 1)
        y = merged['auc'].values
        try:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(x, y)
            xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            yy = reg.predict(xx)
            ax.plot(xx, yy, '--', color='gray', linewidth=1)
        except Exception:
            pass
        ax.set_xlabel(m.replace('_', ' '))
        ax.set_ylabel('AUC')
        ax.set_title(f'Performance vs {m.split("_")[-1]} (subset)')
        save_figure(fig, FIG_DIR / f'auc_vs_{m}.png', dpi=1200, formats=("tiff","png","pdf"))

print("Correlation summary written to:", RES_DIR / 'perf_vs_explanation_summary.json')
print("Figures written to:", FIG_DIR)
