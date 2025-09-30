# Fragment-level SAR and Attribution Validation

This folder contains scripts, data artifacts, and figures for Structure–Activity Relationship (SAR) analysis using fragment-level (Morgan bit) attributions.

Contents
- sar_analysis.py: Main pipeline to compute fragment-level SAR, repeat stability, collision sensitivity, enrichment vs activity, global rankings, clustering, masking tests, cross-method agreement, and quantitative agreement metrics. Saves results to `results/` and figures to `figures/` in this folder.
- results/: JSON/CSV artifacts for publication.
- figures/: PNG/PDF charts for publication.

How to run
```
python analysis/sar/sar_analysis.py
```
Optional env vars
- N_PER_CLASS: subsample size per class (default 50)
- N_BOOTSTRAP: bootstrap runs for stability (default 5)
- NBITS: fingerprint size for SAR (default 2048)
- RADIUS: Morgan radius (default 2)
- USE_TPOT: use TPOT if available (default 0)

Outputs (examples)
- results/fragment_stats.json
- results/fragment_rankings.json
- results/stability.json
- results/method_agreement.json
- figures/top_fragments_shap.png
- figures/stability_jaccard.png
- figures/rank_correlation.png
- figures/collision_vs_weight.png
