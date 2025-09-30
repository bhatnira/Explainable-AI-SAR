# SAR and Parameter Sweep Summary

This report summarizes the fast sweep over circular fingerprint parameters and the subsequent SAR analysis.

## Best configuration (fast sweep)
- radius: 1
- nBits: 4096
- useFeatures: 0
- useChirality: 0
- AUC: ~0.778
- Collision rate: ~0.0475
- Avg fragments per active bit: ~1.050
- Best interpretation: ENSEMBLE (alpha=0.5)
- Faithfulness (higher is better):
  - SHAP: ~0.292
  - LIME: ~0.200
  - ENSEMBLE: ~0.305

Files:
- `analysis/sar/results/fast_sweep_results.csv` — all configs with metrics
- `analysis/sar/results/fast_sweep_topK_details.json` — Top-K deep dive
- `analysis/sar/results/fast_sweep_best.json` — best config summary

## Figures
- Heatmaps: AUC, collision, and objective by (radius × nBits) split by Feature/Chiral flags
- Scatter: AUC vs Collision (with and without best point highlighted)
- Top-K faithfulness comparison across SHAP/LIME/ENSEMBLE
- SAR run figures (confusion matrix, top fragments by SHAP, collision sensitivity, stability, masking faithfulness)

All figures are saved in `analysis/sar/figures` as TIFF/PNG/PDF at 1200 dpi.

## Recommendations
- Prefer nBits 4096–8192 at small radii; 4096 performed best in the latest sweep while keeping collisions acceptable.
- Start with radius=1–2; keep chirality off by default (latest sweep favored USE_CHIRALITY=0). Add features/chirality only if domain knowledge suggests benefit.
- Use ENSEMBLE interpretation (alpha=0.5) for atom-level highlighting and faithfulness.
- For thorough runs, increase N_PER_CLASS and re-run `sar-fast-sweep` and `sar` to validate stability.

## Reproduce
- Fast sweep: `docker compose run --rm sar-fast-sweep`
- SAR with best config:
  - `docker compose run --rm -e NBITS=4096 -e RADIUS=1 -e USECHIRALITY=0 -e USEFEATURES=0 -e N_PER_CLASS=50 -e ENSEMBLE_ALPHA=0.5 sar`

## Scaffold-split evaluation (multi-seed)
- Config: NBITS=8192, RADIUS=1, USE_FEATURES=0, USE_CHIRALITY=1, TEST_FRAC=0.2
- Replicates: 10 seeds (BASE_SEED=123)
- Performance (mean ± 95% CI):
  - acc: 0.679 ± 0.060
  - auc: 0.770 ± 0.045
- Stability (top-k bits):
  - SHAP: 0.397 ± 0.043
  - LIME: 0.756 ± 0.031
  - ENSEMBLE: 0.674 ± 0.024
- Faithfulness AUCs:
  - SHAP: deletion 0.399 ± 0.031, insertion 0.631 ± 0.037
  - LIME: deletion 0.384 ± 0.029, insertion 0.541 ± 0.036
  - ENSEMBLE: deletion 0.402 ± 0.030, insertion 0.627 ± 0.038

Files:
- `analysis/sar/results/scaffold_split_replicates.json` (aggregates with CI)
- `analysis/sar/results/scaffold_split_replicates.csv` (per-seed)
- Figures: `analysis/sar/figures/scaffold_split_confusion_matrix.png`, `scaffold_split_deletion_curves.png`, `scaffold_split_insertion_curves.png`

Reproduce:
- `docker compose run --rm sar-scaffold-split` (REPEATS and BASE_SEED configurable via env)

## MMP and Motif Enrichment
- Eval split performance: acc≈0.639, auc≈0.775
- Motif enrichment (odds ratios; illustrative top signals):
  - Halogen OR≈2.61 (p≈0.289)
  - Amide OR≈2.30 (p≈0.441)
  - HBDonor OR≈1.99 (p≈0.488)
  - Nitro OR≈0.57 (depleted)
  - AromaticRing OR≈1.00 (neutral)
- Attribution–motif overlap: see `motif_overlap_violin.png`
- MMP consistency: no eligible scaffold-matched opposite-label pairs in the eval subset (n_pairs=0); increase subset or relax pairing to populate.

Files:
- `analysis/sar/results/mmp_motif_summary.json`
- Figures: `analysis/sar/figures/motif_enrichment_bar.png`, `motif_overlap_violin.png`, `mmp_consistency_bar.png`

Reproduce:
- `docker compose run --rm sar-mmp-motif`

## Performance vs Explanation Quality
- Dataset-level correlations (fast sweep grid):
  - AUC vs collision rate: Pearson r≈0.109 (p≈0.082), Spearman r≈0.006 (p≈0.922), n=256 → weak/none.
  - AUC vs faithfulness (subset with Top-K details, n=8):
    - ENSEMBLE: Pearson r≈-0.044 (p≈0.918), Spearman r≈0.082 (p≈0.846)
    - SHAP: Pearson r≈-0.014 (p≈0.973), Spearman r≈0.082 (p≈0.846)
    - LIME: Pearson r≈-0.129 (p≈0.761), Spearman r≈0.082 (p≈0.846)
- Interpretation: Across this sweep, higher predictive AUC did not systematically align with higher faithfulness or lower collisions. Explanation quality appears decoupled from small AUC differences at this scale.

Files:
- Summary JSON: `analysis/sar/results/perf_vs_explanation_summary.json`
- Figures: `analysis/sar/figures/auc_vs_collision_scatter.*`, `auc_vs_faithfulness_ENSEMBLE.*`, `auc_vs_faithfulness_SHAP.*`

## Next steps
- Increase EXPLAIN_SUBSET and MAX_MMP_PAIRS to populate MMP pairs.
- Optionally run larger N_PER_CLASS and re-run replicates for tighter CIs.

## NeurIPS submission upgrade plan (prioritized)

- Positioning and theory
  - Formal objective: define J = AUC − λ·collision − μ·instability; derive bounds linking collision rate to attribution sparsity and faithfulness degradation.
  - Stability theory: analyze attribution variance across seeds/splits; connect ensemble α to bias–variance trade-off.
  - Causal framing: relate deletion/insertion to ROAR/KAR; add counterfactual validity as a criterion.

- Method extensions
  - Adaptive ensemble: learn per-sample α via uncertainty/meta-features; add calibration-aware attribution scaling.
  - Pareto/Bayes optimization: replace grid with multi-objective BO over (radius, nBits, features, chirality) optimizing (AUC, collisions, faithfulness, stability).
  - Counterfactuals: generate MMP/SELFIES edits guided by attributions to test causal impact and suggest actionable changes.

- Broader benchmarks
  - Datasets: add BBBP, BACE, HIV, Tox21, ClinTox, SIDER with scaffold/time/cluster splits.
  - Models: LR (current) vs RF/XGBoost vs GNNs (GCN/GAT/GIN) with explainers (Integrated Gradients, Grad-CAM, GNNExplainer, PGExplainer); include kernel/LinearSHAP and Anchors.

- Evaluation upgrades
  - Faithfulness: ROAR/KAR (added), Pointing Game, deletion/insertion AUCs; report per-dataset CIs.
  - Sanity checks: Adebayo parameter and data randomization tests for explanations.
  - Uncertainty: calibration (ECE/Brier), predictive entropy; overlay attribution strength with uncertainty.
  - Motif/MMP rigor: BH/FDR multiple-testing correction; power analysis; increase MAX_MMP_PAIRS and report coverage.
  - Human eval: small blinded study with medicinal chemists (20–30 pairs), inter-rater agreement and preferences.

- Ablations and diagnostics
  - α sweep and adaptive-α vs fixed; features/chirality on/off; nBits/radius sensitivity; collision-penalty λ sensitivity.
  - Fingerprints: ECFP vs FCFP, pharmacophore, topological torsion; learned neural fingerprints as baseline.
  - Collision handling: alternative folding/hash schemes; canonical environment tie-breakers.

- Scalability and engineering
  - Vectorize RDKit environment hashing; cache bitInfo; parallelize with joblib/Dask; wall-time vs size scaling plots.
  - GPU acceleration for GNN baselines; profile hotspots; include memory footprints.

- Reproducibility and artifacts
  - Deterministic seeds, fixed splits; release JSON splits/configs for all datasets.
  - One-click repro: docker compose targets per dataset/model; Makefile targets; CI smoke tests.
  - Package: minimal pip package + CLI (sweep, explain, scaffold, mmp).
  - Artifact evaluation checklist; Zenodo DOI; model cards/datasheets.

- Writing and figures
  - Clear contributions, limitations, broader impacts.
  - Schematic of collision-aware pipeline; Pareto fronts; dataset-wide attribution maps; sanity-check and counterfactual examples.
  - Ensure all figures via `fig_style.py` at 1200 dpi with vector PDFs.

- Low-effort fixes now
  - Remove compose version warning; migrate to RDKit MorganGenerator to silence deprecation.
  - Add BH correction and adjusted p-values in `mmp_motif_analysis.py`.
  - ROAR/KAR already added in `sar_analysis.py`; expose λ, μ via CLI/env.
  - Add datasets/scripts; extend this report with multi-dataset tables.

