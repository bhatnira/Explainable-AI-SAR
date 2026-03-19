# NeurIPS Paper: Implementation Quick Start

**Document**: Implementation Guide & Code Structure  
**Status**: Ready for development  
**Target**: Complete by June 2026

---

## QUICK START

### 1. Run This First (5 min)
```bash
cd /home/nbhatta1/Documents/Explainable-AI-SAR

# Check current state
ls -la models/
ls -la visualization/
ls -la data/

# Verify environment
python -c "import torch, rdkit, sklearn; print('✅ Dependencies OK')"
```

### 2. Create Core Framework (Week 1)
```bash
# Create new deterministic framework
python models/deterministic_sar_framework.py

# Create SAR extraction utilities
python models/sar_extraction_utils.py

# Create benchmark loader
python scripts/benchmark_loader.py
```

### 3. Run Experiments (Week 2-3)
```bash
# Run full pipeline on all benchmarks
python models/run_full_benchmark.py

# Generate results & figures
python models/generate_results.py

# Validate SAR against ground truth
python models/validate_sar_against_literature.py
```

### 4. Write Paper (Week 4-6)
- Edit: `NEURIPS_PAPER_OUTLINE.md` (this file structure)
- Output: `NEURIPS_SUBMISSION.pdf`

---

## FILE STRUCTURE TO CREATE

```
/home/nbhatta1/Documents/Explainable-AI-SAR/
├── models/
│   ├── deterministic_sar_framework.py      [NEW - Core metrics]
│   ├── sar_extraction_utils.py             [NEW - SAR extraction]
│   ├── deterministic_sar_theory.py         [NEW - Theorem proofs]
│   ├── theorem_validators.py               [NEW - Test theorems]
│   ├── agentic_parameter_optimizer.py      [EXISTING - Modify]
│   ├── optimize_for_explainability.py      [EXISTING - Use]
│   └── explanation_quality_metrics.py      [EXISTING - Extend]
│
├── scripts/
│   ├── benchmark_loader.py                 [NEW - Load BBBP, Tox21, etc]
│   ├── run_full_benchmark.py               [NEW - Main experiment]
│   ├── validate_sar_against_literature.py  [NEW - SAR validation]
│   └── generate_paper_figures.py           [NEW - Plots]
│
├── results/
│   ├── benchmark_results.json              [Generate]
│   ├── sar_validation_results.json         [Generate]
│   └── figures/                            [Generate]
│
├── paper/
│   ├── NEURIPS_PAPER_OUTLINE.md            [THIS FILE - Structure]
│   ├── NEURIPS_SUBMISSION.tex              [Your LaTeX]
│   └── appendix_proofs.tex                 [Math proofs]
│
└── REPRODUCIBILITY.md                      [Seeding guide]
```

---

## PHASE 1: DETERMINISTIC FRAMEWORK (Week 1)

### File: `models/deterministic_sar_framework.py`

```python
"""
Core framework for deterministic SAR learning.
Implements 5 explanation quality metrics without ground truth.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import os
import random
import torch

# ============================================
# SEEDING UTILITIES
# ============================================

def set_all_seeds(seed=42):
    """
    Set all random seeds for deterministic behavior.
    Call this at the START of every script.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(f"🔒 All random seeds set to {seed}")

# ============================================
# METRIC 1: STABILITY
# ============================================

def compute_stability(explanations_list, metric='cosine'):
    """
    Measure consistency across multiple runs:
    Same molecule → same explanations?
    
    Args:
        explanations_list: List of N explanation arrays
        metric: 'cosine' or 'euclidean'
    
    Returns:
        float: Stability score in [0, 1]
        - 1.0 = Perfect determinism
        - 0.5 = Moderate stochasticity
        - 0.0 = Completely random
    """
    if len(explanations_list) < 2:
        return np.nan
    
    # Convert to array
    expl_arrays = [np.array(e).flatten() for e in explanations_list]
    
    # Compute pairwise distances
    distances = []
    for i in range(len(expl_arrays)):
        for j in range(i+1, len(expl_arrays)):
            if metric == 'cosine':
                # Cosine distance
                a, b = expl_arrays[i], expl_arrays[j]
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                if norm > 0:
                    dist = 1 - (np.dot(a, b) / norm)
                else:
                    dist = 0
                distances.append(dist)
            elif metric == 'euclidean':
                dist = np.linalg.norm(expl_arrays[i] - expl_arrays[j])
                distances.append(dist)
    
    # Stability = 1 - mean_distance
    mean_distance = np.mean(distances) if distances else 0
    stability = 1 - min(mean_distance, 1.0)  # Clip to [0, 1]
    
    return stability

# ============================================
# METRIC 2: CONSISTENCY
# ============================================

def compute_consistency(smiles_list, explanations, similarity_metric='tanimoto'):
    """
    Measure explanation consistency for similar molecules:
    Similar molecules → similar explanations?
    
    Args:
        smiles_list: List of SMILES strings
        explanations: Array of atom importance scores (n_mols, n_atoms_max)
        similarity_metric: 'tanimoto' for Morgan fingerprints
    
    Returns:
        float: Consistency correlation in [-1, 1]
        - 1.0 = Perfect consistency
        - 0.5 = Moderate
        - 0.0 = No correlation
    """
    n = len(smiles_list)
    if n < 2:
        return np.nan
    
    # Compute pairwise molecular similarities
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) 
           for m in mols if m is not None]
    
    if len(fps) < 2:
        return np.nan
    
    mol_similarities = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            mol_similarities.append(sim)
    
    # Compute pairwise explanation similarities (cosine)
    expl_similarities = []
    for i in range(len(explanations)):
        for j in range(i+1, len(explanations)):
            a, b = explanations[i], explanations[j]
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm > 0:
                sim = np.dot(a, b) / norm
            else:
                sim = 0
            expl_similarities.append(sim)
    
    # Correlation between molecular and explanation similarities
    if len(mol_similarities) < 2:
        return np.nan
    
    consistency, _ = pearsonr(mol_similarities, expl_similarities)
    return float(consistency)

# ============================================
# METRIC 3: FAITHFULNESS
# ============================================

def compute_faithfulness(model, X_test, y_test, explanations, topk=5):
    """
    Measure if important atoms actually impact predictions:
    Mask top-K atoms → prediction change?
    Compare to random masking
    
    Args:
        model: Trained model
        X_test: Test features (n_samples, n_features)
        y_test: Labels
        explanations: Atom importance scores (n_features,)
        topk: Number of top atoms to mask
    
    Returns:
        float: Faithfulness ratio ≥ 1.0
        - >2.0 = Highly faithful
        - 1.5-2.0 = Faithful
        - 1.0-1.5 = Weakly faithful
        - <1.0 = Unfaithful (rare)
    """
    # Get baseline predictions
    baseline_probs = model.predict_proba(X_test)[:, 1]
    
    # Find top-K important atoms
    top_indices = np.argsort(-np.abs(explanations))[:topk]
    
    # Mask top atoms
    X_masked_top = X_test.copy()
    X_masked_top[:, top_indices] = 0
    masked_top_probs = model.predict_proba(X_masked_top)[:, 1]
    
    # Prediction change for top atoms
    delta_top = np.abs(baseline_probs - masked_top_probs)
    mean_delta_top = np.mean(delta_top)
    
    # Random masking (baseline)
    random_indices = np.random.choice(X_test.shape[1], topk, replace=False)
    X_masked_random = X_test.copy()
    X_masked_random[:, random_indices] = 0
    masked_random_probs = model.predict_proba(X_masked_random)[:, 1]
    
    delta_random = np.abs(baseline_probs - masked_random_probs)
    mean_delta_random = np.mean(delta_random)
    
    # Faithfulness = ratio
    faithfulness = (mean_delta_top + 1e-8) / (mean_delta_random + 1e-8)
    return float(faithfulness)

# ============================================
# METRIC 4: PARSIMONY
# ============================================

def compute_parsimony(explanations, threshold=0.1):
    """
    Measure if explanation is concise:
    How many atoms truly matter vs. noise?
    
    Args:
        explanations: Atom importance scores
        threshold: Importance threshold
    
    Returns:
        float: Parsimony score in [0, 1]
        - 1.0 = Few atoms important (sparse, good)
        - 0.5 = ~50% atoms important (moderate)
        - 0.0 = All atoms important (dense, bad)
    """
    # Count important atoms (above threshold)
    important = np.sum(np.abs(explanations) > threshold)
    total = len(explanations)
    
    # Fraction important
    frac_important = important / (total + 1e-8)
    
    # Exponential decay: prefer sparse
    parsimony = np.exp(-frac_important)
    return float(parsimony)

# ============================================
# METRIC 5: ROBUSTNESS
# ============================================

def compute_robustness(model, smiles_list, explanations, n_perturbations=5):
    """
    Measure if explanations stable under molecular perturbations:
    Small molecule changes → small explanation changes?
    
    Args:
        model: Trained model
        smiles_list: SMILES strings
        explanations: Original explanations
        n_perturbations: Number of perturbations per molecule
    
    Returns:
        float: Robustness in [0, 1]
        - 1.0 = Explanations stable (robust)
        - 0.5 = Some sensitivity
        - 0.0 = Highly sensitive (fragile)
    """
    perturbation_distances = []
    
    for smi in smiles_list[:min(10, len(smiles_list))]:  # Test on subset
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        for _ in range(n_perturbations):
            try:
                # Add/remove random atom (simple perturbation)
                perm_mol = Chem.RWMol(mol)
                # ... (add perturbation logic)
                
                # Get new explanation
                # new_expl = model.explain(perturbed_smiles)
                # perturbation_distances.append(cosine_dist(explanations, new_expl))
            except:
                pass
    
    if not perturbation_distances:
        return 0.5  # Default
    
    mean_distance = np.mean(perturbation_distances)
    robustness = 1 - min(mean_distance, 1.0)
    return float(robustness)

# ============================================
# COMBINED QUALITY SCORE
# ============================================

def compute_quality_score(stability, consistency, faithfulness, 
                         parsimony, robustness):
    """
    Combine all 5 metrics into single quality score.
    
    Returns:
        float: Quality score in [0, 1]
    """
    # Weights
    weights = {
        'stability': 0.25,
        'consistency': 0.20,
        'faithfulness': 0.25,
        'parsimony': 0.15,
        'robustness': 0.15
    }
    
    # Normalize to [0, 1]
    consistency_norm = (consistency + 1) / 2  # Convert from [-1, 1] to [0, 1]
    faithfulness_norm = min(faithfulness / 2, 1.0)  # Clip at 2.0
    
    metrics = {
        'stability': stability,
        'consistency': consistency_norm,
        'faithfulness': faithfulness_norm,
        'parsimony': parsimony,
        'robustness': robustness
    }
    
    # Weighted sum
    score = sum(metrics[k] * weights[k] for k in weights)
    return float(score)

# ============================================
# REPORTING
# ============================================

def print_quality_report(model_name, params, stability, consistency, 
                        faithfulness, parsimony, robustness, quality_score):
    """
    Pretty-print explanation quality report.
    """
    print(f"""
    ╔════════════════════════════════════════════════════════╗
    ║  EXPLANATION QUALITY REPORT
    ║  Model: {model_name}
    ╚════════════════════════════════════════════════════════╝
    
    Parameters: {params}
    
    Metrics:
    ✓ Stability       {stability:.3f}    (Determinism across runs)
    ✓ Consistency     {consistency:.3f}    (Similar mols → similar explanations)
    ✓ Faithfulness    {faithfulness:.3f}    (Important atoms impact predictions)
    ✓ Parsimony       {parsimony:.3f}    (Few atoms, many atoms)
    ✓ Robustness      {robustness:.3f}    (Stable under perturbations)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    QUALITY SCORE: {quality_score:.3f}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

if __name__ == "__main__":
    set_all_seeds(42)
    print("✅ Deterministic SAR Framework loaded")
```

---

## PHASE 2: BENCHMARK LOADING (Week 1)

### File: `scripts/benchmark_loader.py`

```python
"""
Load public SAR benchmark datasets for evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from rdkit import Chem

class BenchmarkLoader:
    def __init__(self, cache_dir="data/benchmarks"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_bbbp(self):
        """Blood-Brain Barrier Penetration (2K compounds)"""
        print("📥 Loading BBBP...")
        # Can be loaded via MoleculeNet or direct download
        # Return: (smiles, labels)
        pass
    
    def load_solubility(self):
        """Delaney Solubility (1.1K compounds)"""
        print("📥 Loading Solubility...")
        pass
    
    def load_tox21(self, assay='NR-ER'):
        """Tox21 (13K compounds, 12 assays)"""
        print(f"📥 Loading Tox21 {assay}...")
        pass
    
    def load_chembl_kinase(self, target='EGFR'):
        """ChEMBL Kinase IC50 (2-5K compounds)"""
        print(f"📥 Loading ChEMBL {target} IC50...")
        pass
    
    def load_qsar(self):
        """Your internal QSAR data"""
        path = "data/QSAR_potency_20um_for_GIF.xlsx"
        print(f"📥 Loading QSAR data from {path}...")
        df = pd.read_excel(path)
        smiles = df['cleanedMol'].values
        labels = df['classLabel'].values
        return smiles, labels

if __name__ == "__main__":
    loader = BenchmarkLoader()
    smiles, labels = loader.load_qsar()
    print(f"✅ Loaded {len(smiles)} compounds")
```

---

## DOCUMENTATION

Full outline stored in: `/home/nbhatta1/Documents/Explainable-AI-SAR/NEURIPS_PAPER_OUTLINE.md`

---

## NEXT STEPS

1. ✅ Read `NEURIPS_PAPER_OUTLINE.md` (comprehensive guide)
2. ⬜ Create `deterministic_sar_framework.py` (Week 1)
3. ⬜ Create `benchmark_loader.py` (Week 1)
4. ⬜ Run experiments on benchmarks (Week 2-3)
5. ⬜ Write paper (Week 4-6)

**Ready to start coding?** Let me know!

