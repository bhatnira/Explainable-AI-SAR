#!/usr/bin/env python3
"""
Fast sweep over circular fingerprint parameters and interpretation types.
- Stage 1 (fast): grid over radius x nBits x useFeatures x useChirality
  * Precompute Morgan env hashes per molecule (per radius/features/chiral)
  * Fold to different nBits without recomputing RDKit
  * Train fast LogisticRegression, compute AUC/ACC
  * Compute collision proxies from folded hashes
  * Objective = AUC - LAMBDA_COLLISION * collision_rate
  * Save all results to CSV
- Stage 2 (focused): evaluate SHAP, LIME, and Ensemble faithfulness on top-K configs
  * Small test subset, small LIME samples
  * Output best config and interpretation type (LIME/SHAP/ENSEMBLE)

Outputs:
- analysis/sar/results/fast_sweep_results.csv
- analysis/sar/results/fast_sweep_best.json
- analysis/sar/results/fast_sweep_topK_details.json
"""
import os
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# plotting backend (no GUI)
import matplotlib as mpl
mpl.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import Parallel, delayed

# Chemistry
from rdkit import Chem
from rdkit.Chem import AllChem

# Explainability (used in Stage 2)
import shap
from lime.lime_tabular import LimeTabularExplainer

# Import project utilities (data path)
import sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from visualization.create_dynamic_parameter_movies import DynamicParameterMovieCreator


# --------------------- Config Helpers ---------------------
def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y"}


# --------------------- Data Loading ---------------------
def load_balanced_data(n_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    creator = DynamicParameterMovieCreator()
    df = pd.read_excel(creator._data_path())
    df0 = df[df['classLabel'] == 0].sample(n=n_per_class, random_state=42)
    df1 = df[df['classLabel'] == 1].sample(n=n_per_class, random_state=42)
    dfb = pd.concat([df0, df1], ignore_index=True)
    smiles = dfb['cleanedMol'].astype(str).values
    y = dfb['classLabel'].astype(int).values
    return smiles, y


# --------------------- Morgan Hash Cache ---------------------
def morgan_env_hashes_for_mol(smi: str, radius: int, useFeatures: bool, useChirality: bool) -> Dict[int, int]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    # SparseIntVect keyed by 32-bit ids (environment hashes), values are counts
    fp = AllChem.GetMorganFingerprint(
        mol, radius, useFeatures=useFeatures, useChirality=useChirality
    )
    # Convert to dict {hash_id: count}
    return {int(k): int(v) for k, v in fp.GetNonzeroElements().items()}


def precompute_hashes(smiles: np.ndarray, radius: int, useFeatures: bool, useChirality: bool, n_jobs: int) -> List[Dict[int, int]]:
    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(morgan_env_hashes_for_mol)(s, radius, useFeatures, useChirality) for s in smiles
    )


# --------------------- Folding + Features ---------------------
def fold_hashes_to_bitvect(hashes: Dict[int, int], nBits: int) -> np.ndarray:
    # Binary presence bit vector to mimic RDKit bitvect usage in pipeline
    x = np.zeros(nBits, dtype=np.uint8)
    for h, c in hashes.items():
        b = h % nBits
        x[b] = 1
    return x


def build_X_from_hash_cache(hash_cache: List[Dict[int, int]], nBits: int) -> np.ndarray:
    return np.vstack([fold_hashes_to_bitvect(h, nBits) for h in hash_cache])


# --------------------- Collisions (Proxy) ---------------------
def collision_metrics_from_hash_cache(hash_cache: List[Dict[int, int]], nBits: int) -> Dict[str, float]:
    # Count unique env hashes across dataset
    all_env = set()
    for h in hash_cache:
        all_env.update(h.keys())
    total_unique_env = len(all_env)
    if total_unique_env == 0:
        return {
            'collision_rate': 0.0,
            'avg_frags_per_active_bit': 0.0,
        }
    # Map env hashes to folded bits and count how many envs land per bit
    bit_env_counts = np.zeros(nBits, dtype=np.int64)
    for env in all_env:
        bit_env_counts[env % nBits] += 1
    active_bits = np.count_nonzero(bit_env_counts)
    avg_frags_per_active_bit = float(np.mean(bit_env_counts[bit_env_counts > 0])) if active_bits > 0 else 0.0
    # Expected collision rate: how many envs collided into same bit
    collisions = int(np.sum(bit_env_counts) - active_bits)
    collision_rate = float(collisions / max(1, np.sum(bit_env_counts)))
    return {
        'collision_rate': collision_rate,
        'avg_frags_per_active_bit': avg_frags_per_active_bit,
    }


# --------------------- Modeling ---------------------
def train_eval_lr(X: np.ndarray, y: np.ndarray, test_size=0.2, seed=42) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    clf = LogisticRegression(max_iter=400, n_jobs=1, solver="lbfgs")
    clf.fit(X_train, y_train)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        s = clf.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-s))
    y_pred = (y_proba > 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = float('nan')
    return {'acc': acc, 'auc': auc, 'model': clf, 'split': (X_train, X_test, y_train, y_test)}


# --------------------- Stage 2 Explainability ---------------------
def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = np.sum(np.abs(v))
    return v if s == 0 else v / s


def shap_values_for_lr(model, X_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    try:
        explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
        sv = explainer.shap_values(X_eval)
        return np.array(sv[1]) if isinstance(sv, list) else np.array(sv)
    except Exception:
        masker = shap.maskers.Independent(X_train)
        e = shap.Explainer(model, masker=masker)
        vals = e(X_eval)
        return np.array(vals.values) if hasattr(vals, 'values') else np.array(vals)


def lime_values_for_subset(model, X_train: np.ndarray, X_eval: np.ndarray, idx_subset: List[int], num_features: int = 50, num_samples: int = 400) -> np.ndarray:
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=[f"bit_{i}" for i in range(X_train.shape[1])],
        class_names=['inactive', 'active'],
        discretize_continuous=False,
        sample_around_instance=True,
        mode='classification'
    )
    def predict_fn(xx):
        try:
            p = model.predict_proba(xx)[:, 1]
            return np.vstack([1 - p, p]).T
        except Exception:
            s = model.decision_function(xx)
            p = 1 / (1 + np.exp(-s))
            return np.vstack([1 - p, p]).T
    lime_mat = np.zeros((X_eval.shape[0], X_eval.shape[1]), dtype=float)
    for i in idx_subset:
        exp = explainer.explain_instance(X_eval[i], predict_fn, num_features=min(num_features, X_eval.shape[1]), num_samples=num_samples)
        wmap = dict(exp.as_map().get(1, []))
        for j, w in wmap.items():
            lime_mat[i, int(j)] = float(w)
    return lime_mat


def faithfulness_proxy(model, X_eval: np.ndarray, attributions: np.ndarray, topk: int = 8) -> float:
    # Remove top positive features and measure drop in prob.
    drops = []
    for i in range(min(len(X_eval), attributions.shape[0])):
        w = attributions[i]
        # pick top-k bits with positive contribution
        pos_idx = [j for j in np.argsort(-w)[:topk] if w[j] > 0]
        if not pos_idx:
            continue
        p0 = float(model.predict_proba(X_eval[i:i+1])[:, 1][0]) if hasattr(model, 'predict_proba') else float(1/(1+np.exp(-model.decision_function(X_eval[i:i+1])[0])))
        x_mod = X_eval[i].copy()
        x_mod[pos_idx] = 0
        p1 = float(model.predict_proba(x_mod[None, :])[:, 1][0]) if hasattr(model, 'predict_proba') else float(1/(1+np.exp(-model.decision_function(x_mod[None, :])[0])))
        drops.append(max(0.0, p0 - p1))
    return float(np.mean(drops)) if drops else 0.0


# --------------------- Main Sweep ---------------------
@dataclass
class Config:
    radius: int
    nBits: int
    useFeatures: int
    useChirality: int


def main():
    # Params
    N_PER_CLASS = env_int('N_PER_CLASS', 30)
    MAX_WORKERS = env_int('MAX_WORKERS', max(1, os.cpu_count() or 2))
    LAMBDA_COLLISION = env_float('LAMBDA_COLLISION', 0.2)
    STAGE2_TOPK_CONFIGS = env_int('STAGE2_TOPK_CONFIGS', 8)
    STAGE2_TEST_SUBSET = env_int('STAGE2_TEST_SUBSET', 80)
    ENSEMBLE_ALPHA = env_float('ENSEMBLE_ALPHA', 0.5)

    # Parameter grid (same as GIFs)
    radii = list(range(1, 9))
    nbits_list = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    features_list = [0, 1]
    chiral_list = [0, 1]

    here = Path(__file__).resolve().parent
    results_dir = here / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    smiles, y = load_balanced_data(N_PER_CLASS)

    # Stage 1: Precompute hash caches per (radius, features, chiral)
    rfcs = [(r, f, c) for r in radii for f in features_list for c in chiral_list]
    hash_caches: Dict[Tuple[int, int, int], List[Dict[int, int]]] = {}
    for (r, f, c) in rfcs:
        hash_caches[(r, f, c)] = precompute_hashes(smiles, r, bool(f), bool(c), n_jobs=MAX_WORKERS)

    # Evaluate all configs in parallel (over nBits only needs folding)
    configs: List[Config] = [Config(r, n, f, c) for r, f, c in rfcs for n in nbits_list]

    def eval_config(cfg: Config):
        cache = hash_caches[(cfg.radius, cfg.useFeatures, cfg.useChirality)]
        X = build_X_from_hash_cache(cache, cfg.nBits)
        perf = train_eval_lr(X, y)
        coll = collision_metrics_from_hash_cache(cache, cfg.nBits)
        obj = float(perf['auc']) - LAMBDA_COLLISION * float(coll['collision_rate'])
        return {
            'radius': cfg.radius,
            'nBits': cfg.nBits,
            'useFeatures': cfg.useFeatures,
            'useChirality': cfg.useChirality,
            'acc': perf['acc'],
            'auc': perf['auc'],
            'collision_rate': coll['collision_rate'],
            'avg_frags_per_active_bit': coll['avg_frags_per_active_bit'],
            'objective': obj,
        }

    rows = Parallel(n_jobs=MAX_WORKERS, backend="loky")(delayed(eval_config)(cfg) for cfg in configs)
    df_stage1 = pd.DataFrame(rows)
    csv_path = results_dir / 'fast_sweep_results.csv'
    df_stage1.to_csv(csv_path, index=False)

    # Select top-K for Stage 2
    topK = df_stage1.sort_values('objective', ascending=False).head(STAGE2_TOPK_CONFIGS)
    top_details = []

    for _, rec in topK.iterrows():
        r = int(rec['radius']); n = int(rec['nBits']); f = int(rec['useFeatures']); c = int(rec['useChirality'])
        cache = hash_caches[(r, f, c)]
        X = build_X_from_hash_cache(cache, n)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model = LogisticRegression(max_iter=400, n_jobs=1, solver="lbfgs").fit(X_train, y_train)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            s = model.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-s))
        auc = float(roc_auc_score(y_test, y_proba))

        # Subset for explainability
        m = min(STAGE2_TEST_SUBSET, X_test.shape[0])
        idx_subset = list(range(m))
        X_eval = X_test[idx_subset]

        # SHAP
        shap_vals = shap_values_for_lr(model, X_train, X_eval)
        shap_norm = np.vstack([normalize(shap_vals[i]) for i in range(shap_vals.shape[0])])
        shap_faith = faithfulness_proxy(model, X_eval, shap_norm, topk=8)

        # LIME (smaller budget)
        lime_vals = lime_values_for_subset(model, X_train, X_eval, idx_subset, num_features=40, num_samples=350)
        lime_norm = np.vstack([normalize(lime_vals[i]) for i in range(lime_vals.shape[0])])
        lime_faith = faithfulness_proxy(model, X_eval, lime_norm, topk=8)

        # Ensemble
        ens_vals = ENSEMBLE_ALPHA * shap_norm + (1 - ENSEMBLE_ALPHA) * lime_norm
        ens_faith = faithfulness_proxy(model, X_eval, ens_vals, topk=8)

        # Pick best interpretation by faithfulness, tie-break with lower collision
        faiths = {
            'SHAP': shap_faith,
            'LIME': lime_faith,
            'ENSEMBLE': ens_faith,
        }
        best_interp = max(faiths.items(), key=lambda t: t[1])[0]

        top_details.append({
            'radius': r,
            'nBits': n,
            'useFeatures': f,
            'useChirality': c,
            'auc': auc,
            'collision_rate': float(rec['collision_rate']),
            'avg_frags_per_active_bit': float(rec['avg_frags_per_active_bit']),
            'faithfulness_SHAP': shap_faith,
            'faithfulness_LIME': lime_faith,
            'faithfulness_ENSEMBLE': ens_faith,
            'best_interpretation': best_interp,
        })

    # Decide overall best: maximize faithfulness, penalize collisions; tie-break by AUC
    def score_row(row):
        best_f = max(row['faithfulness_SHAP'], row['faithfulness_LIME'], row['faithfulness_ENSEMBLE'])
        return best_f - LAMBDA_COLLISION * row['collision_rate']

    if top_details:
        best = max(top_details, key=score_row)
        with open(results_dir / 'fast_sweep_topK_details.json', 'w') as f:
            json.dump(top_details, f, indent=2)
        with open(results_dir / 'fast_sweep_best.json', 'w') as f:
            json.dump(best, f, indent=2)
        print("Best config:", best)
    else:
        print("No top-K details computed.")


if __name__ == '__main__':
    main()
