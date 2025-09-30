#!/usr/bin/env python3
"""
Scaffold-split evaluation to assess SAR and chemical reasoning.
- Splits data by Bemis–Murcko scaffolds (no scaffold leakage)
- Trains LR on circular fingerprints for a chosen config
- Computes performance (ACC/AUC), collision proxies
- Computes SHAP/LIME/Ensemble local attributions on test subset
- Computes stability (Jaccard of top-k bits across bootstraps)
- Computes faithfulness curves (deletion/insertion) and AUCs
- Saves figures under analysis/sar/figures/scaffold_split_*
- Saves metrics to analysis/sar/results/scaffold_split_summary.json

Env:
- NBITS (default: from fast_sweep_best.json or 8192)
- RADIUS (default: from fast_sweep_best.json or 1)
- USEFEATURES, USECHIRALITY (0/1)
- N_PER_CLASS (default 60)
- TEST_FRAC (default 0.2)
- ENSEMBLE_ALPHA (default 0.5)
- TOPK_BITS (default 32)
- N_BOOTSTRAP (default 5)
- EXPLAIN_SUBSET (default 100)
- REPEATS (default 1)
- BASE_SEED (default 123)
"""
import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Nature-style figure utils
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.sar.fig_style import apply_nature_style, colsize, save_figure
apply_nature_style()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import shap
from lime.lime_tabular import LimeTabularExplainer

# Project imports
from visualization.create_dynamic_parameter_movies import DynamicParameterMovieCreator


# ----------------- utils -----------------
def env_int(n, d):
    try:
        return int(os.getenv(n, str(d)))
    except Exception:
        return d

def env_float(n, d):
    try:
        return float(os.getenv(n, str(d)))
    except Exception:
        return d

def normalize(v):
    v = np.asarray(v, dtype=float)
    s = np.sum(np.abs(v))
    return v if s == 0 else v / s


def load_best_defaults(here: Path):
    best_path = here / 'results' / 'fast_sweep_best.json'
    if best_path.exists():
        b = json.loads(best_path.read_text())
        return int(b['nBits']), int(b['radius']), int(b['useFeatures']), int(b['useChirality'])
    return 8192, 1, 0, 1


def bemis_murcko_scaffold_smi(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ''
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(scaf) if scaf is not None else ''


def scaffold_split(smiles: np.ndarray, y: np.ndarray, test_frac: float, seed: int = 42):
    # group indices by scaffold
    scaffold_to_idx: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        scaf = bemis_murcko_scaffold_smi(smi)
        scaffold_to_idx.setdefault(scaf, []).append(i)
    scaffolds = list(scaffold_to_idx.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffolds)
    n = len(smiles)
    test_target = int(round(test_frac * n))
    test_idx = []
    total = 0
    for sc in scaffolds:
        idxs = scaffold_to_idx[sc]
        if total < test_target:
            test_idx.extend(idxs)
            total += len(idxs)
        else:
            break
    test_idx = sorted(set(test_idx))
    train_idx = sorted(set(range(n)) - set(test_idx))
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def train_lr(X, y, seed=None):
    # ...existing code...
    clf = LogisticRegression(max_iter=500, n_jobs=1, solver='lbfgs', random_state=seed)
    clf.fit(X, y)
    return clf


def predict_proba(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))


def shap_values_lr(model, X_train, X_eval):
    try:
        explainer = shap.LinearExplainer(model, X_train, feature_dependence='independent')
        sv = explainer.shap_values(X_eval)
        return np.array(sv[1]) if isinstance(sv, list) else np.array(sv)
    except Exception:
        masker = shap.maskers.Independent(X_train)
        e = shap.Explainer(model, masker=masker)
        vals = e(X_eval)
        return np.array(vals.values) if hasattr(vals, 'values') else np.array(vals)


def lime_values(model, X_train, X_eval, idx_subset: List[int], num_features=40, num_samples=400):
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=[f'bit_{i}' for i in range(X_train.shape[1])],
        class_names=['inactive', 'active'],
        discretize_continuous=False,
        sample_around_instance=True,
        mode='classification'
    )
    def predict_fn(xx):
        p = predict_proba(model, xx)
        return np.vstack([1 - p, p]).T
    mat = np.zeros_like(X_eval, dtype=float)
    for i in idx_subset:
        exp = explainer.explain_instance(X_eval[i], predict_fn, num_features=min(num_features, X_eval.shape[1]), num_samples=num_samples)
        wmap = dict(exp.as_map().get(1, []))
        for j, w in wmap.items():
            mat[i, int(j)] = float(w)
    return mat


def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return float(len(A & B) / len(A | B))


def stability_topk_bits(attribs: np.ndarray, X_eval: np.ndarray, topk: int, n_boot=5, seed=123) -> float:
    # bootstrap rows and compute jaccard between top-k bit sets
    rng = np.random.RandomState(seed)
    sets = []
    for b in range(n_boot):
        idx = rng.choice(np.arange(len(X_eval)), size=len(X_eval), replace=True)
        A = np.mean(attribs[idx], axis=0)
        # Only consider active bits on average to avoid selecting inactive
        active = np.where(np.mean(X_eval[idx], axis=0) > 0)[0]
        if active.size == 0:
            sets.append(set())
            continue
        order = active[np.argsort(-np.abs(A[active]))]
        sets.append(set(map(int, order[:topk])))
    if len(sets) < 2:
        return 0.0
    scores = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            scores.append(jaccard(sets[i], sets[j]))
    return float(np.mean(scores)) if scores else 0.0


def deletion_insertion_curves(model, X_eval: np.ndarray, attribs: np.ndarray, steps: int = 10):
    n, d = X_eval.shape
    # positive contribution ranking per sample
    curves_del = []
    curves_ins = []
    for i in range(n):
        w = attribs[i]
        # rank bits by descending attribution
        order = np.argsort(-w)
        # consider only bits active in sample for deletion; for insertion start from zeros
        active = np.where(X_eval[i] > 0)[0]
        order_del = [j for j in order if j in active]
        # Deletion
        x = X_eval[i].copy()
        p0 = float(predict_proba(model, x[None, :])[0])
        del_vals = [p0]
        for t in range(1, steps+1):
            k = int(round(t/steps * len(order_del)))
            if k > 0:
                x2 = x.copy()
                x2[order_del[:k]] = 0
                del_vals.append(float(predict_proba(model, x2[None, :])[0]))
            else:
                del_vals.append(p0)
        curves_del.append(del_vals)
        # Insertion
        order_ins = order  # global for insertion
        xz = np.zeros_like(x)
        p00 = float(predict_proba(model, xz[None, :])[0])
        ins_vals = [p00]
        for t in range(1, steps+1):
            k = int(round(t/steps * len(order_ins)))
            xi = xz.copy()
            if k > 0:
                idx = order_ins[:k]
                xi[idx] = X_eval[i][idx]
            ins_vals.append(float(predict_proba(model, xi[None, :])[0]))
        curves_ins.append(ins_vals)
    # average curves and compute AUCs
    del_arr = np.array(curves_del)
    ins_arr = np.array(curves_ins)
    x_axis = np.linspace(0, 1, del_arr.shape[1])
    del_mean = del_arr.mean(axis=0)
    ins_mean = ins_arr.mean(axis=0)
    auc_del = float(np.trapz(np.maximum(0, del_mean[0] - del_mean), x_axis))
    auc_ins = float(np.trapz(np.maximum(0, ins_mean - ins_mean[0]), x_axis))
    return (x_axis, del_mean, ins_mean, auc_del, auc_ins)


def _ci95(arr: List[float]):
    a = np.asarray(arr, dtype=float)
    n = a.size
    if n == 0:
        return {'mean': 0.0, 'std': 0.0, 'n': 0, 'ci95': 0.0}
    mean = float(np.mean(a))
    if n == 1:
        return {'mean': mean, 'std': 0.0, 'n': 1, 'ci95': 0.0}
    std = float(np.std(a, ddof=1))
    ci = 1.96 * std / np.sqrt(n)
    return {'mean': mean, 'std': std, 'n': int(n), 'ci95': float(ci)}


def run_once(seed: int, here: Path, figs_dir: Path, results_dir: Path,
             NBITS: int, RADIUS: int, USE_FEATURES: int, USE_CHIRAL: int,
             N_PER_CLASS: int, TEST_FRAC: float, ENSEMBLE_ALPHA: float,
             TOPK_BITS: int, N_BOOT: int, EXPL_SUB: int, write_files: bool = True):
    # Ensure per-run reproducibility
    np.random.seed(seed)

    creator = DynamicParameterMovieCreator()
    df = pd.read_excel(creator._data_path())
    # balanced subsample using seed
    df0 = df[df['classLabel'] == 0].sample(n=N_PER_CLASS, random_state=seed)
    df1 = df[df['classLabel'] == 1].sample(n=N_PER_CLASS, random_state=seed)
    dfb = pd.concat([df0, df1], ignore_index=True)
    smiles = dfb['cleanedMol'].astype(str).values
    y = dfb['classLabel'].astype(int).values

    # scaffold split with seed
    tr_idx, te_idx = scaffold_split(smiles, y, TEST_FRAC, seed=seed)
    s_train, s_test = smiles[tr_idx], smiles[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    # featurize
    X_train = creator.featurize(s_train, RADIUS, NBITS, bool(USE_FEATURES), bool(USE_CHIRAL))
    X_test = creator.featurize(s_test, RADIUS, NBITS, bool(USE_FEATURES), bool(USE_CHIRAL))

    # train & eval (seeded)
    model = train_lr(X_train, y_train, seed=seed)
    y_proba = predict_proba(model, X_test)
    y_pred = (y_proba > 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_proba))

    # confusion matrix (optional)
    if write_files:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=colsize(single=True))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f'Scaffold split CM (acc={acc:.3f}, auc={auc:.3f})')
        save_figure(fig, figs_dir / 'scaffold_split_confusion_matrix.png', dpi=1200, formats=("tiff","png","pdf"))

    # explanations on subset
    m = min(EXPL_SUB, X_test.shape[0])
    idx_subset = list(range(m))
    X_eval = X_test[idx_subset]

    shap_vals = shap_values_lr(model, X_train, X_eval)
    shap_norm = np.vstack([normalize(shap_vals[i]) for i in range(shap_vals.shape[0])])
    lime_vals = lime_values(model, X_train, X_eval, idx_subset, num_features=40, num_samples=350)
    lime_norm = np.vstack([normalize(lime_vals[i]) for i in range(lime_vals.shape[0])])
    ens_vals = ENSEMBLE_ALPHA * shap_norm + (1 - ENSEMBLE_ALPHA) * lime_norm

    # stability (bits)
    stab_shap = stability_topk_bits(shap_norm, X_eval, topk=TOPK_BITS, n_boot=N_BOOT, seed=seed)
    stab_lime = stability_topk_bits(lime_norm, X_eval, topk=TOPK_BITS, n_boot=N_BOOT, seed=seed)
    stab_ens = stability_topk_bits(ens_vals, X_eval, topk=TOPK_BITS, n_boot=N_BOOT, seed=seed)

    # faithfulness curves
    steps = 12
    x, del_mean_shap, ins_mean_shap, auc_del_shap, auc_ins_shap = deletion_insertion_curves(model, X_eval, shap_norm, steps)
    _, del_mean_lime, ins_mean_lime, auc_del_lime, auc_ins_lime = deletion_insertion_curves(model, X_eval, lime_norm, steps)
    _, del_mean_ens, ins_mean_ens, auc_del_ens, auc_ins_ens = deletion_insertion_curves(model, X_eval, ens_vals, steps)

    # plot curves (optional)
    if write_files:
        fig, ax = plt.subplots(figsize=colsize(single=False))
        ax.plot(x, del_mean_shap, label='SHAP deletion', color='#4ECDC4')
        ax.plot(x, del_mean_lime, label='LIME deletion', color='#FFBE0B')
        ax.plot(x, del_mean_ens, label='ENS deletion', color='#3A86FF')
        ax.set_xlabel('Fraction removed (top-ranked bits)'); ax.set_ylabel('Mean P(active)')
        ax.set_title('Deletion curves (scaffold split)')
        ax.legend()
        save_figure(fig, figs_dir / 'scaffold_split_deletion_curves.png', dpi=1200, formats=("tiff","png","pdf"))

        fig, ax = plt.subplots(figsize=colsize(single=False))
        ax.plot(x, ins_mean_shap, label='SHAP insertion', color='#4ECDC4')
        ax.plot(x, ins_mean_lime, label='LIME insertion', color='#FFBE0B')
        ax.plot(x, ins_mean_ens, label='ENS insertion', color='#3A86FF')
        ax.set_xlabel('Fraction inserted (top-ranked bits)'); ax.set_ylabel('Mean P(active)')
        ax.set_title('Insertion curves (scaffold split)')
        ax.legend()
        save_figure(fig, figs_dir / 'scaffold_split_insertion_curves.png', dpi=1200, formats=("tiff","png","pdf"))

    # summary dict (optionally persisted by caller)
    out = {
        'config': {'NBITS': NBITS, 'RADIUS': RADIUS, 'USE_FEATURES': USE_FEATURES, 'USE_CHIRALITY': USE_CHIRAL, 'TEST_FRAC': TEST_FRAC},
        'seed': seed,
        'performance': {'acc': acc, 'auc': auc},
        'stability_topk_bits': {'SHAP': stab_shap, 'LIME': stab_lime, 'ENSEMBLE': stab_ens},
        'faithfulness_auc': {
            'SHAP': {'deletion_auc': auc_del_shap, 'insertion_auc': auc_ins_shap},
            'LIME': {'deletion_auc': auc_del_lime, 'insertion_auc': auc_ins_lime},
            'ENSEMBLE': {'deletion_auc': auc_del_ens, 'insertion_auc': auc_ins_ens},
        }
    }

    if write_files:
        with open(results_dir / 'scaffold_split_summary.json', 'w') as f:
            json.dump(out, f, indent=2)

    return out


def main():
    here = Path(__file__).resolve().parent
    figs_dir = here / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = here / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Defaults from fast_sweep
    NBITS_DEF, RADIUS_DEF, FEAT_DEF, CHIR_DEF = load_best_defaults(here)

    NBITS = env_int('NBITS', NBITS_DEF)
    RADIUS = env_int('RADIUS', RADIUS_DEF)
    USE_FEATURES = env_int('USEFEATURES', FEAT_DEF)
    USE_CHIRAL = env_int('USECHIRALITY', CHIR_DEF)
    N_PER_CLASS = env_int('N_PER_CLASS', 60)
    TEST_FRAC = float(os.getenv('TEST_FRAC', '0.2'))
    ENSEMBLE_ALPHA = env_float('ENSEMBLE_ALPHA', 0.5)
    TOPK_BITS = env_int('TOPK_BITS', 32)
    N_BOOT = env_int('N_BOOTSTRAP', 5)
    EXPL_SUB = env_int('EXPLAIN_SUBSET', 100)
    REPEATS = env_int('REPEATS', 1)
    BASE_SEED = env_int('BASE_SEED', 123)

    if REPEATS <= 1:
        out = run_once(
            seed=BASE_SEED, here=here, figs_dir=figs_dir, results_dir=results_dir,
            NBITS=NBITS, RADIUS=RADIUS, USE_FEATURES=USE_FEATURES, USE_CHIRAL=USE_CHIRAL,
            N_PER_CLASS=N_PER_CLASS, TEST_FRAC=TEST_FRAC, ENSEMBLE_ALPHA=ENSEMBLE_ALPHA,
            TOPK_BITS=TOPK_BITS, N_BOOT=N_BOOT, EXPL_SUB=EXPL_SUB, write_files=True
        )
        print('Scaffold-split analysis complete.')
        return

    # Multi-seed replicates
    seeds = [BASE_SEED + i for i in range(REPEATS)]
    per_seed = []
    for i, sd in enumerate(seeds):
        write = (i == 0)  # write figures for first replicate only
        res = run_once(
            seed=sd, here=here, figs_dir=figs_dir, results_dir=results_dir,
            NBITS=NBITS, RADIUS=RADIUS, USE_FEATURES=USE_FEATURES, USE_CHIRAL=USE_CHIRAL,
            N_PER_CLASS=N_PER_CLASS, TEST_FRAC=TEST_FRAC, ENSEMBLE_ALPHA=ENSEMBLE_ALPHA,
            TOPK_BITS=TOPK_BITS, N_BOOT=N_BOOT, EXPL_SUB=EXPL_SUB, write_files=write
        )
        per_seed.append(res)

    # Aggregate with 95% CIs
    accs = [r['performance']['acc'] for r in per_seed]
    aucs = [r['performance']['auc'] for r in per_seed]

    def mget(path):
        # path like ('stability_topk_bits','SHAP') or ('faithfulness_auc','SHAP','deletion_auc')
        vals = []
        for r in per_seed:
            v = r
            for k in path:
                v = v[k]
            vals.append(float(v))
        return vals

    agg = {
        'config': {'NBITS': NBITS, 'RADIUS': RADIUS, 'USE_FEATURES': USE_FEATURES, 'USE_CHIRALITY': USE_CHIRAL, 'TEST_FRAC': TEST_FRAC,
                   'TOPK_BITS': TOPK_BITS, 'N_BOOTSTRAP': N_BOOT, 'EXPLAIN_SUBSET': EXPL_SUB, 'REPEATS': REPEATS, 'BASE_SEED': BASE_SEED},
        'seeds': seeds,
        'performance': {'acc': _ci95(accs), 'auc': _ci95(aucs)},
        'stability_topk_bits': {
            'SHAP': _ci95(mget(('stability_topk_bits','SHAP'))),
            'LIME': _ci95(mget(('stability_topk_bits','LIME'))),
            'ENSEMBLE': _ci95(mget(('stability_topk_bits','ENSEMBLE'))),
        },
        'faithfulness_auc': {
            'SHAP': {
                'deletion_auc': _ci95(mget(('faithfulness_auc','SHAP','deletion_auc'))),
                'insertion_auc': _ci95(mget(('faithfulness_auc','SHAP','insertion_auc'))),
            },
            'LIME': {
                'deletion_auc': _ci95(mget(('faithfulness_auc','LIME','deletion_auc'))),
                'insertion_auc': _ci95(mget(('faithfulness_auc','LIME','insertion_auc'))),
            },
            'ENSEMBLE': {
                'deletion_auc': _ci95(mget(('faithfulness_auc','ENSEMBLE','deletion_auc'))),
                'insertion_auc': _ci95(mget(('faithfulness_auc','ENSEMBLE','insertion_auc'))),
            },
        },
        'per_seed_results': per_seed,
    }

    with open(results_dir / 'scaffold_split_replicates.json', 'w') as f:
        json.dump(agg, f, indent=2)

    # Save per-seed CSV for quick review
    import csv
    csv_path = results_dir / 'scaffold_split_replicates.csv'
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['seed','acc','auc','stab_shap','stab_lime','stab_ens','del_shap','ins_shap','del_lime','ins_lime','del_ens','ins_ens'])
        for r in per_seed:
            w.writerow([
                r['seed'], r['performance']['acc'], r['performance']['auc'],
                r['stability_topk_bits']['SHAP'], r['stability_topk_bits']['LIME'], r['stability_topk_bits']['ENSEMBLE'],
                r['faithfulness_auc']['SHAP']['deletion_auc'], r['faithfulness_auc']['SHAP']['insertion_auc'],
                r['faithfulness_auc']['LIME']['deletion_auc'], r['faithfulness_auc']['LIME']['insertion_auc'],
                r['faithfulness_auc']['ENSEMBLE']['deletion_auc'], r['faithfulness_auc']['ENSEMBLE']['insertion_auc'],
            ])

    print('Scaffold-split replicates complete.')


if __name__ == '__main__':
    main()
