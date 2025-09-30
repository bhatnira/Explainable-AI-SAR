#!/usr/bin/env python3
"""
Fragment-level SAR and Attribution Validation
- Computes fragment/bit-level SAR signals and maps to atoms
- Evaluates stability, collision sensitivity, enrichment, clustering proxies
- Tests perturbation/masking faithfulness
- Assesses cross-method agreement (Permutation, SHAP, LIME, Ensemble)
Outputs JSON/CSV under analysis/sar/results and figures under analysis/sar/figures
"""
import os
import json
import math
import itertools
from pathlib import Path
import sys

# Ensure project root is on sys.path for imports like 'visualization.*'
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Nature-style figure utils
from analysis.sar.fig_style import apply_nature_style, colsize, save_figure
apply_nature_style()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

# Explainability packages
import shap
from lime.lime_tabular import LimeTabularExplainer

# Chemistry
from rdkit import Chem

# Reuse utilities from visualization pipeline
from visualization.create_dynamic_parameter_movies import DynamicParameterMovieCreator


# ------------------------- Utilities -------------------------
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

def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_bit_stats(creator: DynamicParameterMovieCreator, smiles_list, radius, nBits, useFeatures, useChirality):
    b2frag, b_active, total = creator._compute_bit_fragment_stats(
        smiles_list=smiles_list, radius=radius, nBits=nBits,
        useFeatures=useFeatures, useChirality=useChirality,
    )
    coll = creator._compute_collision_metrics(b2frag, b_active, total, nBits)
    return b2frag, coll


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=500, n_jobs=1)
    model.fit(X_train, y_train)
    return model


def predict_proba_safe(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        # fallback to decision_function if available; map to [0,1] via sigmoid
        if hasattr(model, 'decision_function'):
            s = model.decision_function(X)
            return 1 / (1 + np.exp(-s))
        preds = model.predict(X)
        return preds.astype(float)


def shap_values_binary(model, X_train, X_eval):
    # Prefer LinearExplainer for linear models
    try:
        explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
        sv = explainer.shap_values(X_eval)
        # LinearExplainer returns list for classes; take class 1
        if isinstance(sv, list):
            return np.array(sv[1])
        return np.array(sv)
    except Exception:
        pass
    # Fallback generic Explainer
    try:
        masker = shap.maskers.Independent(X_train)
        explainer = shap.Explainer(model, masker=masker)
        vals = explainer(X_eval)
        # vals.values shape: (n, features)
        if hasattr(vals, 'values'):
            return np.array(vals.values)
        return np.array(vals)
    except Exception:
        # Last resort: permutation importance signed by correlation with probas
        try:
            y_proba = predict_proba_safe(model, X_eval)
        except Exception:
            y_proba = None
        with np.errstate(all='ignore'):
            imp = permutation_importance(model, X_eval, (y_proba > 0.5).astype(int) if y_proba is not None else model.predict(X_eval), n_repeats=3, random_state=0).importances_mean
        if y_proba is not None:
            corrs = [np.corrcoef(X_eval[:, j], y_proba)[0, 1] if np.std(X_eval[:, j]) > 0 else 0.0 for j in range(X_eval.shape[1])]
            corrs = np.nan_to_num(corrs)
            signed = np.sign(corrs) * np.abs(imp)
        else:
            signed = imp
        return np.tile(signed, (X_eval.shape[0], 1))


def normalize(v, ord=1):
    v = np.array(v, dtype=float)
    if v.size == 0:
        return v
    if ord == 1:
        s = np.sum(np.abs(v))
    elif ord == 2:
        s = np.sqrt(np.sum(v * v))
    else:
        s = np.max(np.abs(v))
    return v if s == 0 else (v / s)


def to_feature_weight_dict(vec):
    return {int(i): float(w) for i, w in enumerate(vec) if abs(w) > 0}


def map_bits_to_atoms_for_mol(creator, mol, bit_weights, radius, nBits, useFeatures, useChirality):
    if mol is None or not bit_weights:
        return {}
    return creator.map_fragment_weights_to_atoms(
        mol, bit_weights, radius=radius, nBits=nBits, useFeatures=useFeatures, useChirality=useChirality
    )


def jaccard(a, b):
    A = set(a)
    B = set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return float(len(A & B) / len(A | B))


def run_sar():
    here = Path(__file__).resolve().parent
    results_dir = here / 'results'
    figs_dir = here / 'figures'
    ensure_dirs(results_dir)
    ensure_dirs(figs_dir)

    # Config
    NBITS = env_int('NBITS', 2048)
    RADIUS = env_int('RADIUS', 2)
    N_PER_CLASS = env_int('N_PER_CLASS', 50)
    N_BOOT = env_int('N_BOOTSTRAP', 5)
    ENSEMBLE_ALPHA = env_float('ENSEMBLE_ALPHA', 0.5)
    TOPK_ATOMS = env_int('TOPK_ATOMS', 8)
    USE_FEATURES = os.getenv('USEFEATURES', '0') == '1'
    USE_CHIRAL = os.getenv('USECHIRALITY', '0') == '1'

    creator = DynamicParameterMovieCreator()

    # Load data
    data_file = creator._data_path()
    df = pd.read_excel(data_file)
    # balanced subsample
    df0 = df[df['classLabel'] == 0].sample(n=N_PER_CLASS, random_state=42)
    df1 = df[df['classLabel'] == 1].sample(n=N_PER_CLASS, random_state=42)
    dfb = pd.concat([df0, df1], ignore_index=True)

    smiles = dfb['cleanedMol'].astype(str).values
    y = dfb['classLabel'].astype(int).values

    # Bit stats/collision metrics on all sampled smiles
    bit2frag, coll_metrics = compute_bit_stats(creator, smiles, RADIUS, NBITS, USE_FEATURES, USE_CHIRAL)

    # Featurize
    X = creator.featurize(smiles, radius=RADIUS, nBits=NBITS, useFeatures=USE_FEATURES, useChirality=USE_CHIRAL)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, smiles, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    y_proba = predict_proba_safe(model, X_test)
    y_pred = (y_proba > 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = None

    # Save confusion matrix figure
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=colsize(single=True))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (acc={acc:.3f}, auc={auc if auc is not None else float("nan"):.3f})')
    save_figure(fig, figs_dir / 'confusion_matrix.png', dpi=1200, formats=("tiff","png","pdf"))

    # SHAP values for test set
    shap_vals = shap_values_binary(model, X_train, X_test)  # shape (n_test, NBITS)
    # LIME values for a subset of test molecules
    subset_idx = list(range(min(20, X_test.shape[0])))
    lime_mat = np.zeros_like(shap_vals)

    # prepare LIME explainer and prediction function
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=[f"bit_{i}" for i in range(X_train.shape[1])],
        class_names=['inactive', 'active'],
        discretize_continuous=False,
        sample_around_instance=True,
        mode='classification'
    )

    def predict_fn(xx):
        return np.vstack([1 - predict_proba_safe(model, xx), predict_proba_safe(model, xx)]).T

    for i in subset_idx:
        exp = explainer.explain_instance(X_test[i], predict_fn, num_features=min(50, NBITS))
        # LIME returns mapping for class 1 by default in binary classification via as_map()[1]
        wmap = dict(exp.as_map().get(1, []))  # {feature_index: weight}
        for j, w in wmap.items():
            lime_mat[i, int(j)] = float(w)

    # Ensemble weights (alpha*shap + (1-alpha)*lime) after normalization per-sample
    ens_vals = np.zeros_like(shap_vals)
    for i in range(shap_vals.shape[0]):
        s = normalize(shap_vals[i], ord=1)
        l = normalize(lime_mat[i], ord=1)
        ens_vals[i] = ENSEMBLE_ALPHA * s + (1 - ENSEMBLE_ALPHA) * l

    # Fragment enrichment vs activity (bit presence correlation + mutual information proxy)
    bit_presence = X  # 0/1
    # Pearson correlation of each bit with label
    corrs = []
    for j in range(NBITS):
        xj = bit_presence[:, j]
        if np.std(xj) == 0:
            corrs.append(0.0)
        else:
            c = np.corrcoef(xj, y)[0, 1]
            corrs.append(0.0 if np.isnan(c) else float(c))
    df_enrich = pd.DataFrame({
        'bit': np.arange(NBITS, dtype=int),
        'pearson_corr': corrs,
    })
    df_enrich['abs_corr'] = df_enrich['pearson_corr'].abs()
    df_enrich.sort_values('abs_corr', ascending=False, inplace=True)
    df_enrich.head(1000).to_csv(results_dir / 'fragment_enrichment_top.csv', index=False)

    # Global SAR ranking agreement (Spearman) between methods on test set
    # rank by mean absolute attribution across test samples
    meanabs_perm = np.abs(permutation_importance(model, X_test, y_test, n_repeats=5, random_state=0).importances_mean)
    meanabs_shap = np.mean(np.abs(shap_vals), axis=0)
    meanabs_lime = np.mean(np.abs(lime_mat), axis=0)
    meanabs_ens = np.mean(np.abs(ens_vals), axis=0)

    def rank(vec):
        # Higher value -> better rank (1 is the top)
        order = np.argsort(-vec)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(vec) + 1)
        return ranks

    r_perm = rank(meanabs_perm)
    r_shap = rank(meanabs_shap)
    r_lime = rank(meanabs_lime)
    r_ens = rank(meanabs_ens)

    def spearman(a, b):
        a = a.astype(float)
        b = b.astype(float)
        n = len(a)
        if n == 0:
            return 0.0
        dif2 = np.sum((a - b) ** 2)
        return 1 - (6 * dif2) / (n * (n * n - 1))

    method_agreement = {
        'spearman': {
            'perm_shap': float(spearman(r_perm, r_shap)),
            'perm_lime': float(spearman(r_perm, r_lime)),
            'perm_ens': float(spearman(r_perm, r_ens)),
            'shap_lime': float(spearman(r_shap, r_lime)),
            'shap_ens': float(spearman(r_shap, r_ens)),
            'lime_ens': float(spearman(r_lime, r_ens)),
        }
    }
    with open(results_dir / 'method_agreement.json', 'w') as f:
        json.dump(method_agreement, f, indent=2)

    # ------------------------- ROAR/KAR Faithfulness -------------------------
    # Use ensemble mean |attribution| ranking to define important bits
    top_order = np.argsort(-meanabs_ens)
    frac_list = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    roar_aucs, kar_aucs = [], []
    baseline_auc = float(auc) if auc is not None else float(roc_auc_score(y_test, y_proba))

    nbits_idx = np.arange(NBITS)
    for f in frac_list:
        k = max(1, int(round(f * NBITS)))
        sel = top_order[:k]
        not_sel = np.setdiff1d(nbits_idx, sel, assume_unique=False)
        # ROAR: remove (zero out) selected bits
        Xtr_roar = X_train.copy()
        Xte_roar = X_test.copy()
        Xtr_roar[:, sel] = 0
        Xte_roar[:, sel] = 0
        mr = train_model(Xtr_roar, y_train)
        proba_roar = predict_proba_safe(mr, Xte_roar)
        try:
            roar_aucs.append(float(roc_auc_score(y_test, proba_roar)))
        except Exception:
            roar_aucs.append(float('nan'))
        # KAR: keep only selected bits (zero others)
        Xtr_kar = np.zeros_like(X_train)
        Xte_kar = np.zeros_like(X_test)
        Xtr_kar[:, sel] = X_train[:, sel]
        Xte_kar[:, sel] = X_test[:, sel]
        mk = train_model(Xtr_kar, y_train)
        proba_kar = predict_proba_safe(mk, Xte_kar)
        try:
            kar_aucs.append(float(roc_auc_score(y_test, proba_kar)))
        except Exception:
            kar_aucs.append(float('nan'))

    # Plot ROAR curve
    fig, ax = plt.subplots(figsize=colsize(single=True))
    ax.plot(frac_list * 100, roar_aucs, marker='o', label='ROAR (remove top-k)')
    ax.axhline(baseline_auc, color='gray', linestyle='--', linewidth=1, label=f'Baseline AUC={baseline_auc:.3f}')
    ax.set_xlabel('Top-k bits removed (% of features)')
    ax.set_ylabel('AUC')
    ax.set_title('ROAR faithfulness curve (Ensemble ranking)')
    ax.legend(frameon=False)
    save_figure(fig, figs_dir / 'roar_curve.png', dpi=1200, formats=("tiff","png","pdf"))

    # Plot KAR curve
    fig, ax = plt.subplots(figsize=colsize(single=True))
    ax.plot(frac_list * 100, kar_aucs, marker='o', color='#4ECDC4', label='KAR (keep top-k)')
    ax.axhline(baseline_auc, color='gray', linestyle='--', linewidth=1, label=f'Baseline AUC={baseline_auc:.3f}')
    ax.set_xlabel('Top-k bits kept (% of features)')
    ax.set_ylabel('AUC')
    ax.set_title('KAR faithfulness curve (Ensemble ranking)')
    ax.legend(frameon=False)
    save_figure(fig, figs_dir / 'kar_curve.png', dpi=1200, formats=("tiff","png","pdf"))

    with open(results_dir / 'roar_kar.json', 'w') as f:
        json.dump({
            'fractions': frac_list.tolist(),
            'baseline_auc': baseline_auc,
            'roar_auc': roar_aucs,
            'kar_auc': kar_aucs,
            'ranking': 'ensemble_mean_abs'
        }, f, indent=2)

    # Plot top fragments by SHAP
    top_idx = np.argsort(-meanabs_shap)[:20]
    fig, ax = plt.subplots(figsize=colsize(single=False))
    sns.barplot(x=meanabs_shap[top_idx], y=[f"bit_{i}" for i in top_idx], orient='h', color='#4ECDC4', ax=ax)
    ax.set_title('Top-20 fragments (mean |SHAP|)')
    ax.set_xlabel('Mean |SHAP|')
    ax.set_ylabel('Bit')
    save_figure(fig, figs_dir / 'top_fragments_shap.png', dpi=1200, formats=("tiff","png","pdf"))

    # Collision sensitivity: relationship between per-bit fragment count and attribution magnitude
    per_bit_frag_counts = coll_metrics['per_bit_fragment_counts']
    counts = np.array([per_bit_frag_counts.get(i, 0) for i in range(NBITS)], dtype=float)
    fig, ax = plt.subplots(figsize=colsize(single=True))
    ax.scatter(np.log1p(counts), np.log1p(meanabs_shap + 1e-12), s=8, alpha=0.5)
    ax.set_xlabel('log(Fragments mapped to bit + 1)')
    ax.set_ylabel('log(mean |SHAP| + 1e-12)')
    ax.set_title('Collision sensitivity: fragments per bit vs |SHAP|')
    save_figure(fig, figs_dir / 'collision_vs_weight.png', dpi=1200, formats=("tiff","png","pdf"))

    # Stability via bootstrap retraining on top-k atoms for a fixed molecule
    target_smiles = s_test[0]
    target_mol = Chem.MolFromSmiles(target_smiles)
    top_sets = []
    for b in range(N_BOOT):
        # bootstrap train
        idx = np.random.RandomState(100 + b).choice(np.arange(len(X_train)), size=len(X_train), replace=True)
        Xb, yb = X_train[idx], y_train[idx]
        mb = train_model(Xb, yb)
        # SHAP for target
        sh_target = shap_values_binary(mb, Xb, X_test[:1])[0]  # 1 x NBITS -> NBITS
        # keep only active bits in target
        x_t = creator.featurize([target_smiles], RADIUS, NBITS, USE_FEATURES, USE_CHIRAL)[0]
        active = np.where(x_t > 0)[0]
        fw = {int(j): float(sh_target[j]) for j in active if sh_target[j] != 0}
        # collision adjust
        fw_adj = creator._apply_collision_adjusted_attributions(fw, per_bit_frag_counts)
        atom_w = map_bits_to_atoms_for_mol(creator, target_mol, fw_adj, RADIUS, NBITS, USE_FEATURES, USE_CHIRAL)
        # top-k atoms by |weight|
        if atom_w:
            ranked_atoms = sorted(atom_w.items(), key=lambda t: abs(t[1]), reverse=True)
            top = [a for a, w in ranked_atoms[:TOPK_ATOMS]]
        else:
            top = []
        top_sets.append(top)

    # Jaccard stability across all pairs
    if len(top_sets) >= 2:
        pairs = list(itertools.combinations(range(len(top_sets)), 2))
        j_scores = [jaccard(top_sets[i], top_sets[j]) for i, j in pairs]
        stability = {'jaccard_mean': float(np.mean(j_scores)), 'jaccard_std': float(np.std(j_scores, ddof=1)) if len(j_scores) > 1 else 0.0}
    else:
        stability = {'jaccard_mean': 0.0, 'jaccard_std': 0.0}
    with open(results_dir / 'stability.json', 'w') as f:
        json.dump(stability, f, indent=2)

    # Plot stability distribution
    if len(top_sets) >= 2:
        pairs = list(itertools.combinations(range(len(top_sets)), 2))
        fig, ax = plt.subplots(figsize=colsize(single=True))
        ax.hist([jaccard(top_sets[i], top_sets[j]) for i, j in pairs], bins=10, range=(0, 1), color='#45B7D1', alpha=0.8)
        ax.set_xlabel('Jaccard (top-k atoms)')
        ax.set_ylabel('Count')
        ax.set_title('Bootstrap stability')
        save_figure(fig, figs_dir / 'stability_jaccard.png', dpi=1200, formats=("tiff","png","pdf"))

    # Perturbation/masking test: remove top positive atoms and measure Δprob
    deltas = []
    for idx in subset_idx:
        smi = s_test[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        sh = shap_vals[idx]
        x_i = X_test[idx]
        active = np.where(x_i > 0)[0]
        fw = {int(j): float(sh[j]) for j in active if sh[j] != 0}
        fw_adj = creator._apply_collision_adjusted_attributions(fw, per_bit_frag_counts)
        atom_w = map_bits_to_atoms_for_mol(creator, mol, fw_adj, RADIUS, NBITS, USE_FEATURES, USE_CHIRAL)
        if not atom_w:
            continue
        # select top positive atoms
        pos_atoms = [a for a, w in sorted(atom_w.items(), key=lambda t: t[1], reverse=True) if w > 0][:TOPK_ATOMS]
        # probability before
        p0 = float(predict_proba_safe(model, X_test[idx:idx+1])[0])
        # remove atoms and recompute proba
        try:
            # Build new mol by removing pos_atoms
            rwm = Chem.RWMol(mol)
            for a in sorted(set(pos_atoms), reverse=True):
                if 0 <= a < rwm.GetNumAtoms():
                    rwm.RemoveAtom(a)
            newmol = rwm.GetMol()
            Chem.SanitizeMol(newmol)
            if newmol.GetNumAtoms() == 0:
                continue
            smi_new = Chem.MolToSmiles(newmol)
            Xi = creator.featurize([smi_new], RADIUS, NBITS, USE_FEATURES, USE_CHIRAL)
            p1 = float(predict_proba_safe(model, Xi)[0])
            deltas.append(max(0.0, p0 - p1))
        except Exception:
            continue
    if deltas:
        fig, ax = plt.subplots(figsize=colsize(single=True))
        sns.violinplot(y=deltas, color='#FF6B6B', ax=ax)
        ax.set_ylabel('Δ prob (remove top + atoms)')
        ax.set_title('Directional faithfulness (proxy)')
        save_figure(fig, figs_dir / 'masking_faithfulness.png', dpi=1200, formats=("tiff","png","pdf"))
        with open(results_dir / 'masking_faithfulness.json', 'w') as f:
            json.dump({'delta_mean': float(np.mean(deltas)), 'delta_std': float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0}, f, indent=2)

    # Save summary JSONs
    summary = {
        'config': {
            'NBITS': NBITS, 'RADIUS': RADIUS, 'N_PER_CLASS': N_PER_CLASS,
            'N_BOOTSTRAP': N_BOOT, 'ENSEMBLE_ALPHA': ENSEMBLE_ALPHA,
            'USE_FEATURES': USE_FEATURES, 'USE_CHIRALITY': USE_CHIRAL
        },
        'performance': {'accuracy': acc, 'roc_auc': auc},
        'collision_metrics': {
            'collision_rate_mean': coll_metrics.get('collision_rate_mean', 0.0),
            'avg_frags_per_active_bit': coll_metrics.get('avg_frags_per_active_bit', 0.0),
            'bit_entropy_mean': coll_metrics.get('bit_entropy_mean', 0.0),
        }
    }
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("Done. Results in:", results_dir)
    print("Figures in:", figs_dir)


if __name__ == '__main__':
    run_sar()
