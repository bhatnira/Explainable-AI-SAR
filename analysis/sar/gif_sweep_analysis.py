#!/usr/bin/env python3
"""
GIF Sweep Analysis for Circular Fingerprints
- Replicates the exact parameter iteration schedule used by the GIF creator
- For each iteration, trains a model, computes performance + collision metrics
- Evaluates explanation quality proxies for LIME, SHAP-local, and Ensemble
- Selects best parameters and interpretation type prioritizing AUC, then minimal collisions
Outputs:
- analysis/sar/results/gif_sweep_metrics.csv (per-iteration metrics)
- analysis/sar/results/best_outcomes.json (best per method and overall)
"""
import os
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance

import shap
from lime.lime_tabular import LimeTabularExplainer
from rdkit import Chem

from visualization.create_dynamic_parameter_movies import DynamicParameterMovieCreator

# ---- Helpers ----
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


def predict_proba_safe(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
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


def map_bits_to_atoms_for_mol(creator, mol, bit_weights, radius, nBits, useFeatures, useChirality):
    if mol is None or not bit_weights:
        return {}
    return creator.map_fragment_weights_to_atoms(
        mol, bit_weights, radius=radius, nBits=nBits, useFeatures=useFeatures, useChirality=useChirality
    )


def directional_faithfulness_delta(creator, model, smiles, bit_weights, radius, nBits, useFeatures, useChirality, topk=8):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Probability before
    Xi = creator.featurize([smiles], radius, nBits, useFeatures, useChirality)
    p0 = float(predict_proba_safe(model, Xi)[0])
    # Collision adjust and map to atoms
    # Need per-bit fragment counts; approximate by building for this single mol
    bit_info = {}
    try:
        from rdkit.Chem import AllChem
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=nBits, bitInfo=bit_info,
            useChirality=useChirality, useFeatures=useFeatures
        )
    except Exception:
        return None
    per_bit_counts = {b: len(info) for b, info in bit_info.items()}
    bw_adj = creator._apply_collision_adjusted_attributions(bit_weights, per_bit_counts)
    atom_w = map_bits_to_atoms_for_mol(creator, mol, bw_adj, radius, nBits, useFeatures, useChirality)
    if not atom_w:
        return 0.0
    # Top positive atoms
    pos_atoms = [a for a, w in sorted(atom_w.items(), key=lambda t: t[1], reverse=True) if w > 0][:topk]
    # Remove atoms and recompute prob
    try:
        rwm = Chem.RWMol(mol)
        for a in sorted(set(pos_atoms), reverse=True):
            if 0 <= a < rwm.GetNumAtoms():
                rwm.RemoveAtom(a)
        newmol = rwm.GetMol()
        Chem.SanitizeMol(newmol)
        if newmol.GetNumAtoms() == 0:
            return 0.0
        smi_new = Chem.MolToSmiles(newmol)
        Xnew = creator.featurize([smi_new], radius, nBits, useFeatures, useChirality)
        p1 = float(predict_proba_safe(model, Xnew)[0])
        return max(0.0, p0 - p1)
    except Exception:
        return 0.0


def run_sweep():
    # Config
    MAX_ITERS = env_int('MAX_ITERS', 24)
    N_PER_CLASS = env_int('N_PER_CLASS', 50)
    ENSEMBLE_ALPHA = env_float('ENSEMBLE_ALPHA', 0.5)
    TOPK_ATOMS = env_int('TOPK_ATOMS', 8)

    here = Path(__file__).resolve().parent
    results_dir = here / 'results'
    ensure_dirs(results_dir)

    creator = DynamicParameterMovieCreator()
    # Align sampling and iteration schedule to the GIF creator
    os.environ['N_PER_CLASS'] = str(N_PER_CLASS)
    creator.max_iters = MAX_ITERS
    creator.n_per_class_env = N_PER_CLASS

    # Load once
    df = creator.load_and_sample_data()
    smiles_all = df['cleanedMol'].astype(str).values
    labels_all = df['classLabel'].astype(int).values

    records = []

    for iteration in range(MAX_ITERS):
        params = creator.get_parameters_for_iteration('circular_fingerprint', iteration)
        radius = int(params['radius'])
        nBits = int(params['nBits'])
        useFeatures = bool(params.get('useFeatures', False))
        useChirality = bool(params.get('useChirality', False))

        # Collision stats on the sampled dataset
        b2frag, b_active, total = creator._compute_bit_fragment_stats(
            smiles_list=smiles_all, radius=radius, nBits=nBits,
            useFeatures=useFeatures, useChirality=useChirality
        )
        coll = creator._compute_collision_metrics(b2frag, b_active, total, nBits)

        # Featurize and split
        X = creator.featurize(smiles_all, radius, nBits, useFeatures, useChirality)
        y = labels_all
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X, y, smiles_all, test_size=0.2, random_state=42, stratify=y
        )

        # Train baseline
        model = LogisticRegression(max_iter=500, n_jobs=1)
        model.fit(X_tr, y_tr)

        # Performance
        y_proba = predict_proba_safe(model, X_te)
        y_pred = (y_proba > 0.5).astype(int)
        acc = float(accuracy_score(y_te, y_pred))
        try:
            auc = float(roc_auc_score(y_te, y_proba))
        except Exception:
            auc = None

        # Prepare explanations for a small subset for speed
        subset_n = min(20, X_te.shape[0])
        idxs = list(range(subset_n))

        # SHAP values
        shap_vals = shap_values_binary(model, X_tr, X_te[idxs])  # (subset_n, nBits)

        # LIME values
        lime_mat = np.zeros_like(shap_vals)
        explainer = LimeTabularExplainer(
            X_tr,
            feature_names=[f"bit_{i}" for i in range(X_tr.shape[1])],
            class_names=['inactive', 'active'],
            discretize_continuous=False,
            sample_around_instance=True,
            mode='classification'
        )
        def predict_fn(xx):
            p = predict_proba_safe(model, xx)
            return np.vstack([1 - p, p]).T
        for i in range(subset_n):
            exp = explainer.explain_instance(X_te[i], predict_fn, num_features=min(50, nBits))
            wmap = dict(exp.as_map().get(1, []))
            for j, w in wmap.items():
                lime_mat[i, int(j)] = float(w)

        # Ensemble (normalize per-sample)
        ens_vals = np.zeros_like(shap_vals)
        for i in range(subset_n):
            s = normalize(shap_vals[i], ord=1)
            l = normalize(lime_mat[i], ord=1)
            ens_vals[i] = ENSEMBLE_ALPHA * s + (1 - ENSEMBLE_ALPHA) * l

        # Per-method directional faithfulness proxy (avg delta)
        def vec_to_fw(vec, x_row):
            active = np.where(x_row > 0)[0]
            return {int(j): float(vec[int(j)]) for j in active if vec[int(j)] != 0}

        lime_deltas = []
        shap_deltas = []
        ens_deltas = []
        for i in range(subset_n):
            smi = s_te[i]
            # LIME
            fw = vec_to_fw(lime_mat[i], X_te[i])
            d = directional_faithfulness_delta(creator, model, smi, fw, radius, nBits, useFeatures, useChirality, topk=TOPK_ATOMS)
            if d is not None:
                lime_deltas.append(d)
            # SHAP
            fw = vec_to_fw(shap_vals[i], X_te[i])
            d = directional_faithfulness_delta(creator, model, smi, fw, radius, nBits, useFeatures, useChirality, topk=TOPK_ATOMS)
            if d is not None:
                shap_deltas.append(d)
            # Ensemble
            fw = vec_to_fw(ens_vals[i], X_te[i])
            d = directional_faithfulness_delta(creator, model, smi, fw, radius, nBits, useFeatures, useChirality, topk=TOPK_ATOMS)
            if d is not None:
                ens_deltas.append(d)

        rec = {
            'iteration': iteration,
            'radius': radius,
            'nBits': nBits,
            'useFeatures': int(useFeatures),
            'useChirality': int(useChirality),
            'accuracy': acc,
            'auc': auc if auc is not None else np.nan,
            'collision_rate_mean': coll.get('collision_rate_mean', 0.0),
            'avg_frags_per_active_bit': coll.get('avg_frags_per_active_bit', 0.0),
            'bit_entropy_mean': coll.get('bit_entropy_mean', 0.0),
            'lime_delta_mean': float(np.mean(lime_deltas)) if lime_deltas else 0.0,
            'shap_delta_mean': float(np.mean(shap_deltas)) if shap_deltas else 0.0,
            'ens_delta_mean': float(np.mean(ens_deltas)) if ens_deltas else 0.0,
        }
        records.append(rec)
        print(f"Iter {iteration}: AUC={rec['auc']:.3f} FRAG/bit={rec['avg_frags_per_active_bit']:.3f} Ent={rec['bit_entropy_mean']:.3f} "
              f"Δ(L)={rec['lime_delta_mean']:.3f} Δ(S)={rec['shap_delta_mean']:.3f} Δ(E)={rec['ens_delta_mean']:.3f}")

    dfm = pd.DataFrame.from_records(records)
    csv_path = results_dir / 'gif_sweep_metrics.csv'
    dfm.to_csv(csv_path, index=False)

    # Best selection per method by AUC, tie-break on collisions
    def select_best(df: pd.DataFrame, delta_col: str):
        # Primary: max AUC; window within 0.005 of max; among them: min avg_frags_per_active_bit, then min bit_entropy_mean, then min collision_rate_mean, finally max delta
        d = df.copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(subset=['auc'])
        if d.empty:
            return None
        max_auc = d['auc'].max()
        window = d[d['auc'] >= max_auc - 0.005]
        # sort by collisions ascending and -delta to prefer higher delta
        window = window.sort_values(by=['avg_frags_per_active_bit', 'bit_entropy_mean', 'collision_rate_mean', f'-{delta_col}'], ascending=[True, True, True, True])
        # Pandas can't sort by negative column; workaround
        window[delta_col + '_neg'] = -window[delta_col]
        window = window.sort_values(by=['avg_frags_per_active_bit', 'bit_entropy_mean', 'collision_rate_mean', delta_col + '_neg'], ascending=[True, True, True, True])
        row = window.iloc[0].to_dict()
        row.pop(delta_col + '_neg', None)
        return row

    best_lime = select_best(dfm, 'lime_delta_mean')
    best_shap = select_best(dfm, 'shap_delta_mean')
    best_ens = select_best(dfm, 'ens_delta_mean')

    # Overall best among those three using the same tie-break
    candidates = []
    for tag, row in [('lime', best_lime), ('shap_local', best_shap), ('ensemble', best_ens)]:
        if row:
            r = row.copy()
            r['method'] = tag
            candidates.append(r)
    over = None
    if candidates:
        dd = pd.DataFrame(candidates)
        # same window logic
        max_auc = dd['auc'].max()
        window = dd[dd['auc'] >= max_auc - 0.005]
        window = window.sort_values(by=['avg_frags_per_active_bit', 'bit_entropy_mean', 'collision_rate_mean'], ascending=True)
        over = window.iloc[0].to_dict()

    out = {
        'best_per_method': {
            'lime': best_lime,
            'shap_local': best_shap,
            'ensemble': best_ens,
        },
        'overall_best': over,
    }

    with open(results_dir / 'best_outcomes.json', 'w') as f:
        json.dump(out, f, indent=2)

    print("Saved:", csv_path)
    print("Saved:", results_dir / 'best_outcomes.json')


if __name__ == '__main__':
    run_sweep()
