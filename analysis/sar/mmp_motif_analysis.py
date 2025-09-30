#!/usr/bin/env python3
"""
MMP and Motif Enrichment Analysis
- Motif enrichment via SMARTS patterns (odds ratios, chi-squared p-values)
- Attribution-motif overlap (fraction of top-k attributed atoms that hit motifs)
- MMP consistency: For scaffold-matched opposite-label pairs, compute MCS-difference regions and
  measure whether top attributions hit the changed region (and direction consistency for actives)

Outputs:
- analysis/sar/results/mmp_motif_summary.json
- analysis/sar/figures/motif_enrichment_bar.png
- analysis/sar/figures/motif_overlap_violin.png
- analysis/sar/figures/mmp_consistency_bar.png

Env:
- NBITS, RADIUS, USEFEATURES, USECHIRALITY (defaults from fast_sweep_best.json)
- N_PER_CLASS (default 60)
- EXPLAIN_SUBSET (default 120)
- TOPK_ATOMS (default 12)
- ENSEMBLE_ALPHA (default 0.5)
- MAX_MMP_PAIRS (default 200)
- MCS_TIMEOUT (sec, default 2)
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

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

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

# Reuse project code
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from visualization.create_dynamic_parameter_movies import DynamicParameterMovieCreator

# ----------------- Utils -----------------
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


def normalize(v: np.ndarray):
    v = np.asarray(v, dtype=float)
    s = np.sum(np.abs(v))
    return v if s == 0 else v / s

# Simple chi-squared p-value for 2x2 table
# table = [[a, b],[c, d]] with continuity correction

def chi2_yates_p(a,b,c,d):
    import math
    n = a+b+c+d
    if n == 0:
        return 1.0
    row1 = a+b; row2 = c+d
    col1 = a+c; col2 = b+d
    exp_a = row1*col1/n
    exp_b = row1*col2/n
    exp_c = row2*col1/n
    exp_d = row2*col2/n
    # chi2 with Yates correction
    chi2 = 0.0
    for obs, exp in ((a,exp_a),(b,exp_b),(c,exp_c),(d,exp_d)):
        if exp > 0:
            chi2 += (abs(obs-exp)-0.5)**2/exp
    # 1 df; approximate p-value via survival function of chi2
    # use math.erfc approximation for df=1: p ~ erfc(sqrt(chi2/2))
    try:
        p = math.erfc((chi2/2.0)**0.5)
    except Exception:
        p = 1.0
    return float(p)

# ----------------- Motifs -----------------
MOTIFS = [
    ("AromaticRing", "a"),
    ("Pyridine", "n1cccc1"),
    ("Amide", "C(=O)N"),
    ("CarboxylicAcid", "C(=O)[O;H1,-]"),
    ("Sulfonamide", "S(=O)(=O)N"),
    ("Nitro", "[N+](=O)[O-]"),
    ("Halogen", "[F,Cl,Br,I]"),
    ("Phenol", "c[OH]"),
    ("Thiol", "[SH]"),
    ("QuaternaryAmmonium", "[N+](C)(C)C"),
    ("HBDonor", "[N,O;H1]"),
    ("HBAcceptor", "[#7,#8,#16;X2]"),
]
MOTIF_SMARTS = [(name, Chem.MolFromSmarts(smi)) for name, smi in MOTIFS]

# ----------------- Core -----------------
def map_bits_to_top_atoms(creator: DynamicParameterMovieCreator, smi: str, bit_weights: Dict[int,float],
                          radius:int, nBits:int, useFeatures:bool, useChirality:bool, topk:int) -> List[int]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    atoms_w = creator.map_fragment_weights_to_atoms(mol, bit_weights, radius=radius, nBits=nBits,
                                                    useFeatures=useFeatures, useChirality=useChirality)
    if not atoms_w:
        return []
    order = sorted(atoms_w.items(), key=lambda x: -abs(x[1]))
    return [int(idx) for idx, _ in order[:topk]]


def compute_explanations(X_train, X_eval, model, method: str, ENSEMBLE_ALPHA: float = 0.5):
    import shap as _shap
    from lime.lime_tabular import LimeTabularExplainer as _Lime

    def normalize_rowwise(A):
        return np.vstack([normalize(A[i]) for i in range(A.shape[0])])

    if method == 'SHAP' or method == 'ENSEMBLE':
        try:
            explainer = _shap.LinearExplainer(model, X_train, feature_dependence='independent')
            sv = explainer.shap_values(X_eval)
            shap_vals = np.array(sv[1]) if isinstance(sv, list) else np.array(sv)
        except Exception:
            masker = _shap.maskers.Independent(X_train)
            e = _shap.Explainer(model, masker=masker)
            vals = e(X_eval)
            shap_vals = np.array(vals.values) if hasattr(vals, 'values') else np.array(vals)
        shap_norm = normalize_rowwise(shap_vals)
    if method == 'LIME' or method == 'ENSEMBLE':
        explainer = _Lime(
            X_train,
            feature_names=[f'bit_{i}' for i in range(X_train.shape[1])],
            class_names=['inactive','active'],
            discretize_continuous=False,
            sample_around_instance=True,
            mode='classification')
        def predict_fn(xx):
            p = model.predict_proba(xx)[:,1]
            return np.vstack([1-p,p]).T
        lime_vals = np.zeros_like(X_eval, dtype=float)
        for i in range(X_eval.shape[0]):
            exp = explainer.explain_instance(X_eval[i], predict_fn, num_features=min(40, X_eval.shape[1]), num_samples=350)
            wmap = dict(exp.as_map().get(1, []))
            for j, w in wmap.items():
                lime_vals[i, int(j)] = float(w)
        lime_norm = normalize_rowwise(lime_vals)
    if method == 'SHAP':
        return shap_norm
    if method == 'LIME':
        return lime_norm
    # Ensemble
    return ENSEMBLE_ALPHA * shap_norm + (1 - ENSEMBLE_ALPHA) * lime_norm


def motif_enrichment(smiles: List[str], labels: List[int]) -> Tuple[pd.DataFrame, Dict[str, Dict[str,float]]]:
    rows = []
    enrich = {}
    for name, patt in MOTIF_SMARTS:
        if patt is None:
            continue
        has = []
        for smi in smiles:
            m = Chem.MolFromSmiles(smi)
            has.append(1 if (m is not None and m.HasSubstructMatch(patt)) else 0)
        has = np.array(has, dtype=int)
        y = np.array(labels, dtype=int)
        a = int(((y==1) & (has==1)).sum())
        b = int(((y==1) & (has==0)).sum())
        c = int(((y==0) & (has==1)).sum())
        d = int(((y==0) & (has==0)).sum())
        # odds ratio (add 0.5 to avoid div by zero)
        or_val = ((a+0.5)*(d+0.5))/((b+0.5)*(c+0.5))
        pval = chi2_yates_p(a,b,c,d)
        rows.append({'motif': name, 'a_active_has': a, 'b_active_no': b, 'c_inactive_has': c, 'd_inactive_no': d,
                     'odds_ratio': or_val, 'p_value': pval})
        enrich[name] = {'odds_ratio': float(or_val), 'p_value': float(pval)}
    df = pd.DataFrame(rows).sort_values('odds_ratio', ascending=False)
    return df, enrich


def motif_overlap(smiles: List[str], top_atoms: Dict[int, List[int]]) -> pd.DataFrame:
    # returns per-mol counts of overlaps across motifs
    data = []
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        atoms = set(top_atoms.get(i, []))
        row = {'idx': i}
        for name, patt in MOTIF_SMARTS:
            if m is None or patt is None:
                row[name] = 0
                continue
            matches = m.GetSubstructMatches(patt)
            motif_atoms = set(a for match in matches for a in match)
            row[name] = int(len(atoms & motif_atoms))
        data.append(row)
    return pd.DataFrame(data)


def choose_mmp_pairs(smiles: List[str], labels: List[int], max_pairs: int, seed: int = 123):
    # Pair molecules with same Murcko scaffold but different labels
    scaf_to_idx: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        scaf = bemis_murcko_scaffold_smi(smi)
        scaf_to_idx.setdefault(scaf, []).append(i)
    rng = np.random.RandomState(seed)
    pairs = []
    for scaf, idxs in scaf_to_idx.items():
        pos = [i for i in idxs if labels[i] == 1]
        neg = [i for i in idxs if labels[i] == 0]
        if not pos or not neg:
            continue
        rng.shuffle(pos); rng.shuffle(neg)
        for i in pos:
            for j in neg:
                pairs.append((i, j))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def mcs_diff_atoms(smi_a: str, smi_b: str, timeout: int = 2) -> Tuple[set, set]:
    ma = Chem.MolFromSmiles(smi_a)
    mb = Chem.MolFromSmiles(smi_b)
    if ma is None or mb is None:
        return set(), set()
    res = rdFMCS.FindMCS([ma, mb], timeout=timeout, completeRingsOnly=True)
    if not res or not res.smartsString:
        return set(), set()
    mcs = Chem.MolFromSmarts(res.smartsString)
    match_a = ma.GetSubstructMatch(mcs)
    match_b = mb.GetSubstructMatch(mcs)
    set_a = set(range(ma.GetNumAtoms())) - set(match_a)
    set_b = set(range(mb.GetNumAtoms())) - set(match_b)
    return set_a, set_b


def run():
    here = Path(__file__).resolve().parent
    figs_dir = here / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = here / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    NBITS_DEF, RADIUS_DEF, FEAT_DEF, CHIR_DEF = load_best_defaults(here)
    NBITS = env_int('NBITS', NBITS_DEF)
    RADIUS = env_int('RADIUS', RADIUS_DEF)
    USE_FEATURES = env_int('USEFEATURES', FEAT_DEF)
    USE_CHIRAL = env_int('USECHIRALITY', CHIR_DEF)
    N_PER_CLASS = env_int('N_PER_CLASS', 60)
    EXPL_SUB = env_int('EXPLAIN_SUBSET', 120)
    TOPK_ATOMS = env_int('TOPK_ATOMS', 12)
    ENSEMBLE_ALPHA = env_float('ENSEMBLE_ALPHA', 0.5)
    MAX_MMP = env_int('MAX_MMP_PAIRS', 200)
    MCS_TIMEOUT = env_int('MCS_TIMEOUT', 2)

    creator = DynamicParameterMovieCreator()
    df = pd.read_excel(creator._data_path())
    df0 = df[df['classLabel'] == 0].sample(n=N_PER_CLASS, random_state=42)
    df1 = df[df['classLabel'] == 1].sample(n=N_PER_CLASS, random_state=42)
    dfb = pd.concat([df0, df1], ignore_index=True)
    smiles = dfb['cleanedMol'].astype(str).values
    y = dfb['classLabel'].astype(int).values

    # Featurize & model
    X = creator.featurize(smiles, RADIUS, NBITS, bool(USE_FEATURES), bool(USE_CHIRAL))
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(X, y, smiles, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=500, n_jobs=1, solver='lbfgs').fit(X_tr, y_tr)
    y_proba = clf.predict_proba(X_te)[:,1]
    auc = float(roc_auc_score(y_te, y_proba)); acc = float(((y_proba>0.5)==y_te).mean())

    # Explanations on subset
    m = min(EXPL_SUB, X_te.shape[0])
    X_eval = X_te[:m]
    s_eval = s_te[:m]
    methods = ['SHAP','LIME','ENSEMBLE']
    expl = {}
    for method in methods:
        expl[method] = compute_explanations(X_tr, X_eval, clf, method, ENSEMBLE_ALPHA)

    # Map to top atoms per method
    top_atoms = {meth: {} for meth in methods}
    for meth in methods:
        A = expl[meth]
        for i in range(A.shape[0]):
            bit_weights = {int(j): float(A[i, j]) for j in np.where(np.abs(A[i])>0)[0]}
            top_atoms[meth][i] = map_bits_to_top_atoms(creator, s_eval[i], bit_weights,
                                                       RADIUS, NBITS, bool(USE_FEATURES), bool(USE_CHIRAL), TOPK_ATOMS)

    # Motif enrichment
    df_enrich, enrich = motif_enrichment(list(s_eval), list(y_te[:m]))
    # Plot OR bar (top 12)
    topN = min(12, len(df_enrich))
    fig, ax = plt.subplots(figsize=colsize(single=False))
    sns.barplot(data=df_enrich.head(topN), x='motif', y='odds_ratio', color='#3A86FF', ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Odds Ratio (active vs inactive)')
    ax.set_title('Motif enrichment')
    save_figure(fig, figs_dir / 'motif_enrichment_bar.png', dpi=1200, formats=("tiff","png","pdf"))

    # Motif overlap violin (per method, total overlaps across motifs)
    overlap_totals = []
    for meth in methods:
        df_ov = motif_overlap(list(s_eval), top_atoms[meth])
        totals = df_ov.drop(columns=['idx']).sum(axis=1)
        overlap_totals.append(pd.DataFrame({'method': meth, 'overlap_count': totals}))
    df_tot = pd.concat(overlap_totals, ignore_index=True)
    fig, ax = plt.subplots(figsize=colsize(single=True))
    sns.violinplot(data=df_tot, x='method', y='overlap_count', inner='box', ax=ax)
    ax.set_title('Total motif-atom overlaps (top-k atoms)')
    save_figure(fig, figs_dir / 'motif_overlap_violin.png', dpi=1200, formats=("tiff","png","pdf"))

    # MMP pairs and consistency
    pairs = choose_mmp_pairs(list(s_eval), list(y_te[:m]), max_pairs=MAX_MMP, seed=123)
    hit_rates = {m: [] for m in methods}
    for (i, j) in pairs:
        sa, sb = s_eval[i], s_eval[j]
        ya = int(y_te[:m][i]); yb = int(y_te[:m][j])
        if ya == yb:
            continue
        diff_a, diff_b = mcs_diff_atoms(sa, sb, timeout=MCS_TIMEOUT)
        # which is active?
        if ya == 1:
            active_idx, inactive_idx = i, j; active_diff = diff_a; inactive_diff = diff_b
        else:
            active_idx, inactive_idx = j, i; active_diff = diff_b; inactive_diff = diff_a
        for meth in methods:
            act_atoms = set(top_atoms[meth].get(active_idx, []))
            inact_atoms = set(top_atoms[meth].get(inactive_idx, []))
            # hits if top atoms intersect changed region of corresponding mol
            hit_act = 1 if act_atoms & active_diff else 0
            hit_inact = 1 if inact_atoms & inactive_diff else 0
            hit_rates[meth].append((hit_act, hit_inact))
    # Aggregate hit rates
    mmp_summary = {}
    for meth in methods:
        hits = hit_rates[meth]
        if not hits:
            mmp_summary[meth] = {'active_hit_rate': 0.0, 'inactive_hit_rate': 0.0, 'n_pairs': 0}
        else:
            ah = float(np.mean([h[0] for h in hits])); ih = float(np.mean([h[1] for h in hits]))
            mmp_summary[meth] = {'active_hit_rate': ah, 'inactive_hit_rate': ih, 'n_pairs': len(hits)}
    # Plot
    fig, ax = plt.subplots(figsize=colsize(single=True))
    xs = np.arange(len(methods)); vals = [mmp_summary[m]['active_hit_rate'] for m in methods]
    sns.barplot(x=methods, y=vals, palette=['#4ECDC4','#FFBE0B','#3A86FF'], ax=ax)
    ax.set_ylabel('Active changed-region hit rate')
    ax.set_title('MMP attribution consistency')
    ax.set_ylim(0,1)
    save_figure(fig, figs_dir / 'mmp_consistency_bar.png', dpi=1200, formats=("tiff","png","pdf"))

    # Save summary JSON
    out = {
        'config': {'NBITS': NBITS, 'RADIUS': RADIUS, 'USE_FEATURES': USE_FEATURES, 'USE_CHIRALITY': USE_CHIRAL,
                   'N_PER_CLASS': N_PER_CLASS, 'EXPLAIN_SUBSET': EXPL_SUB, 'TOPK_ATOMS': TOPK_ATOMS,
                   'ENSEMBLE_ALPHA': ENSEMBLE_ALPHA, 'MAX_MMP_PAIRS': MAX_MMP, 'MCS_TIMEOUT': MCS_TIMEOUT},
        'performance_eval_split': {'acc': acc, 'auc': auc},
        'motif_enrichment': enrich,
        'mmp_consistency': mmp_summary,
    }
    with open(results_dir / 'mmp_motif_summary.json', 'w') as f:
        json.dump(out, f, indent=2)

    print('MMP & motif analysis complete.')


if __name__ == '__main__':
    run()
