#!/usr/bin/env python3
"""
Dynamic Parameter Evolution Movie Creator
=========================================

Creates movies showing ACTUAL parameter changes across iterations,
now using real TPOT AutoML training on Morgan (circular) fingerprints
instead of simulated contributions. Keeps the original layout.
"""

import os
# Suppress noisy logs and warnings as early as possible
os.environ.setdefault("PYTHONWARNINGS", "ignore::urllib3.exceptions.NotOpenSSLWarning")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_STDERR_THRESHOLD", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("RUST_LOG", "error")
os.environ.setdefault("RUST_BACKTRACE", "0")
# Reduce thread contention messages
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")
# Reduce XGBoost verbosity if used by TPOT
os.environ.setdefault("XGBOOST_VERBOSITY", "0")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import time
import json
import warnings
# Suppress urllib3 LibreSSL warning early using message-based filter (does not import urllib3)
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1.1.1\+.*",
)
import contextlib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# Add colormap utilities to match chemberta_attention.png style
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import PandasTools
# Silence RDKit logs (e.g., 'No normalization for Phi. Feature removed!')
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
except Exception:
    pass
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
# Silence urllib3 NotOpenSSLWarning on macOS LibreSSL Python builds
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

@contextlib.contextmanager
def suppress_stderr():
    try:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            yield
    except Exception:
        # Fallback: no suppression
        yield

@contextlib.contextmanager
def suppress_output():
    try:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull), contextlib.redirect_stdout(devnull):
            yield
    except Exception:
        yield

@contextlib.contextmanager
def silence_fds(fds=(1, 2)):
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    saved = {}
    try:
        for fd in fds:
            try:
                saved[fd] = os.dup(fd)
                os.dup2(devnull_fd, fd)
            except OSError:
                pass
        yield
    finally:
        for fd, sv in saved.items():
            try:
                os.dup2(sv, fd)
                os.close(sv)
            except OSError:
                pass
        try:
            os.close(devnull_fd)
        except OSError:
            pass

# --- Lazy import handle for DeepChem to avoid loading TF when not needed ---
dc = None

# --- New: sklearn baseline imports for fallback when TPOT fails ---
from sklearn.linear_model import LogisticRegression
# --- New: extra metrics & curves ---
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, average_precision_score

class DynamicParameterMovieCreator:
    def __init__(self):
        self.colors = {
            'circular_fingerprint': '#FF6B6B',
            'chemberta': '#4ECDC4', 
            'graphconv': '#45B7D1'
        }
        
        # High-resolution scale (2x => ~2400x1600 canvas)
        self.scale = 2.0
        
        # Keep a consistent molecule across iterations (picked from dataset at runtime)
        self.target_smiles = None
        self.sampled_df = None
        
        # Define parameter exploration spaces
        self.parameter_spaces = {
            'circular_fingerprint': {
                'radius': [1, 2, 3, 4, 2, 3],
                'nBits': [1024, 2048, 4096, 2048, 1024, 2048]
            },
            # --- New: GraphConv parameter space ---
            'graphconv': {
                'hidden_dim': [32, 64, 128, 96],
                'num_layers': [2, 3, 4, 3],
                'dropout': [0.1, 0.2, 0.3, 0.25],
                'learning_rate': [1e-3, 5e-4, 1e-3, 7e-4]
            },
            # --- New: ChemBERTa parameter space ---
            'chemberta': {
                'learning_rate': [2e-5, 1e-5, 3e-5, 2e-5],
                'num_train_epochs': [1, 1, 2, 2],
                'max_seq_length': [128, 256, 512, 256],
                'attention_layer': [-1, -2, -1, -3],
                'attention_head': [0, 3, 5, 0]
            }
        }
        
        # Placeholders (computed dynamically now)
        self.quality_evolution = {'circular_fingerprint': [], 'graphconv': [], 'chemberta': []}
        self.performance_evolution = {'circular_fingerprint': [], 'graphconv': [], 'chemberta': []}
        # Track last AUC/AUPRC for display
        self.last_auc = {'circular_fingerprint': None, 'graphconv': None, 'chemberta': None}
        self.last_auprc = {'circular_fingerprint': None, 'graphconv': None, 'chemberta': None}

        # Runtime knobs via env vars to control memory/time
        self.use_tpot = os.getenv('USE_TPOT', '0') == '1'
        try:
            self.max_iters = int(os.getenv('MAX_ITERS', '4'))
        except Exception:
            self.max_iters = 4
        try:
            self.n_per_class_env = int(os.getenv('N_PER_CLASS', '50'))
        except Exception:
            self.n_per_class_env = 50
        # GraphConv controls
        try:
            self.graphconv_epochs = int(os.getenv('GRAPHCONV_EPOCHS', '5'))
        except Exception:
            self.graphconv_epochs = 5
        try:
            self.graphconv_batch = int(os.getenv('GRAPHCONV_BATCH', '16'))
        except Exception:
            self.graphconv_batch = 16
        # ChemBERTa controls
        try:
            self.chemberta_batch = int(os.getenv('CHEMBERTA_BATCH', '8'))
        except Exception:
            self.chemberta_batch = 8
        self.chemberta_offline = os.getenv('CHEMBERTA_OFFLINE', '0') == '1'

    # ------------------------ Data & Features ------------------------
    def _data_path(self):
        # Resolve dataset path relative to repo root
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent / 'data' / 'StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx',
            here / 'data' / 'StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx',
            Path('data') / 'StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx',
        ]
        for p in candidates:
            if p.exists():
                return p
        # Fallback: original location mentioned elsewhere
        return Path('StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx')

    def load_and_sample_data(self, n_per_class=50, random_state=42):
        if self.sampled_df is not None:
            return self.sampled_df
        # Allow override via env for memory control
        try:
            n_per_class = int(self.n_per_class_env)
        except Exception:
            pass
        data_file = self._data_path()
        print(f"📂 Loading data from: {data_file}")
        df = pd.read_excel(data_file)
        # Balanced sample
        class_0 = df[df['classLabel'] == 0].sample(n=n_per_class, random_state=random_state)
        class_1 = df[df['classLabel'] == 1].sample(n=n_per_class, random_state=random_state)
        self.sampled_df = pd.concat([class_0, class_1], ignore_index=True)
        print(f"✅ Loaded {len(self.sampled_df)} molecules (balanced sample), n_per_class={n_per_class}")
        return self.sampled_df

    def featurize(self, smiles_list, radius=2, nBits=2048, useFeatures=False, useChirality=False):
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                bv = AllChem.GetMorganFingerprintAsBitVect(
                    mol, int(radius), nBits=int(nBits), useChirality=bool(useChirality), useFeatures=bool(useFeatures)
                )
                arr = np.frombuffer(bv.ToBitString().encode('ascii'), 'S1').astype('U1').astype(int)
                fps.append(arr)
            else:
                fps.append(np.zeros(int(nBits), dtype=int))
        return np.asarray(fps, dtype=int)

    # ------------------------ Interpretation Utils ------------------------
    def generate_circular_fingerprint_dict(self, mol, radius=2, nBits=2048, useFeatures=False, useChirality=False):
        bit_info = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=nBits, bitInfo=bit_info, useChirality=useChirality, useFeatures=useFeatures
        )
        fragments = {}
        for bit_idx, info_list in bit_info.items():
            if not info_list:
                continue
            # Use the first occurrence for visualization
            atom_idx, rad = info_list[0]
            try:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
                if env:
                    submol = Chem.PathToSubmol(mol, env)
                    if submol:
                        fragments[bit_idx] = submol
            except Exception:
                continue
        return fragments

    def map_fragment_weights_to_atoms(self, parent_mol, fragment_weights, radius=2, nBits=2048, useFeatures=False, useChirality=False):
        atom_weights = {}
        fragments = self.generate_circular_fingerprint_dict(parent_mol, radius, nBits, useFeatures, useChirality)
        for bit_idx, weight in fragment_weights.items():
            submol = fragments.get(bit_idx)
            if not submol:
                continue
            try:
                matches = parent_mol.GetSubstructMatches(submol)
                for match in matches:
                    for aidx in match:
                        atom_weights[aidx] = atom_weights.get(aidx, 0.0) + float(weight)
            except Exception:
                continue
        # Normalize to [-1, 1]
        if atom_weights:
            max_abs = max(abs(v) for v in atom_weights.values()) or 1.0
            for k in list(atom_weights.keys()):
                atom_weights[k] = float(atom_weights[k] / max_abs)
        return atom_weights

    def compute_signed_feature_importance(self, model, X_test, y_test, n_repeats=5, random_state=42):
        try:
            proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            proba = None
        base_preds = model.predict(X_test)
        base_acc = accuracy_score(y_test, base_preds)
        signed_importances = np.zeros(X_test.shape[1], dtype=float)
        # Use permutation importance; sign by correlation with class label
        with suppress_stderr():
            result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
        importances = result.importances_mean  # shape (n_features,)
        for j in range(X_test.shape[1]):
            if proba is not None:
                try:
                    corr = np.corrcoef(X_test[:, j], proba)[0, 1]
                except Exception:
                    corr = 0.0
            else:
                corr = 0.0
            signed_importances[j] = float(np.sign(corr) * abs(importances[j]))
        return signed_importances

    # ------------------------ Drawing & Panels ------------------------
    def _fig_to_image(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0.05)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        buf.close()
        plt.close(fig)
        return img

    def _plot_accuracy_evolution(self, model_name):
        fig, ax = plt.subplots(figsize=(1.25 * self.scale, 0.9 * self.scale), dpi=200)
        vals = self.performance_evolution.get(model_name, [])
        ax.plot(range(1, len(vals) + 1), vals, marker='o', color=self.colors.get(model_name, '#333333'))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.0, 1.0)
        ax.set_title('Accuracy Evolution')
        ax.grid(True, alpha=0.3)
        return self._fig_to_image(fig)

    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(1.25 * self.scale, 0.9 * self.scale), dpi=200)
        im = ax.imshow(cm, cmap='Blues')
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center', color='black')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return self._fig_to_image(fig)

    def _plot_roc_pr(self, y_true, y_score):
        """Return a single combined panel image with ROC (left) and PR (right), plus AUC/AUPRC values."""
        # Handle missing probability scores
        if y_score is None or (isinstance(y_score, np.ndarray) and (np.all(np.isnan(y_score)) or y_score.size == 0)):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5 * self.scale, 0.9 * self.scale), dpi=200)
            # ROC placeholder
            ax1.plot([], [])
            ax1.set_title('ROC Curve (no probas)')
            ax1.set_xlabel('FPR')
            ax1.set_ylabel('TPR')
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            # PR placeholder
            ax2.plot([], [])
            ax2.set_title('PR Curve (no probas)')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            img = self._fig_to_image(fig)
            return img, None, None
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        try:
            auc_val = float(np.trapz(tpr, fpr))
        except Exception:
            auc_val = None
        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        try:
            ap_val = float(average_precision_score(y_true, y_score))
        except Exception:
            ap_val = None
        # Combined figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5 * self.scale, 0.9 * self.scale), dpi=200)
        # ROC subplot
        ax1.plot(fpr, tpr, label=f'AUC={auc_val:.3f}' if auc_val is not None else 'AUC=NA')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.set_title('ROC Curve')
        ax1.legend()
        # PR subplot
        ax2.plot(recall, precision, label=f'AUPRC={ap_val:.3f}' if ap_val is not None else 'AUPRC=NA')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('PR Curve')
        ax2.legend()
        img = self._fig_to_image(fig)
        return img, auc_val, ap_val

    def draw_molecule_with_dynamic_parameters(self, atom_weights, title, iteration, quality_score, performance_score, model_type, params, mol,
                                              auc_val=None, auprc_val=None, acc_img=None, cm_img=None, rocpr_img=None):
        if mol is None:
            return None
        
        s = self.scale
        drawer = rdMolDraw2D.MolDraw2DCairo(int(500 * s), int(400 * s))
        
        # Normalize contributions for coloring
        num_atoms = mol.GetNumAtoms()
        contrib_array = np.zeros(num_atoms, dtype=float)
        for idx, w in (atom_weights or {}).items():
            if 0 <= idx < num_atoms:
                contrib_array[idx] = w
        
        # Determine coloring strategy
        atom_colors = {}
        atom_radii = {}
        if model_type == 'chemberta' and num_atoms > 0 and np.any(contrib_array != 0):
            # Use RdYlBu_r colormap and radii scaling similar to chemberta_attention.py
            vmin = float(contrib_array.min())
            vmax = float(contrib_array.max())
            if vmax - vmin < 1e-12:
                vmax = vmin + 1e-12
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap('RdYlBu_r')
            for i in range(num_atoms):
                w = float(contrib_array[i])
                rgba = cmap(norm(w))
                atom_colors[i] = tuple(rgba[:3])
                # scale radius 0.3..0.7 with normalized weight
                wn = (w - vmin) / (vmax - vmin)
                atom_radii[i] = 0.3 + 0.4 * wn
        else:
            # Generic positive/negative mapping
            if num_atoms > 0:
                max_abs = max(1e-9, float(np.max(np.abs(contrib_array))))
                normalized_contribs = [0.5 + 0.5 * (c / max_abs) for c in contrib_array]
            else:
                normalized_contribs = []
            for i, contrib in enumerate(normalized_contribs):
                if contrib >= 0.55:
                    strength = (contrib - 0.55) / 0.45
                    atom_colors[i] = (0.2 * (1 - strength), 0.2 * (1 - strength), 1.0 * (0.6 + 0.4 * strength))
                elif contrib <= 0.45:
                    strength = (0.45 - contrib) / 0.45
                    atom_colors[i] = (1.0 * (0.6 + 0.4 * strength), 0.2 * (1 - strength), 0.2 * (1 - strength))
                else:
                    atom_colors[i] = (0.85, 0.85, 0.85)
        
        drawer.SetFontSize(int(16 * s))
        if atom_radii:
            drawer.DrawMolecule(mol, highlightAtoms=list(range(num_atoms)), highlightAtomColors=atom_colors, highlightAtomRadii=atom_radii)
        else:
            drawer.DrawMolecule(mol, highlightAtoms=list(range(num_atoms)), highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        canvas = Image.new('RGB', (int(1200 * s), int(800 * s)), 'white')
        canvas.paste(img, (int(50 * s), int(50 * s)))
        
        draw = ImageDraw.Draw(canvas)
        try:
            from pathlib import Path as _Path
            import matplotlib as _mpl
            _font_dir = _Path(_mpl.get_data_path()) / "fonts" / "ttf"
            _regular_path = str(_font_dir / "DejaVuSans.ttf")
            _bold_path = str(_font_dir / "DejaVuSans-Bold.ttf")
            title_font = ImageFont.truetype(_bold_path, int(26 * s))
            metric_font = ImageFont.truetype(_regular_path, int(16 * s))
            metric_font_bold = ImageFont.truetype(_bold_path, int(18 * s))
            small_font = ImageFont.truetype(_regular_path, int(15 * s))
            small_font_bold = ImageFont.truetype(_bold_path, int(16 * s))
            param_font = ImageFont.truetype(_regular_path, int(14 * s))
            param_font_bold = ImageFont.truetype(_bold_path, int(14 * s))
        except Exception:
            title_font = ImageFont.load_default()
            metric_font = ImageFont.load_default()
            metric_font_bold = ImageFont.load_default()
            small_font = ImageFont.load_default()
            small_font_bold = ImageFont.load_default()
            param_font = ImageFont.load_default()
            param_font_bold = ImageFont.load_default()
        
        # Title and model info
        draw.text((int(20 * s), int(15 * s)), title, fill='black', font=title_font)
        # Bold model label
        draw.text((int(20 * s), int(460 * s)), f"Model: {model_type.replace('_', ' ').title()}", 
                 fill='black', font=metric_font_bold)
        
        # Metrics (include AUC/AUPRC) — make Iteration bold as a key label
        draw.text((int(350 * s), int(460 * s)), f"Iteration: {iteration}", fill='black', font=metric_font_bold)
        draw.text((int(20 * s), int(480 * s)), f"Quality: {quality_score:.3f}", fill='blue', font=small_font)
        draw.text((int(170 * s), int(480 * s)), f"Accuracy: {performance_score:.3f}", fill='green', font=small_font)
        draw.text((int(330 * s), int(480 * s)), f"ROC AUC: {auc_val:.3f}" if auc_val is not None else "ROC AUC: NA", fill='darkred', font=small_font)
        draw.text((int(500 * s), int(480 * s)), f"AUPRC: {auprc_val:.3f}" if auprc_val is not None else "AUPRC: NA", fill='darkgreen', font=small_font)
        draw.text((int(330 * s), int(500 * s)), f"Combined: {0.6*performance_score + 0.4*quality_score:.3f}", 
                 fill='purple', font=small_font)
        
        # Current parameters box
        param_y_start = int(500 * s)
        draw.text((int(20 * s), param_y_start), "🔧 Current Parameters:", fill='black', font=small_font_bold)
        y_offset = param_y_start + int(24 * s)
        for name, val in params.items():
            draw.text((int(25 * s), y_offset), f"• {name}: {val}", fill='darkblue', font=param_font)
            y_offset += int(16 * s)
        
        # Parameter evolution tracking (right side) — closer to molecule
        param_box_x = int(680 * s)
        draw.rectangle([param_box_x-5, int(80 * s), param_box_x+int(290 * s), int(450 * s)], outline='black', width=1)
        draw.text((param_box_x, int(85 * s)), "📊 Parameter Evolution History:", fill='black', font=small_font_bold)
        
        history_y = int(112 * s)
        hist_len = min(len(self.performance_evolution.get(model_type, [])), 4)
        start_idx = max(0, len(self.performance_evolution.get(model_type, [])) - hist_len)
        for idx in range(start_idx, start_idx + hist_len):
            draw.text((param_box_x, history_y), f"Iteration {idx}:", fill='black', font=param_font_bold)
            history_y += int(16 * s)
            params = self.get_parameters_for_iteration(model_type, idx)
            for pname, pval in params.items():
                draw.text((param_box_x + int(10 * s), history_y), f"• {pname}: {pval}", fill='darkgreen', font=param_font)
                history_y += int(14 * s)
            q_list = self.quality_evolution.get(model_type, [])
            q_val = q_list[idx] if idx < len(q_list) else 0.0
            draw.text((param_box_x + int(10 * s), history_y), f"→ Quality: {q_val:.3f}", fill='blue', font=param_font)
            history_y += int(22 * s)
        
        # Legend — closer and bold header
        legend_x = int(680 * s)
        legend_y = int(460 * s)
        draw.text((legend_x, legend_y), "🎨 Contribution Legend:", fill='black', font=small_font_bold)
        colors_legend = [
            ("High Positive", (0, 0, 200)),
            ("Med Positive", (100, 100, 255)),
            ("Neutral", (200, 200, 200)),
            ("Med Negative", (255, 100, 100)),
            ("High Negative", (200, 0, 0))
        ]
        leg_y = legend_y + int(18 * s)
        for label, color in colors_legend:
            draw.rectangle([legend_x, leg_y, legend_x+int(18 * s), leg_y+int(14 * s)], fill=color)
            draw.text((legend_x+int(24 * s), leg_y), label, fill='black', font=param_font)
            leg_y += int(18 * s)
        
        # Panels row (bottom) — dynamic spacing for consistent layout
        px, py = int(50 * s), int(600 * s)
        gap = int(30 * s)
        x_cursor = px
        if acc_img is not None:
            canvas.paste(acc_img, (x_cursor, py))
            x_cursor += acc_img.size[0] + gap
        if cm_img is not None:
            canvas.paste(cm_img, (x_cursor, py))
            x_cursor += cm_img.size[0] + gap
        if rocpr_img is not None:
            canvas.paste(rocpr_img, (x_cursor, py))
        
        return canvas

    # ------------------------ Circular FP movie ------------------------
    def get_parameters_for_iteration(self, model_name, iteration):
        params = {}
        space = self.parameter_spaces.get(model_name, {})
        for k, v in space.items():
            params[k] = v[iteration % len(v)]
        return params

    def _train_sklearn_fallback(self, X_train, y_train):
        """Train a simple, robust sklearn baseline when TPOT is unavailable or fails."""
        clf = LogisticRegression(max_iter=200, n_jobs=1)
        clf.fit(X_train, y_train)
        return clf

    def create_dynamic_parameter_movie(self, model_name):
        """Create movie showing parameter changes with real TPOT training (circular_fingerprint)
        Falls back to a sklearn LogisticRegression baseline if TPOT is not installed or fails.
        """
        if model_name != 'circular_fingerprint':
            raise ValueError("This method handles only 'circular_fingerprint'. Use the specific methods for others.")
        
        print(f"🎬 Creating dynamic parameter movie for {model_name}")
        frames = []
        script_dir = Path(__file__).resolve().parent
        frame_dir = script_dir / "frames"
        frame_dir.mkdir(exist_ok=True)
        
        # Load and sample data once
        df = self.load_and_sample_data()
        smiles = df['cleanedMol'].values
        labels = df['classLabel'].values
        
        # Prepare target molecule (first from test split in first iteration)
        target_mol = None
        
        # Iterations
        self.quality_evolution['circular_fingerprint'] = []
        self.performance_evolution['circular_fingerprint'] = []
        
        for iteration in range(self.max_iters):
            params = self.get_parameters_for_iteration(model_name, iteration)
            radius = int(params['radius'])
            nBits = int(params['nBits'])
            # Fixed settings per your request
            useFeatures = False
            useChirality = False
            
            print(f"\n🔧 Iteration {iteration} params: radius={radius}, nBits={nBits}")
            
            # Featurize
            X = self.featurize(smiles, radius, nBits, useFeatures, useChirality)
            y = labels
            X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
                X, y, smiles, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train TPOT or fallback
            pipeline = None
            used_fallback = False
            try:
                if self.use_tpot:
                    with suppress_stderr():
                        from tpot import TPOTClassifier
                    t0 = time.time()
                    tpot = TPOTClassifier(
                        generations=2,
                        population_size=8,
                        cv=3,
                        scoring='accuracy',
                        random_state=42,
                        verbosity=2,
                        n_jobs=1,
                        max_time_mins=1
                    )
                    with silence_fds(), suppress_stderr():
                        tpot.fit(X_train, y_train)
                    pipeline = tpot.fitted_pipeline_
                    train_time = time.time() - t0
                else:
                    raise ImportError("TPOT disabled via USE_TPOT=0")
            except ImportError:
                print("⚠️ TPOT not installed or disabled. Using sklearn fallback (LogisticRegression).")
                used_fallback = True
            except Exception as e:
                print(f"⚠️ TPOT training failed at iteration {iteration}: {e}\n   → Falling back to sklearn baseline (LogisticRegression).")
                used_fallback = True
            
            if used_fallback:
                t0 = time.time()
                pipeline = self._train_sklearn_fallback(X_train, y_train)
                train_time = time.time() - t0
            
            # Evaluate
            try:
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                try:
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                except Exception:
                    y_proba = None
                    auc = np.nan
                cm = confusion_matrix(y_test, y_pred)
                # Update evolutions BEFORE plotting panels so current point is included
                performance_score = float(acc)
                # Quality will be computed later, append after computing it
                self.performance_evolution[model_name].append(performance_score)
                # Now render panels
                with suppress_stderr():
                    rocpr_img, auc_val, ap_val = self._plot_roc_pr(y_test, y_proba)
                    cm_img = self._plot_confusion_matrix(cm)
                    acc_img = self._plot_accuracy_evolution(model_name)
                self.last_auc[model_name] = auc_val
                self.last_auprc[model_name] = ap_val
                print(f"   ✅ Acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}, time={train_time:.1f}s")
            except Exception:
                acc, auc = 0.0, np.nan
                y_proba = None
                performance_score = float(acc)
                self.performance_evolution[model_name].append(performance_score)
                cm_img = self._plot_confusion_matrix(np.array([[0,0],[0,0]]))
                rocpr_img, auc_val, ap_val = self._plot_roc_pr(np.array([0,1]), np.array([0.0, 1.0]))
                acc_img = self._plot_accuracy_evolution(model_name)
                self.last_auc[model_name] = auc_val
                self.last_auprc[model_name] = ap_val
            
            # Choose consistent target molecule
            if self.target_smiles is None:
                self.target_smiles = str(smiles_test[0])
                print(f"   🎯 Target molecule selected for visualization: {self.target_smiles}")
            mol = Chem.MolFromSmiles(self.target_smiles)
            
            # Compute signed feature importances via permutation importance
            signed_imps = self.compute_signed_feature_importance(pipeline, X_test, y_test)
            
            # Get feature weights active in target sample
            x_target = self.featurize([self.target_smiles], radius, nBits, useFeatures, useChirality)[0]
            active_indices = np.where(x_target > 0)[0]
            top_k = min(100, len(active_indices))
            # Select top features by absolute importance intersected with active bits
            active_imps = [(idx, signed_imps[idx]) for idx in active_indices]
            active_imps.sort(key=lambda t: abs(t[1]), reverse=True)
            feature_weights = {idx: float(w) for idx, w in active_imps[:top_k] if abs(w) > 0}
            
            # Map to atom weights
            atom_weights = {}
            if mol is not None and feature_weights:
                atom_weights = self.map_fragment_weights_to_atoms(
                    mol, feature_weights, radius=radius, nBits=nBits, useFeatures=useFeatures, useChirality=useChirality
                )
            
            # Compute quality score from atom weights coverage and magnitude
            if mol is not None and atom_weights:
                aw = np.array(list(atom_weights.values()), dtype=float)
                coverage = len(atom_weights) / max(1, mol.GetNumAtoms())
                magnitude = float(np.mean(np.abs(aw) / (np.max(np.abs(aw)) + 1e-9)))
                quality_score = float(0.5 * coverage + 0.5 * magnitude)
            else:
                quality_score = 0.0
            performance_score = float(acc)
            self.quality_evolution['circular_fingerprint'].append(quality_score)
            # performance_evolution already updated above
            
            # Draw frame
            title = f"Dynamic Parameter Optimization - Circular Fingerprint"
            canvas = self.draw_molecule_with_dynamic_parameters(
                atom_weights, title, iteration, quality_score, performance_score, model_name, params, mol,
                auc_val=self.last_auc[model_name], auprc_val=self.last_auprc[model_name],
                acc_img=acc_img, cm_img=cm_img, rocpr_img=rocpr_img
            )
            if canvas is not None:
                frame_path = frame_dir / f"{model_name}_dynamic_frame_{iteration:02d}.png"
                canvas.save(frame_path)
                frames.append(str(frame_path))
        
        if not frames:
            print(f"❌ No frames created for {model_name}")
            return None
        
        # Create animation (match canvas size: 1200*s x 800*s at 200 dpi => figsize=6*s x 4*s)
        fig, ax = plt.subplots(figsize=(6 * self.scale, 4 * self.scale), dpi=200)
        ax.axis('off')
        
        def animate(frame_idx):
            ax.clear()
            ax.axis('off')
            if frame_idx < len(frames):
                img = plt.imread(frames[frame_idx])
                ax.imshow(img)
                progress = (frame_idx + 1) / len(frames)
                ax.text(0.02, 0.02, f"Agentic Progress: {progress:.0%} ({frame_idx+1}/{len(frames)})", 
                        transform=ax.transAxes, fontsize=24, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="yellow", alpha=0.9))
            return [ax]
        
        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=4000, blit=False, repeat=True)
        movie_path = (Path(__file__).resolve().parent / f"{model_name}_dynamic_parameters.gif")
        print(f"💾 Saving {model_name} dynamic parameter movie to {movie_path}")
        ani.save(str(movie_path), writer='pillow', fps=1.0)
        plt.close()
        
        # Clean up frame files
        for frame_path in frames:
            try:
                Path(frame_path).unlink()
            except Exception:
                pass
        try:
            frame_dir.rmdir()
        except Exception:
            pass
        
        return movie_path

    # ------------------------ GraphConv movie ------------------------
    def create_graphconv_dynamic_parameter_movie(self):
        # Lazy import DeepChem only when needed
        global dc
        if dc is None:
            try:
                with silence_fds(), suppress_stderr():
                    import deepchem as _dc
                dc = _dc
            except Exception as e:
                print(f"❌ DeepChem not available: {e}")
                return None
        
        model_name = 'graphconv'
        print(f"🎬 Creating dynamic parameter movie for {model_name}")
        frames = []
        script_dir = Path(__file__).resolve().parent
        frame_dir = script_dir / "frames"
        frame_dir.mkdir(exist_ok=True)
        
        # Load and sample data
        df = self.load_and_sample_data()
        smiles = df['cleanedMol'].values.tolist()
        labels = df['classLabel'].values.astype(int)
        
        # Featurize once with DeepChem
        featurizer = dc.feat.ConvMolFeaturizer()
        X_all = featurizer.featurize(smiles)
        dataset = dc.data.NumpyDataset(X_all, labels, ids=np.array(smiles))
        splitter = dc.splits.RandomSplitter()
        train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.8, seed=42)
        
        # Choose consistent target molecule
        if self.target_smiles is None:
            self.target_smiles = str(test_dataset.ids[0])
            print(f"   🎯 Target molecule selected for visualization: {self.target_smiles}")
        mol = Chem.MolFromSmiles(self.target_smiles)
        
        # Reset evolutions
        self.quality_evolution['graphconv'] = []
        self.performance_evolution['graphconv'] = []
        
        # Iterate through parameter sets (use 4 iterations)
        for iteration in range(self.max_iters):
            params = self.get_parameters_for_iteration(model_name, iteration)
            hidden_dim = int(params['hidden_dim'])
            num_layers = int(params['num_layers'])
            dropout = float(params['dropout'])
            lr = float(params['learning_rate'])
            print(f"\n🔧 Iteration {iteration} params: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, lr={lr}")
            
            # Build and train model
            try:
                graph_layers = [hidden_dim] * num_layers
                model = dc.models.GraphConvModel(
                    n_tasks=1,
                    graph_conv_layers=graph_layers,
                    dense_layer_size=hidden_dim,
                    dropout=dropout,
                    learning_rate=lr,
                    batch_size=self.graphconv_batch,
                    mode='classification'
                )
                with silence_fds(), suppress_stderr():
                    model.fit(train_dataset, nb_epoch=self.graphconv_epochs)
            except Exception as e:
                print(f"❌ GraphConv training failed at iteration {iteration}: {e}")
                continue
            
            # Evaluate performance on test set
            try:
                with silence_fds(), suppress_stderr():
                    y_pred_raw = model.predict(test_dataset)
                # Attempt to extract probabilities for class 1
                y_proba = None
                arr = np.array(y_pred_raw)
                # Common shapes: (N, 1, 2) or (N, 2)
                if arr.ndim == 3 and arr.shape[-1] >= 2:
                    y_proba = arr[:, 0, 1]
                elif arr.ndim == 2 and arr.shape[-1] >= 2:
                    y_proba = arr[:, 1]
                else:
                    # Fallback to sigmoid of first logit or raw
                    y_proba = 1 / (1 + np.exp(-arr.squeeze()))
                y_pred = (y_proba > 0.5).astype(int)
                acc = accuracy_score(test_dataset.y.astype(int), y_pred)
                try:
                    auc = roc_auc_score(test_dataset.y.astype(int), y_proba)
                except Exception:
                    auc = np.nan
                cm = confusion_matrix(test_dataset.y.astype(int), y_pred)
                # Update evolutions BEFORE plotting
                performance_score = float(acc)
                self.performance_evolution['graphconv'].append(performance_score)
                # Panels
                with suppress_stderr():
                    rocpr_img, auc_val, ap_val = self._plot_roc_pr(test_dataset.y.astype(int), y_proba)
                    cm_img = self._plot_confusion_matrix(cm)
                    acc_img = self._plot_accuracy_evolution('graphconv')
                self.last_auc['graphconv'] = auc_val
                self.last_auprc['graphconv'] = ap_val
                print(f"   ✅ Acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}")
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                acc = 0.0
                y_proba = None
                performance_score = float(acc)
                self.performance_evolution['graphconv'].append(performance_score)
                cm_img = self._plot_confusion_matrix(np.array([[0,0],[0,0]]))
                rocpr_img, auc_val, ap_val = self._plot_roc_pr(np.array([0,1]), np.array([0.0, 1.0]))
                acc_img = self._plot_accuracy_evolution('graphconv')
                self.last_auc['graphconv'] = auc_val
                self.last_auprc['graphconv'] = ap_val
            
            # Compute fragment-based atom contributions for target molecule
            atom_weights = {}
            try:
                sdf_path = frame_dir / "tmp_target.sdf"
                df_tmp = pd.DataFrame({'cleanedMol': [self.target_smiles]})
                df_tmp['Molecule'] = [Chem.MolFromSmiles(self.target_smiles)]
                df_tmp['Name'] = ['Target']
                with silence_fds(), suppress_stderr():
                    from rdkit.Chem import PandasTools
                    PandasTools.WriteSDF(df_tmp[df_tmp['Molecule'].notna()], str(sdf_path), molColName='Molecule', idName='Name', properties=[])
                    loader_whole = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(), sanitize=True)
                    dataset_whole = loader_whole.create_dataset(str(sdf_path), shard_size=2000)
                    loader_frag = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True), sanitize=True)
                    dataset_frag = loader_frag.create_dataset(str(sdf_path), shard_size=2000)
                    tr = dc.trans.FlatteningTransformer(dataset_frag)
                    dataset_frag = tr.transform(dataset_frag)
                    pred_whole = model.predict(dataset_whole)
                    pred_frag = model.predict(dataset_frag)
            except Exception as e:
                print(f"   ⚠️ Contribution generation failed: {e}")
                atom_weights = {}
            finally:
                try:
                    if 'sdf_path' in locals() and Path(sdf_path).exists():
                        Path(sdf_path).unlink()
                except Exception:
                    pass
            
            # Quality score
            if mol is not None and atom_weights:
                aw = np.array(list(atom_weights.values()), dtype=float)
                coverage = len(atom_weights) / max(1, mol.GetNumAtoms())
                magnitude = float(np.mean(np.abs(aw) / (np.max(np.abs(aw)) + 1e-9)))
                quality_score = float(0.5 * coverage + 0.5 * magnitude)
            else:
                quality_score = 0.0
            performance_score = float(acc)
            self.quality_evolution['graphconv'].append(quality_score)
            # Removed duplicate performance_evolution append here
            
            # Draw frame
            title = "Dynamic Parameter Optimization - GraphConv"
            canvas = self.draw_molecule_with_dynamic_parameters(
                atom_weights, title, iteration, quality_score, performance_score, 'graphconv', params, mol,
                auc_val=self.last_auc['graphconv'], auprc_val=self.last_auprc['graphconv'],
                acc_img=acc_img, cm_img=cm_img, rocpr_img=rocpr_img
            )
            if canvas is not None:
                frame_path = frame_dir / f"{model_name}_dynamic_frame_{iteration:02d}.png"
                canvas.save(frame_path)
                frames.append(str(frame_path))
        
        if not frames:
            print(f"❌ No frames created for {model_name}")
            return None
        
        # Create animation (match canvas size and dpi)
        fig, ax = plt.subplots(figsize=(6 * self.scale, 4 * self.scale), dpi=200)
        ax.axis('off')
        def animate(frame_idx):
            ax.clear()
            ax.axis('off')
            if frame_idx < len(frames):
                img = plt.imread(frames[frame_idx])
                ax.imshow(img)
                progress = (frame_idx + 1) / len(frames)
                ax.text(0.02, 0.02, f"Agentic Progress: {progress:.0%} ({frame_idx+1}/{len(frames)})", 
                        transform=ax.transAxes, fontsize=24, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="yellow", alpha=0.9))
            return [ax]
        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=4000, blit=False, repeat=True)
        movie_path = (Path(__file__).resolve().parent / f"{model_name}_dynamic_parameters.gif")
        print(f"💾 Saving {model_name} dynamic parameter movie to {movie_path}")
        ani.save(str(movie_path), writer='pillow', fps=1.0)
        plt.close()
        
        # Cleanup
        for frame_path in frames:
            try:
                Path(frame_path).unlink()
            except Exception:
                pass
        try:
            frame_dir.rmdir()
        except Exception:
            pass
        return movie_path

    # --- New: ChemBERTa dynamic movie ---
    def create_chemberta_dynamic_parameter_movie(self):
        model_name = 'chemberta'
        print(f"🎬 Creating dynamic parameter movie for {model_name}")
        frames = []
        script_dir = Path(__file__).resolve().parent
        frame_dir = script_dir / "frames"
        frame_dir.mkdir(exist_ok=True)
        
        # Load and sample data
        df = self.load_and_sample_data()
        smiles = df['cleanedMol'].values
        labels = df['classLabel'].values.astype(int)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
            smiles, labels, smiles, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Choose target molecule
        if self.target_smiles is None:
            self.target_smiles = str(smiles_test[0])
            print(f"   🎯 Target molecule selected for visualization: {self.target_smiles}")
        mol = Chem.MolFromSmiles(self.target_smiles)
        
        # Reset evolutions
        self.quality_evolution['chemberta'] = []
        self.performance_evolution['chemberta'] = []
        
        for iteration in range(self.max_iters):
            params = self.get_parameters_for_iteration(model_name, iteration)
            lr = float(params['learning_rate'])
            epochs = int(params['num_train_epochs'])
            max_len = int(params['max_seq_length'])
            att_layer = int(params['attention_layer'])
            att_head = int(params['attention_head'])
            print(f"\n🔧 Iteration {iteration} params: lr={lr}, epochs={epochs}, max_len={max_len}, layer={att_layer}, head={att_head}")
            
            try:
                import torch
                from torch.utils.data import Dataset, DataLoader
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
                
                # Offline mode handling
                local_only = True if self.chemberta_offline else False
                
                class SmilesDataset(Dataset):
                    def __init__(self, smiles_list, labels_list, tokenizer, max_length):
                        self.smiles = list(smiles_list)
                        self.labels = list(labels_list)
                        self.tokenizer = tokenizer
                        self.max_length = max_length
                    def __len__(self):
                        return len(self.smiles)
                    def __getitem__(self, idx):
                        s = self.smiles[idx]
                        enc = self.tokenizer(
                            s,
                            truncation=True,
                            padding='max_length',
                            max_length=self.max_length,
                            return_tensors='pt'
                        )
                        item = {k: v.squeeze(0) for k, v in enc.items()}
                        item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
                        return item
                
                base_model = 'DeepChem/ChemBERTa-77M-MLM'
                with silence_fds(), suppress_stderr():
                    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=local_only)
                try:
                    with silence_fds(), suppress_stderr():
                        config = AutoConfig.from_pretrained(base_model, num_labels=2, local_files_only=local_only)
                        model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config, local_files_only=local_only)
                except Exception:
                    if self.chemberta_offline:
                        print("⚠️ ChemBERTa offline mode enabled and model not cached. Skipping iteration.")
                        continue
                    with silence_fds(), suppress_stderr():
                        config = AutoConfig.from_pretrained(base_model, num_labels=2)
                        model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)
                device = torch.device('cpu')
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                
                train_ds = SmilesDataset(X_train, y_train, tokenizer, max_len)
                train_loader = DataLoader(train_ds, batch_size=self.chemberta_batch, shuffle=True)
                
                model.train()
                with suppress_stderr():
                    for ep in range(max(1, epochs)):
                        for batch in train_loader:
                            batch = {k: v.to(device) for k, v in batch.items()}
                            out = model(**batch)
                            loss = out.loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                
                # Evaluate
                model.eval()
                test_ds = SmilesDataset(X_test, y_test, tokenizer, max_len)
                test_loader = DataLoader(test_ds, batch_size=max(2, self.chemberta_batch), shuffle=False)
                probs, preds, labels_list = [], [], []
                with torch.no_grad():
                    with silence_fds(), suppress_stderr():
                        for batch in test_loader:
                            labels_list.extend(batch['labels'].numpy().tolist())
                            batch = {k: v.to(device) for k, v in batch.items()}
                            out = model(**batch)
                            logits = out.logits
                            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                            probs.extend(p.tolist())
                            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                from sklearn.metrics import accuracy_score, roc_auc_score
                acc = accuracy_score(labels_list, preds)
                try:
                    auc = roc_auc_score(labels_list, probs)
                except Exception:
                    auc = np.nan
                cm = confusion_matrix(labels_list, preds)
                # Update perf before plotting
                performance_score = float(acc)
                self.performance_evolution[model_name].append(performance_score)
                with suppress_stderr():
                    rocpr_img, auc_val, ap_val = self._plot_roc_pr(np.array(labels_list), np.array(probs))
                    cm_img = self._plot_confusion_matrix(cm)
                    acc_img = self._plot_accuracy_evolution(model_name)
                self.last_auc[model_name] = auc_val
                self.last_auprc[model_name] = ap_val
                print(f"   ✅ Acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}")
            except Exception as e:
                print(f"❌ ChemBERTa training failed at iteration {iteration}: {e}")
                continue
            
            # Interpretation via [CLS]→tokens attention mapped to atoms
            atom_weights = self._chemberta_cls_attention_to_atom_weights(
                model, tokenizer, self.target_smiles, max_len, att_layer, att_head
            )
            
            # Quality score
            if mol is not None and atom_weights:
                aw = np.array(list(atom_weights.values()), dtype=float)
                coverage = len(atom_weights) / max(1, mol.GetNumAtoms())
                magnitude = float(np.mean(np.abs(aw) / (np.max(np.abs(aw)) + 1e-9)))
                quality_score = float(0.5 * coverage + 0.5 * magnitude)
            else:
                quality_score = 0.0
            
            self.quality_evolution['chemberta'].append(quality_score)
            
            # Draw frame
            title = 'Dynamic Parameter Optimization - ChemBERTa'
            canvas = self.draw_molecule_with_dynamic_parameters(
                atom_weights, title, iteration, quality_score, performance_score, 'chemberta', params, mol,
                auc_val=self.last_auc['chemberta'], auprc_val=self.last_auprc['chemberta'],
                acc_img=acc_img, cm_img=cm_img, rocpr_img=rocpr_img
            )
            if canvas is not None:
                frame_path = frame_dir / f"{model_name}_dynamic_frame_{iteration:02d}.png"
                canvas.save(frame_path)
                frames.append(str(frame_path))
        
        if not frames:
            print(f"❌ No frames created for {model_name}")
            return None
        
        # Create animation (match canvas size and dpi)
        fig, ax = plt.subplots(figsize=(6 * self.scale, 4 * self.scale), dpi=200)
        ax.axis('off')
        def animate(frame_idx):
            ax.clear()
            ax.axis('off')
            if frame_idx < len(frames):
                img = plt.imread(frames[frame_idx])
                ax.imshow(img)
                progress = (frame_idx + 1) / len(frames)
                ax.text(0.02, 0.02, f"Agentic Progress: {progress:.0%} ({frame_idx+1}/{len(frames)})", 
                        transform=ax.transAxes, fontsize=24, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="yellow", alpha=0.9))
            return [ax]
        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=4000, blit=False, repeat=True)
        movie_path = (Path(__file__).resolve().parent / f"{model_name}_dynamic_parameters.gif")
        print(f"💾 Saving {model_name} dynamic parameter movie to {movie_path}")
        ani.save(str(movie_path), writer='pillow', fps=1.0)
        plt.close()
        
        # Cleanup
        for frame_path in frames:
            try:
                Path(frame_path).unlink()
            except Exception:
                pass
        try:
            frame_dir.rmdir()
        except Exception:
            pass
        return movie_path

    def _smiles_atom_spans(self, smiles: str):
        """Return list of (start, end, atom_index) spans for atoms in a SMILES string.
        Handles bracket atoms [..] as one atom and organic subset atoms (B,C,N,O,P,S,F,I,b,c,n,o,p,s)
        plus two-letter halogens Cl/Br outside brackets. This is a heuristic but works well for most SMILES.
        """
        spans = []
        i = 0
        atom_idx = 0
        L = len(smiles)
        while i < L:
            ch = smiles[i]
            if ch == '[':
                j = i + 1
                while j < L and smiles[j] != ']':
                    j += 1
                j = min(j + 1, L)  # include closing ']'
                spans.append((i, j, atom_idx))
                atom_idx += 1
                i = j
                continue
            # two-letter halogens outside brackets
            if i + 1 < L and smiles[i:i+2] in ("Cl", "Br"):
                spans.append((i, i+2, atom_idx))
                atom_idx += 1
                i += 2
                continue
            # organic subset atoms
            if ch in set('BCNOPSFIbcnops'):
                spans.append((i, i+1, atom_idx))
                atom_idx += 1
                i += 1
                continue
            # not an atom (bonds, ring digits, branches, etc.)
            i += 1
        return spans

    def _chemberta_cls_attention_to_atom_weights(self, model, tokenizer, smiles: str, max_len: int, att_layer: int, att_head: int):
        """Map ChemBERTa [CLS]->token attention to RDKit atom weights using token offset mappings into SMILES.
        Returns dict: atom_index -> weight (normalized to [-1, 1] scale).
        """
        try:
            import torch
        except Exception:
            return {}
        # Tokenize with offsets (Fast tokenizer required for offsets)
        enc = tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        offsets = enc.get('offset_mapping', None)
        device = torch.device('cpu')
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = out.attentions  # tuple(layers) of (B, H, L, L)
        if not attentions:
            return {}
        num_layers = len(attentions)
        layer_idx = att_layer if att_layer >= 0 else num_layers + att_layer
        layer_idx = max(0, min(layer_idx, num_layers - 1))
        att = attentions[layer_idx][0]  # (H, L, L)
        num_heads = att.shape[0]
        head_idx = max(0, min(att_head, num_heads - 1))
        # Attention from CLS (position 0) to all tokens
        cls_to_tok = att[head_idx, 0, :]  # (L,)
        # Build SMILES atom spans
        atom_spans = self._smiles_atom_spans(smiles)
        if not atom_spans:
            return {}
        # Aggregate token attention to atoms via offset overlap
        atom_weights = {idx: 0.0 for _, _, idx in atom_spans}
        L = cls_to_tok.shape[0]
        # If no offsets (e.g., slow tokenizer), distribute over non-special tokens uniformly
        if offsets is None:
            # heuristic: map non-padding tokens (mask==1) excluding position 0 to atoms round-robin
            tok_indices = [i for i in range(1, int(attention_mask.sum().item()))]
            if not tok_indices:
                return {}
            per_tok = float(torch.sum(cls_to_tok[tok_indices]).item()) / max(1, len(atom_weights))
            for k in atom_weights.keys():
                atom_weights[k] = per_tok
        else:
            offs = offsets[0].tolist()  # list of (start, end)
            for ti in range(min(L, len(offs))):
                start, end = offs[ti]
                if (end - start) <= 0:
                    continue  # special or padding token
                w = float(cls_to_tok[ti].item())
                # assign to any atom span overlapping this token span
                for a_start, a_end, a_idx in atom_spans:
                    if not (end <= a_start or start >= a_end):  # overlap
                        atom_weights[a_idx] = atom_weights.get(a_idx, 0.0) + w
        # Normalize to [-1, 1]
        if atom_weights:
            vals = list(atom_weights.values())
            vmax = max(vals) if vals else 1.0
            if vmax == 0:
                vmax = 1.0
            for k in list(atom_weights.keys()):
                atom_weights[k] = float(atom_weights[k] / vmax)
        return atom_weights