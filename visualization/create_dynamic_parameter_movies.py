#!/usr/bin/env python3
"""
Dynamic Parameter Evolution Movie Creator
=========================================

Creates movies showing ACTUAL parameter changes across iterations,
now using real TPOT AutoML training on Morgan (circular) fingerprints
instead of simulated contributions. Keeps the original layout.
"""

import os
import time
import json
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
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
# --- Added for GraphConv support ---
import warnings
warnings.filterwarnings('ignore')
try:
    import deepchem as dc
    from rdkit.Chem import PandasTools
except Exception:
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
        data_file = self._data_path()
        print(f"📂 Loading data from: {data_file}")
        df = pd.read_excel(data_file)
        # Balanced sample
        class_0 = df[df['classLabel'] == 0].sample(n=n_per_class, random_state=random_state)
        class_1 = df[df['classLabel'] == 1].sample(n=n_per_class, random_state=random_state)
        self.sampled_df = pd.concat([class_0, class_1], ignore_index=True)
        print(f"✅ Loaded {len(self.sampled_df)} molecules (balanced sample)")
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
        return atom_weights

    def compute_signed_feature_importance(self, pipeline, X_test, y_test):
        # Permutation importance on the input to the pipeline
        try:
            pi = permutation_importance(pipeline, X_test, y_test, scoring='accuracy', n_repeats=5, random_state=42)
            importances = pi.importances_mean
        except Exception:
            # Fallback: zero importances
            importances = np.zeros(X_test.shape[1], dtype=float)
        # Sign via correlation between feature and predicted probabilities
        try:
            if hasattr(pipeline, 'predict_proba'):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
            else:
                # Use decision function or predictions as proxy
                y_proba = pipeline.predict(X_test)
                if y_proba.ndim > 1:
                    y_proba = y_proba[:, 0]
            # Compute correlation sign efficiently
            X_centered = X_test - X_test.mean(axis=0)
            y_centered = y_proba - y_proba.mean()
            denom = (np.sqrt((X_centered**2).sum(axis=0)) * np.sqrt((y_centered**2).sum())) + 1e-9
            corr = (X_centered.T @ y_centered) / denom
            signs = np.sign(corr)
        except Exception:
            signs = np.ones_like(importances)
        signed = importances * signs
        return signed

    # ------------------------ Small plot renderers ------------------------
    def _fig_to_image(self, fig, size=(280, 180)):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=160)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img.resize(size, Image.BICUBIC)

    def _plot_accuracy_evolution(self, model_type):
        vals = self.performance_evolution.get(model_type, [])
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(range(len(vals)), vals, marker='o', color='tab:blue')
        ax.set_title('Accuracy Evolution')
        ax.set_xlabel('Iter')
        ax.set_ylabel('Acc')
        ax.set_ylim(0, 1)
        ax.grid(True, ls='--', alpha=0.3)
        return self._fig_to_image(fig)

    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(3, 2))
        im = ax.imshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha='center', va='center', color='black', fontsize=9)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Pred')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return self._fig_to_image(fig)

    def _plot_roc_pr(self, y_true, y_scores):
        fig, axes = plt.subplots(1, 2, figsize=(6, 2))
        auc_val = None
        ap_val = None
        if y_scores is not None and np.ndim(y_scores) == 1 and len(y_scores) == len(y_true):
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            try:
                auc_val = roc_auc_score(y_true, y_scores)
            except Exception:
                auc_val = None
            axes[0].plot(fpr, tpr, color='tab:red')
            axes[0].plot([0, 1], [0, 1], 'k--', lw=0.8)
            axes[0].set_title(f'ROC (AUC={auc_val:.3f})' if auc_val is not None else 'ROC (AUC=NA)')
            axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
            axes[0].grid(True, ls='--', alpha=0.3)
            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            try:
                ap_val = average_precision_score(y_true, y_scores)
            except Exception:
                ap_val = None
            axes[1].plot(rec, prec, color='tab:green')
            axes[1].set_title(f'PR (AUPRC={ap_val:.3f})' if ap_val is not None else 'PR (AUPRC=NA)')
            axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
            axes[1].grid(True, ls='--', alpha=0.3)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, 'No probabilities', ha='center', va='center')
                ax.axis('off')
        plt.tight_layout()
        img = self._fig_to_image(fig, size=(560, 180))
        return img, auc_val, ap_val

    # ------------------------ ChemBERTa attention helper ------------------------
    def _chemberta_cls_attention_to_atom_weights(self, model, tokenizer, smiles, max_len, att_layer, att_head):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        try:
            # Match interpret_chemberta.py: ensure eager attention outputs
            try:
                model.config.attn_implementation = "eager"
            except Exception:
                pass
            model.config.output_attentions = True
            inputs = tokenizer(smiles, return_tensors='pt', truncation=True, max_length=max_len)
            import torch
            device = next(model.parameters()).device
            with torch.no_grad():
                outputs = model(**{k: v.to(device) for k, v in inputs.items()}, output_attentions=True)
            attentions = outputs.attentions
            layer_idx = att_layer if att_layer >= 0 else (len(attentions) + att_layer)
            layer_idx = max(0, min(layer_idx, len(attentions)-1))
            head_idx = max(0, min(att_head, attentions[layer_idx].shape[1]-1))
            att = attentions[layer_idx][0, head_idx].detach().cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            atom_count = mol.GetNumAtoms()
            token_atom_indices = list(range(1, min(len(tokens), 1 + atom_count)))
            atom_weights = {}
            if len(token_atom_indices) > 0:
                cls_attention = att[0, token_atom_indices]
                if cls_attention.sum() > 0:
                    cls_attention = cls_attention / cls_attention.sum()
                else:
                    cls_attention = np.ones_like(cls_attention) / len(cls_attention)
                for i in range(len(token_atom_indices)):
                    atom_weights[i] = float(cls_attention[i])
            return atom_weights
        except Exception:
            return {}

    # ------------------------ Drawing ------------------------
    def draw_molecule_with_dynamic_parameters(self, contributions, title, iteration, 
                                             quality_score, performance_score, model_type, current_params, mol,
                                             auc_val=None, auprc_val=None, acc_img=None, cm_img=None, rocpr_img=None):
        """Draw molecule with dynamically changing parameters (layout + metric panels)"""
        if mol is None:
            return None
        
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 400)
        
        # Normalize contributions for coloring
        num_atoms = mol.GetNumAtoms()
        contrib_array = np.zeros(num_atoms, dtype=float)
        for idx, w in contributions.items():
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
        
        drawer.SetFontSize(14)
        if atom_radii:
            drawer.DrawMolecule(mol, highlightAtoms=list(range(num_atoms)), highlightAtomColors=atom_colors, highlightAtomRadii=atom_radii)
        else:
            drawer.DrawMolecule(mol, highlightAtoms=list(range(num_atoms)), highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        canvas = Image.new('RGB', (1200, 800), 'white')
        canvas.paste(img, (50, 50))
        
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype("Arial.ttf", 18)
            metric_font = ImageFont.truetype("Arial.ttf", 14)
            small_font = ImageFont.truetype("Arial.ttf", 12)
            param_font = ImageFont.truetype("Arial.ttf", 10)
        except Exception:
            title_font = ImageFont.load_default()
            metric_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            param_font = ImageFont.load_default()
        
        # Title and model info
        draw.text((20, 15), title, fill='black', font=title_font)
        draw.text((20, 460), f"Model: {model_type.replace('_', ' ').title()}", 
                 fill=self.colors.get(model_type, 'black'), font=metric_font)
        
        # Metrics (include AUC/AUPRC)
        draw.text((350, 460), f"Iteration: {iteration}", fill='black', font=metric_font)
        draw.text((20, 480), f"Quality: {quality_score:.3f}", fill='blue', font=small_font)
        draw.text((150, 480), f"Accuracy: {performance_score:.3f}", fill='green', font=small_font)
        draw.text((280, 480), f"ROC AUC: {auc_val:.3f}" if auc_val is not None else "ROC AUC: NA", fill='darkred', font=small_font)
        draw.text((420, 480), f"AUPRC: {auprc_val:.3f}" if auprc_val is not None else "AUPRC: NA", fill='darkgreen', font=small_font)
        draw.text((300, 500), f"Combined: {0.6*performance_score + 0.4*quality_score:.3f}", 
                 fill='purple', font=small_font)
        
        # Current parameters box
        param_y_start = 500
        draw.text((20, param_y_start), "🔧 Current Parameters:", fill='black', font=small_font)
        y_offset = param_y_start + 20
        for name, val in current_params.items():
            draw.text((25, y_offset), f"• {name}: {val}", fill='darkblue', font=param_font)
            y_offset += 12
        
        # Parameter evolution tracking (right side)
        param_box_x = 900
        draw.rectangle([param_box_x-5, 80, param_box_x+290, 450], outline='black', width=1)
        draw.text((param_box_x, 85), "📊 Parameter Evolution History:", fill='black', font=small_font)
        
        history_y = 105
        hist_len = min(len(self.performance_evolution.get(model_type, [])), 4)
        start_idx = max(0, len(self.performance_evolution.get(model_type, [])) - hist_len)
        for idx in range(start_idx, start_idx + hist_len):
            draw.text((param_box_x, history_y), f"Iteration {idx}:", fill='black', font=param_font)
            history_y += 12
            params = self.get_parameters_for_iteration(model_type, idx)
            for pname, pval in params.items():
                draw.text((param_box_x + 10, history_y), f"• {pname}: {pval}", fill='darkgreen', font=param_font)
                history_y += 11
            q = self.quality_evolution.get(model_type, [0]* (idx+1))[idx]
            draw.text((param_box_x + 10, history_y), f"→ Quality: {q:.3f}", fill='blue', font=param_font)
            history_y += 20
        
        # Strategy (model-specific)
        strategy_y = 350
        draw.text((param_box_x, strategy_y), "🎯 Agentic Strategy:", fill='black', font=small_font)
        if model_type == 'circular_fingerprint':
            strategy_info = [
                "• Exploring radius 1-4 for coverage",
                "• Testing nBits 1024-4096 for detail",
                "• TPOT or sklearn baseline optimizes downstream model"
            ]
        elif model_type == 'graphconv':
            strategy_info = [
                "• Varying hidden_dim for capacity",
                "• Adjusting num_layers for depth",
                "• Tuning dropout & lr for stability"
            ]
        elif model_type == 'chemberta':
            strategy_info = [
                "• Tune lr/epochs/seq length",
                "• Vary attention layer/head for interpretability",
                "• Use [CLS] attention → atom weights"
            ]
        else:
            strategy_info = ["• Iterative parameter search", "• Balance perf & explainability"]
        strat_y = strategy_y + 15
        for info in strategy_info:
            draw.text((param_box_x, strat_y), info, fill='purple', font=param_font)
            strat_y += 12
        
        # Legend
        legend_x = 900
        legend_y = 460
        draw.text((legend_x, legend_y), "🎨 Contribution Legend:", fill='black', font=small_font)
        colors_legend = [
            ("High Positive", (0, 0, 200)),
            ("Med Positive", (100, 100, 255)),
            ("Neutral", (200, 200, 200)),
            ("Med Negative", (255, 100, 100)),
            ("High Negative", (200, 0, 0))
        ]
        leg_y = legend_y + 15
        for label, color in colors_legend:
            draw.rectangle([legend_x, leg_y, legend_x+15, leg_y+12], fill=color)
            draw.text((legend_x+20, leg_y), label, fill='black', font=param_font)
            leg_y += 15
        
        # Panels row (bottom)
        px, py = 50, 600
        if acc_img is not None:
            canvas.paste(acc_img, (px, py))
        if cm_img is not None:
            canvas.paste(cm_img, (px + 300, py))
        if rocpr_img is not None:
            canvas.paste(rocpr_img, (px + 600, py))
        
        return canvas

    # ------------------------ Parameter helpers ------------------------
    def get_parameters_for_iteration(self, model_type, iteration):
        params = {}
        for param_name, values in self.parameter_spaces[model_type].items():
            params[param_name] = values[iteration % len(values)]
        return params

    # --- New: sklearn fallback trainer ---
    def _train_sklearn_fallback(self, X_train, y_train):
        """Train a simple, robust sklearn baseline when TPOT is unavailable or fails."""
        clf = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
        clf.fit(X_train, y_train)
        return clf

    # ------------------------ Training & Movie Creation ------------------------
    def create_dynamic_parameter_movie(self, model_name):
        """Create movie showing parameter changes with real TPOT training (circular_fingerprint)
        Falls back to a sklearn LogisticRegression baseline if TPOT is not installed or fails.
        """
        if model_name != 'circular_fingerprint':
            print(f"⏭️ Skipping unsupported model: {model_name}")
            return None
        
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
        
        for iteration in range(4):
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
                    max_time_mins=1  # ~1 minute per iteration
                )
                tpot.fit(X_train, y_train)
                pipeline = tpot.fitted_pipeline_
                train_time = time.time() - t0
            except ImportError:
                print("⚠️ TPOT not installed. Using sklearn fallback (LogisticRegression).")
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
                rocpr_img, auc_val, ap_val = self._plot_roc_pr(y_test, y_proba)
                cm_img = self._plot_confusion_matrix(cm)
                acc_img = self._plot_accuracy_evolution(model_name)
                self.last_auc[model_name] = auc_val
                self.last_auprc[model_name] = ap_val
                print(f"   ✅ Acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}, time={train_time:.1f}s")
            except Exception:
                acc, auc = 0.0, np.nan
                y_proba = None
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
            self.performance_evolution['circular_fingerprint'].append(performance_score)
            
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
        
        # Create animation
        fig, ax = plt.subplots(figsize=(15, 11))
        ax.axis('off')
        
        def animate(frame_idx):
            ax.clear()
            ax.axis('off')
            if frame_idx < len(frames):
                img = plt.imread(frames[frame_idx])
                ax.imshow(img)
                progress = (frame_idx + 1) / len(frames)
                ax.text(0.02, 0.02, f"Agentic Progress: {progress:.0%} ({frame_idx+1}/{len(frames)})", 
                        transform=ax.transAxes, fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
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

    # --- New: GraphConv dynamic movie ---
    def create_graphconv_dynamic_parameter_movie(self):
        model_name = 'graphconv'
        if dc is None:
            print("❌ DeepChem is not installed. Please install with: pip install deepchem")
            return None
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
        for iteration in range(4):
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
                    batch_size=32,
                    mode='classification'
                )
                model.fit(train_dataset, nb_epoch=10)
            except Exception as e:
                print(f"❌ GraphConv training failed at iteration {iteration}: {e}")
                continue
            
            # Evaluate performance on test set
            try:
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
                PandasTools.WriteSDF(df_tmp[df_tmp['Molecule'].notna()], str(sdf_path), molColName='Molecule', idName='Name', properties=[])
                loader_whole = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(), sanitize=True)
                dataset_whole = loader_whole.create_dataset(str(sdf_path), shard_size=2000)
                loader_frag = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True), sanitize=True)
                dataset_frag = loader_frag.create_dataset(str(sdf_path), shard_size=2000)
                tr = dc.trans.FlatteningTransformer(dataset_frag)
                dataset_frag = tr.transform(dataset_frag)
                pred_whole = model.predict(dataset_whole)
                pred_frag = model.predict(dataset_frag)
                def to_prob(arr):
                    arr = np.array(arr)
                    if arr.ndim == 3 and arr.shape[-1] >= 2:
                        return arr[:, 0, 1]
                    if arr.ndim == 2 and arr.shape[-1] >= 2:
                        return arr[:, 1]
                    return 1 / (1 + np.exp(-arr.squeeze()))
                p_whole = to_prob(pred_whole)
                p_frag = to_prob(pred_frag)
                n = min(len(p_whole), len(dataset_whole.ids))
                p_whole = p_whole[:n]
                ids_whole = dataset_whole.ids[:n]
                m = min(len(p_frag), len(dataset_frag.ids))
                p_frag = p_frag[:m]
                ids_frag = dataset_frag.ids[:m]
                pred_whole_df = pd.DataFrame(p_whole, index=ids_whole, columns=["Molecule"]) 
                pred_frag_df = pd.DataFrame(p_frag, index=ids_frag, columns=["Fragment"]) 
                df_merged = pd.merge(pred_frag_df, pred_whole_df, right_index=True, left_index=True, how='left')
                df_merged['Contrib'] = df_merged['Molecule'] - df_merged['Fragment']
                contrib_series = None
                if self.target_smiles in df_merged.index:
                    contrib_series = df_merged.loc[self.target_smiles, 'Contrib']
                elif 'Target' in df_merged.index:
                    contrib_series = df_merged.loc['Target', 'Contrib']
                else:
                    contrib_series = df_merged['Contrib']
                if hasattr(contrib_series, 'values') and getattr(contrib_series, 'shape', ()): 
                    contrib_values = np.array(contrib_series).flatten()
                else:
                    contrib_values = np.array([float(contrib_series)])
                num_heavy = mol.GetNumHeavyAtoms() if mol is not None else 0
                if num_heavy > 0:
                    vals = np.zeros(num_heavy)
                    k = min(num_heavy, len(contrib_values))
                    vals[:k] = contrib_values[:k]
                    atom_weights = {i: float(vals[i]) for i in range(num_heavy)}
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
            self.performance_evolution['graphconv'].append(performance_score)
            
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
        
        # Create animation
        fig, ax = plt.subplots(figsize=(15, 11))
        ax.axis('off')
        def animate(frame_idx):
            ax.clear()
            ax.axis('off')
            if frame_idx < len(frames):
                img = plt.imread(frames[frame_idx])
                ax.imshow(img)
                progress = (frame_idx + 1) / len(frames)
                ax.text(0.02, 0.02, f"Agentic Progress: {progress:.0%} ({frame_idx+1}/{len(frames)})", 
                        transform=ax.transAxes, fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
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
        
        # Iterate parameter sets
        for iteration in range(4):
            params = self.get_parameters_for_iteration(model_name, iteration)
            lr = float(params['learning_rate'])
            epochs = int(params['num_train_epochs'])
            max_len = int(params['max_seq_length'])
            att_layer = int(params['attention_layer'])
            att_head = int(params['attention_head'])
            print(f"\n🔧 Iteration {iteration} params: lr={lr}, epochs={epochs}, max_len={max_len}, layer={att_layer}, head={att_head}")
            
            # Train ChemBERTa using Transformers directly
            try:
                import torch
                from torch.utils.data import Dataset, DataLoader
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
                from sklearn.metrics import accuracy_score, roc_auc_score
                
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
                tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
                config = AutoConfig.from_pretrained(base_model, num_labels=2)
                model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)
                device = torch.device('cpu')
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                
                train_ds = SmilesDataset(X_train, y_train, tokenizer, max_len)
                train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
                
                model.train()
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
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
                probs, preds, labels_list = [], [], []
                with torch.no_grad():
                    for batch in test_loader:
                        labels_list.extend(batch['labels'].numpy().tolist())
                        batch = {k: v.to(device) for k, v in batch.items()}
                        out = model(**batch)
                        logits = out.logits
                        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                        probs.extend(p.tolist())
                        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                acc = accuracy_score(labels_list, preds)
                try:
                    auc = roc_auc_score(labels_list, probs)
                except Exception:
                    auc = np.nan
                cm = confusion_matrix(labels_list, preds)
                rocpr_img, auc_val, ap_val = self._plot_roc_pr(np.array(labels_list), np.array(probs))
                cm_img = self._plot_confusion_matrix(cm)
                acc_img = self._plot_accuracy_evolution(model_name)
                self.last_auc[model_name] = auc_val
                self.last_auprc[model_name] = ap_val
                print(f"   ✅ Acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}")
            except Exception as e:
                print(f"❌ ChemBERTa training failed at iteration {iteration}: {e}")
                continue
            
            # Attention-based interpretation for target_smiles from the fine-tuned model
            atom_weights = {}
            try:
                # Enable attentions for this forward pass
                model.config.output_attentions = True
                inputs = tokenizer(self.target_smiles, return_tensors='pt', truncation=True, max_length=max_len)
                with torch.no_grad():
                    outputs = model(**{k: v.to(device) for k, v in inputs.items()}, output_attentions=True)
                attentions = outputs.attentions
                # Select layer/head
                layer_idx = att_layer if att_layer >= 0 else (len(attentions) + att_layer)
                layer_idx = max(0, min(layer_idx, len(attentions)-1))
                head_idx = max(0, min(att_head, attentions[layer_idx].shape[1]-1))
                att = attentions[layer_idx][0, head_idx].detach().cpu().numpy()
                # Map [CLS]→tokens to atoms heuristically
                atom_count = mol.GetNumAtoms() if mol is not None else 0
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                seq_len = len(tokens)
                token_atom_indices = list(range(1, min(seq_len, 1 + atom_count)))
                if len(token_atom_indices) > 0:
                    cls_attention = att[0, token_atom_indices]
                    if cls_attention.sum() > 0:
                        cls_attention = cls_attention / cls_attention.sum()
                    else:
                        cls_attention = np.ones_like(cls_attention) / len(cls_attention)
                    for i in range(len(token_atom_indices)):
                        atom_weights[i] = float(cls_attention[i])
            except Exception as e:
                print(f"   ⚠️ Attention extraction failed: {e}")
                atom_weights = {}
            
            # Quality score
            if mol is not None and atom_weights:
                aw = np.array(list(atom_weights.values()), dtype=float)
                coverage = len(atom_weights) / max(1, mol.GetNumAtoms())
                magnitude = float(np.mean(np.abs(aw) / (np.max(np.abs(aw)) + 1e-9)))
                quality_score = float(0.5 * coverage + 0.5 * magnitude)
            else:
                quality_score = 0.0
            performance_score = float(acc)
            self.quality_evolution['chemberta'].append(quality_score)
            self.performance_evolution['chemberta'].append(performance_score)
            
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
        
        # Create animation
        fig, ax = plt.subplots(figsize=(15, 11))
        ax.axis('off')
        def animate(frame_idx):
            ax.clear()
            ax.axis('off')
            if frame_idx < len(frames):
                img = plt.imread(frames[frame_idx])
                ax.imshow(img)
                progress = (frame_idx + 1) / len(frames)
                ax.text(0.02, 0.02, f"Agentic Progress: {progress:.0%} ({frame_idx+1}/{len(frames)})", 
                        transform=ax.transAxes, fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
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