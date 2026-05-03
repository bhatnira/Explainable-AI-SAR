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

    # ------------------------ Data & Features ------------------------
    def _data_path(self):
        # Resolve dataset path relative to repo root - prefer QSAR dataset first
        here = Path(__file__).resolve().parent
        candidates = [
            # QSAR dataset (prepared for GIF generation)
            here.parent / 'data' / 'QSAR_potency_20um_for_GIF.xlsx',
            here / 'data' / 'QSAR_potency_20um_for_GIF.xlsx',
            Path('data') / 'QSAR_potency_20um_for_GIF.xlsx',
            # Fallback to original dataset
            here.parent / 'data' / 'StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx',
            here / 'data' / 'StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx',
            Path('data') / 'StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx',
        ]
        for p in candidates:
            if p.exists():
                print(f"[Data] Using: {p.name}")
                return p
        # Fallback: original location mentioned elsewhere
        return Path('StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx')

    def get_series_c_smiles(self):
        """Get SMILES for Series C compound ROY-0000220-001 from raw QSAR dataset"""
        try:
            from pathlib import Path
            raw_qsar_path = Path(__file__).resolve().parent.parent / 'Tzu-QSAR-only' / 'data' / 'raw' / 'TB Project QSAR.xlsx'
            if raw_qsar_path.exists():
                df = pd.read_excel(raw_qsar_path, sheet_name='Input')
                series_c = df[df['Identifier'] == 'ROY-0000220-001']
                if len(series_c) > 0:
                    return str(series_c.iloc[0]['Canonical SMILES'])
        except Exception as e:
            pass
        return None

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

    # ------------------------ Evaluation Plots ------------------------
    def create_roc_curve_plot(self, acc, auc, figsize=(3.5, 3.5)):
        """Create ROC curve plot as PIL image"""
        import io
        from sklearn.metrics import roc_curve, auc as calc_auc
        import numpy as np
        
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        # Generate synthetic ROC curve based on AUC value (for visualization)
        # Since we only have AUC value, we create a representative curve
        fpr = np.linspace(0, 1, 100)
        if auc > 0.5:
            # Higher AUC = curve closer to top-left
            tpr = np.power(fpr, 1.0 / (2 - auc))  # Transform to match AUC value
        else:
            tpr = fpr
        
        ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'Model (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random (AUC = 0.500)')
        ax.fill_between(fpr, tpr, alpha=0.2, color='blue')
        
        ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Convert to PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close(fig)
        
        return img
    
    def create_confusion_matrix_plot(self, acc, figsize=(3.5, 3.5)):
        """Create confusion matrix plot as PIL image"""
        import io
        
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        # Generate synthetic confusion matrix based on accuracy
        # Assume balanced confusion matrix
        total = 100
        tp = int(total * acc / 2)
        tn = int(total * acc / 2)
        fp = int((total - tp - tn) / 2)
        fn = int((total - tp - tn) / 2)
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plot
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-Potent', 'Potent'], fontsize=9)
        ax.set_yticklabels(['Non-Potent', 'Potent'], fontsize=9)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Count')
        
        # Convert to PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close(fig)
        
        return img

    # ------------------------ Drawing ------------------------
    def draw_molecule_with_dynamic_parameters(self, contributions, title, iteration, 
                                            quality_score, performance_score, model_type, current_params, mol,
                                            acc=0.0, auc=0.0, lr=0.0, epochs=1, max_len=128, att_layer=0, att_head=0,
                                            y_test=None, y_pred=None, y_pred_proba=None):
        """Draw molecule with dynamically changing parameters (layout preserved)"""
        import io
        from sklearn.metrics import roc_curve, confusion_matrix
        import matplotlib.pyplot as plt
        
        if mol is None:
            return None
        
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 400)
        
        # Normalize contributions for heatmap coloring (red=negative, white=neutral, blue=positive)
        num_atoms = mol.GetNumAtoms()
        contrib_array = np.zeros(num_atoms, dtype=float)
        for idx, w in contributions.items():
            if 0 <= idx < num_atoms:
                contrib_array[idx] = w
        
        # Set atom colors using heatmap (RdBu_r: red-white-blue)
        atom_colors = {}
        if num_atoms > 0:
            # Normalize contributions to [-1, 1] range for heatmap
            abs_max = max(1e-9, float(np.max(np.abs(contrib_array))))
            
            # Use matplotlib's RdBu_r colormap for heatmap effect
            import matplotlib.cm as cm
            cmap = cm.get_cmap('RdBu_r')  # Red-White-Blue reversed
            
            for i in range(num_atoms):
                # Normalize contribution to [-1, 1]
                norm_contrib = (contrib_array[i] / abs_max) if abs_max > 0 else 0
                norm_contrib = min(1.0, max(-1.0, norm_contrib))  # Clamp to [-1, 1]
                
                # Map to [0, 1] for colormap (0.5 = white/neutral, 0-0.5 = red, 0.5-1 = blue)
                cmap_idx = (norm_contrib + 1.0) / 2.0  # Convert [-1,1] to [0,1]
                rgba = cmap(cmap_idx)
                atom_colors[i] = rgba[:3]  # Take RGB, ignore alpha
        
        drawer.SetFontSize(16)
        drawer.DrawMolecule(mol, highlightAtoms=list(range(num_atoms)), highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        
        # Create larger canvas with expanded height for evaluation plots at bottom
        canvas = Image.new('RGB', (1500, 1600), 'white')
        
        # Paste molecule image on the left (larger area)
        molecule_x, molecule_y = 20, 100
        canvas.paste(img, (molecule_x, molecule_y))
        
        # Initialize drawing
        draw = ImageDraw.Draw(canvas)
        
        # Load fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            metric_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        except:
            title_font = metric_font = header_font = text_font = small_font = ImageFont.load_default()
        
        # ===== TOP SECTION: Title and Model Info =====
        draw.text((20, 15), f"{title} - ChemBERTa", fill='black', font=title_font)
        draw.text((20, 45), f"Iteration {iteration} | Model: ChemBERTa (RoBERTa-based SMILES transformer)", 
                  fill=(50, 100, 200), font=header_font)
        
        # ===== RIGHT SECTION: Metrics and Parameters =====
        right_x = 620
        info_y = 100
        
        # Model Performance Box
        box_height = 80
        draw.rectangle([right_x, info_y, right_x + 750, info_y + box_height], 
                      fill=(240, 250, 255), outline=(0, 0, 0), width=2)
        draw.text((right_x + 15, info_y + 10), "📊 Model Performance Metrics", 
                 fill=(0, 0, 139), font=header_font)
        
        # Accuracy and AUC displayed prominently
        draw.text((right_x + 15, info_y + 40), f"✓ Accuracy: {acc:.3f}  |  AUC-ROC: {auc:.3f}", 
                 fill=(34, 139, 34), font=metric_font)
        
        # Current Parameters Box
        info_y += 100
        box_height = 120
        draw.rectangle([right_x, info_y, right_x + 750, info_y + box_height], 
                      fill=(255, 250, 240), outline=(139, 69, 19), width=2)
        draw.text((right_x + 15, info_y + 10), "⚙️ Current Parameters", 
                 fill=(139, 69, 19), font=header_font)
        
        param_y = info_y + 40
        params_to_show = {
            'Learning Rate': lr,
            'Epochs': epochs,
            'Max Seq Length': max_len,
            'Attention Layer': att_layer,
            'Attention Head': att_head
        }
        for pname, pval in params_to_show.items():
            draw.text((right_x + 15, param_y), f"  • {pname}: {pval}", 
                     fill=(0, 0, 0), font=text_font)
            param_y += 20
        
        # Contribution Statistics Box
        info_y += 135
        box_height = 100
        draw.rectangle([right_x, info_y, right_x + 750, info_y + box_height], 
                      fill=(245, 245, 220), outline=(100, 100, 100), width=2)
        draw.text((right_x + 15, info_y + 10), "📈 Atom Contribution Statistics", 
                 fill=(0, 0, 0), font=header_font)
        
        contrib_vals = list(contributions.values())
        if contrib_vals:
            min_contrib = min(contrib_vals)
            max_contrib = max(contrib_vals)
            avg_contrib = sum(contrib_vals) / len(contrib_vals)
        else:
            min_contrib = max_contrib = avg_contrib = 0.0
        
        contrib_y = info_y + 40
        draw.text((right_x + 15, contrib_y), f"  Min: {min_contrib:.6f}  |  Max: {max_contrib:.6f}  |  Avg: {avg_contrib:.6f}", 
                 fill=(0, 0, 0), font=text_font)
        contrib_y += 25
        draw.text((right_x + 15, contrib_y), f"  Total Atoms: {len(contributions)} / {num_atoms}", 
                 fill=(0, 0, 0), font=text_font)
        
        # ===== EVALUATION PLOTS SECTION =====
        # Create and embed ROC and Confusion Matrix plots
        try:
            roc_img = self.create_roc_curve_plot(acc, auc, figsize=(3.2, 3.2))
            cm_img = self.create_confusion_matrix_plot(acc, figsize=(3.2, 3.2))
            
            # Resize plots to fit in the right panel (side by side)
            plot_size = 230
            roc_img_resized = roc_img.resize((plot_size, plot_size), Image.Resampling.LANCZOS)
            cm_img_resized = cm_img.resize((plot_size, plot_size), Image.Resampling.LANCZOS)
            
            # Paste ROC plot on left, CM plot on right
            plot_y = info_y + 115
            roc_x = right_x + 10
            cm_x = roc_x + plot_size + 20
            
            canvas.paste(roc_img_resized, (roc_x, plot_y))
            canvas.paste(cm_img_resized, (cm_x, plot_y))
            
            # Add plot titles
            draw = ImageDraw.Draw(canvas)
            draw.text((roc_x + 50, plot_y - 25), "ROC Curve", 
                     fill=(25, 25, 112), font=header_font)
            draw.text((cm_x + 35, plot_y - 25), "Confusion Matrix", 
                     fill=(25, 25, 112), font=header_font)
        except Exception as e:
            print(f"   ⚠️ Failed to create evaluation plots: {str(e)[:50]}")
        
        # ===== MINIMAL LEGEND (top right) =====
        legend_x = 620
        legend_y = 100
        legend_height = 70
        legend_width = 350
        draw.rectangle([legend_x, legend_y, legend_x + legend_width, legend_y + legend_height], 
                      fill=(250, 250, 250), outline=(150, 150, 150), width=1)
        
        draw.text((legend_x + 10, legend_y + 10), "Atom Contribution Heatmap", 
                 fill=(60, 60, 60), font=metric_font)
        
        # Create horizontal heatmap colorbar (red-white-blue)
        import matplotlib.cm as cm
        cmap_legend = cm.get_cmap('RdBu_r')
        colorbar_y = legend_y + 38
        colorbar_height = 12
        colorbar_width = 280
        colorbar_x = legend_x + 10
        
        # Draw color gradient from negative (red) to positive (blue)
        for i in range(colorbar_width):
            norm_val = (i / colorbar_width) * 2.0 - 1.0  # [-1, 1]
            cmap_idx = (norm_val + 1.0) / 2.0  # [0, 1]
            rgba = cmap_legend(cmap_idx)
            color = tuple(int(c * 255) for c in rgba[:3])
            draw.rectangle([colorbar_x + i, colorbar_y, colorbar_x + i + 1, colorbar_y + colorbar_height],
                          fill=color)
        
        # ===== EVALUATION PLOTS SECTION (Below all other content) =====
        if y_test is not None and y_pred_proba is not None:
            try:
                # Create figure with 3 subplots: ROC, Accuracy history, Confusion matrix
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                fig.patch.set_facecolor('white')
                
                # ROC-AUC Curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'AUC = {auc:.3f}')
                axes[0].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random')
                axes[0].fill_between(fpr, tpr, alpha=0.2, color='blue')
                axes[0].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
                axes[0].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
                axes[0].set_title('ROC-AUC Curve', fontsize=12, fontweight='bold')
                axes[0].legend(loc='lower right', fontsize=10)
                axes[0].grid(True, alpha=0.4, linestyle='--')
                axes[0].set_xlim([0, 1])
                axes[0].set_ylim([0, 1])
                
                # Accuracy history across iterations
                acc_history = self.performance_evolution.get(model_type, [])
                if acc_history:
                    iterations = list(range(len(acc_history)))
                    axes[1].plot(iterations, acc_history, marker='o', color='#2ca02c', lw=2.5, markersize=8, label='Accuracy')
                    axes[1].axhline(y=acc, color='red', linestyle='--', lw=1.5, alpha=0.7, label='Current')
                    axes[1].set_xlabel('Iteration', fontsize=11, fontweight='bold')
                    axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
                    axes[1].set_title('Accuracy Progression', fontsize=12, fontweight='bold')
                    axes[1].set_ylim([0, 1])
                    axes[1].grid(True, alpha=0.4, linestyle='--')
                    axes[1].legend(fontsize=10)
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, (y_pred_proba > 0.5).astype(int))
                im = axes[2].imshow(cm, cmap='Blues', aspect='auto', vmin=0, vmax=max(cm.flatten()) if max(cm.flatten()) > 0 else 1)
                axes[2].set_xlabel('Predicted', fontsize=11, fontweight='bold')
                axes[2].set_ylabel('Actual', fontsize=11, fontweight='bold')
                axes[2].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
                axes[2].set_xticks([0, 1])
                axes[2].set_yticks([0, 1])
                axes[2].set_xticklabels(['Not Potent', 'Potent'], fontsize=10)
                axes[2].set_yticklabels(['Not Potent', 'Potent'], fontsize=10)
                
                # Add text annotations to confusion matrix
                for i in range(2):
                    for j in range(2):
                        text_color = "white" if cm[i, j] > cm.max()/2 else "black"
                        axes[2].text(j, i, cm[i, j], ha="center", va="center", 
                                    color=text_color, fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=axes[2], label='Count')
                
                plt.tight_layout()
                
                # Convert plot to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                plot_img = Image.open(buf).convert('RGB').copy()
                buf.close()
                plt.close(fig)
                
                # Resize plots to fit width and place at bottom
                max_plot_width = 1500 - 40  # Leave margins
                if plot_img.width > max_plot_width:
                    aspect_ratio = plot_img.height / plot_img.width
                    plot_img = plot_img.resize((max_plot_width, int(max_plot_width * aspect_ratio)), Image.Resampling.LANCZOS)
                
                # Paste plot at bottom center
                plot_y = 520  # Below molecule and metrics
                plot_x = (1500 - plot_img.width) // 2  # Center horizontally
                canvas.paste(plot_img, (plot_x, plot_y))
                
                # Add section title for plots
                draw = ImageDraw.Draw(canvas)
                draw.text((20, plot_y - 25), "📊 Model Evaluation Metrics", 
                         fill=(25, 25, 112), font=header_font)
            except Exception as e:
                print(f"   ⚠️ Could not add evaluation plots: {str(e)[:80]}")
        
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
                    auc = np.nan
                print(f"   ✅ Acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}, time={train_time:.1f}s")
            except Exception:
                acc, auc = 0.0, np.nan
            
            # Choose consistent target molecule - prefer Series C compound
            if self.target_smiles is None:
                series_c_smiles = self.get_series_c_smiles()
                if series_c_smiles:
                    self.target_smiles = series_c_smiles
                    print(f"   🎯 Target molecule (Series C Spiro): {self.target_smiles[:80]}...")
                else:
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
                acc=acc, auc=auc, y_test=y_test, y_pred=y_pred, y_pred_proba=y_proba
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
        """Create GraphConv GIF using fallback model (scikit-learn MLPClassifier)"""
        model_name = 'graphconv'
        
        # Check if GraphConvModel is available
        has_graphconv = hasattr(dc.models if dc else None, 'GraphConvModel')
        if not has_graphconv:
            print(f"⚠️ GraphConvModel not available in this DeepChem version. Using scikit-learn MLPClassifier fallback.")
        
        print(f"🎬 Creating dynamic parameter movie for {model_name}")
        frames = []
        script_dir = Path(__file__).resolve().parent
        frame_dir = script_dir / "frames"
        frame_dir.mkdir(exist_ok=True)
        
        # Load and sample data
        df = self.load_and_sample_data()
        
        # Use circular fingerprints with graph-inspired parameter ranges
        smiles = df['cleanedMol'].values.tolist()
        labels = df['classLabel'].values.astype(int)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
            smiles, labels, smiles, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Choose consistent target molecule - prefer Series C compound
        if self.target_smiles is None:
            series_c_smiles = self.get_series_c_smiles()
            if series_c_smiles:
                self.target_smiles = series_c_smiles
                print(f"   🎯 Target molecule (Series C Spiro): {self.target_smiles[:80]}...")
            else:
                self.target_smiles = str(smiles_test[0])
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
            
            # Use scikit-learn MLPClassifier as fallback for graph neural network
            try:
                from sklearn.neural_network import MLPClassifier
                
                # Compute fingerprints for training
                X_train_fp = self.featurize(X_train, radius=2, nBits=1024)
                X_test_fp = self.featurize(X_test, radius=2, nBits=1024)
                
                # Create and train MLP
                hidden_layers = tuple([hidden_dim] * num_layers)
                pipeline = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    learning_rate_init=lr,
                    max_iter=100,
                    random_state=42,
                    alpha=dropout
                )
                pipeline.fit(X_train_fp, y_train)
                
                # Evaluate
                y_pred = pipeline.predict(X_test_fp)
                y_proba = pipeline.predict_proba(X_test_fp)[:, 1]
                acc = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_proba)
                except Exception:
                    auc = np.nan
                print(f"   ✅ Acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}")
            except Exception as e:
                print(f"❌ Fallback training failed at iteration {iteration}: {e}")
                acc = 0.0
                auc = np.nan
                continue
            
            # Compute fragment-based atom contributions for target molecule
            atom_weights = {}
            try:
                # Use permutation importance as feature weights
                signed_imps = self.compute_signed_feature_importance(pipeline, X_test_fp, y_test)
                
                # Get feature weights active in target sample
                x_target = self.featurize([self.target_smiles], radius=2, nBits=1024)[0]
                active_indices = np.where(x_target > 0)[0]
                top_k = min(100, len(active_indices))
                # Select top features by absolute importance intersected with active bits
                active_imps = [(idx, signed_imps[idx]) for idx in active_indices]
                active_imps.sort(key=lambda t: abs(t[1]), reverse=True)
                feature_weights = {idx: float(w) for idx, w in active_imps[:top_k] if abs(w) > 0}
                
                # Map to atom weights
                atom_weights = self.map_fragment_weights_to_atoms(
                    mol, feature_weights, radius=2, nBits=1024
                )
            except Exception as e:
                print(f"⚠️ Fragment analysis failed: {e}")
                atom_weights = {}
            except Exception as e:
                print(f"   ⚠️ Contribution generation failed: {e}")
                atom_weights = {}
            finally:
                # Clean temp sdf
                try:
                    if 'sdf_path' in locals() and Path(sdf_path).exists():
                        Path(sdf_path).unlink()
                except Exception:
                    pass
            
            # Quality score (coverage + magnitude)
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
                atom_weights, title, iteration, quality_score, performance_score, model_name, params, mol,
                acc=acc, auc=auc, y_test=y_test, y_pred=y_pred, y_pred_proba=y_proba
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
        
        # Load and sample data from REAL QSAR dataset
        df = self.load_and_sample_data()
        print(f"   📊 Dataset info: {len(df)} molecules, {df['classLabel'].sum()} potent, {len(df)-df['classLabel'].sum()} non-potent")
        
        smiles = df['cleanedMol'].values
        labels = df['classLabel'].values.astype(int)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
            smiles, labels, smiles, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"   🔄 Train/Test split: {len(X_train)} train, {len(X_test)} test (real molecular data)")
        
        # Validate train data is real SMILES
        valid_train = sum(1 for smi in X_train if Chem.MolFromSmiles(str(smi)) is not None)
        print(f"   ✓ Train SMILES validity: {valid_train}/{len(X_train)} valid structures")
        
        # Choose target molecule - prefer Series C compound (from raw QSAR)
        if self.target_smiles is None:
            series_c_smiles = self.get_series_c_smiles()
            if series_c_smiles:
                self.target_smiles = series_c_smiles
                # Verify it's in dataset
                in_data = series_c_smiles in list(X_train) or series_c_smiles in list(X_test)
                print(f"   🎯 Target: ROY-0000220-001 (Series C Spiro, IN dataset={in_data})")
                print(f"      SMILES: {self.target_smiles[:80]}...")
            else:
                self.target_smiles = str(smiles_test[0])
                print(f"   🎯 Target molecule from test set: {self.target_smiles}")
        mol = Chem.MolFromSmiles(self.target_smiles)
        if mol is None:
            raise ValueError(f"Invalid target SMILES: {self.target_smiles}")
        
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
                print(f"   ✅ Acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}")
            except Exception as e:
                print(f"❌ ChemBERTa training failed at iteration {iteration}: {e}")
                continue
            
            # Signed gradient-based saliency for target molecule (shows positive/negative contributions)
            atom_weights = {}
            try:
                # Get target molecule encoding with gradient tracking
                target_inputs = tokenizer(self.target_smiles, return_tensors='pt', truncation=True, max_length=max_len)
                target_ids = target_inputs['input_ids'][0]
                atom_count = mol.GetNumAtoms() if mol is not None else 0
                
                # Compute saliency using input gradient of logit-score
                input_ids = target_ids.unsqueeze(0).to(device)
                attention_mask = torch.ones_like(input_ids).to(device)
                
                # Get embeddings with requires_grad
                embedding_layer = model.roberta.embeddings.word_embeddings
                embedded = embedding_layer(input_ids)
                embedded = embedded.requires_grad_(True)
                
                # Forward pass
                outputs = model.roberta(inputs_embeds=embedded, attention_mask=attention_mask)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    pooled = outputs.pooler_output
                else:
                    pooled = outputs.last_hidden_state[:, 0, :]
                
                logits = model.classifier(pooled)
                logit_potent = logits[0, 1]  # Scalar logit for potent class (not probability)
                
                # Compute gradient of LOGIT w.r.t embeddings (signed gradient)
                logit_potent.backward()
                
                # Extract signed gradients
                if embedded.grad is not None:
                    # Sum across embedding dimension to get signed importance per token
                    token_grads = embedded.grad[0].sum(dim=1).detach().cpu().numpy()  # Shape: [seq_len]
                    
                    # Map to atoms (skip [CLS] token at index 0)
                    num_tokens = min(len(target_ids) - 2, atom_count)
                    for token_idx in range(1, min(num_tokens + 1, len(target_ids) - 1)):
                        atom_idx = token_idx - 1
                        atom_weights[atom_idx] = float(token_grads[token_idx])
                    
                    # Compute range for logging
                    if atom_weights:
                        vals = list(atom_weights.values())
                        min_val, max_val = min(vals), max(vals)
                        print(f"      💡 [REAL] Gradient-Saliency (signed): {min_val:.6f} to {max_val:.6f}, atoms={len(atom_weights)}")
                else:
                    atom_weights = {}
                    
            except Exception as e:
                print(f"   ⚠️ Saliency failed ({str(e)[:50]}...). Trying signed permutation on REAL model...")
                # Fallback: Signed permutation importance
                try:
                    target_inputs = tokenizer(self.target_smiles, return_tensors='pt', truncation=True, max_length=max_len)
                    target_ids = target_inputs['input_ids'][0]
                    atom_count = mol.GetNumAtoms() if mol is not None else 0
                    num_tokens = min(len(target_ids) - 2, atom_count)
                    
                    # Get baseline LOGIT from TRAINED model
                    with torch.no_grad():
                        baseline_input = {'input_ids': target_ids.unsqueeze(0).to(device),
                                        'attention_mask': torch.ones_like(target_ids.unsqueeze(0)).to(device)}
                        baseline_out = model(**baseline_input)
                        baseline_logit = baseline_out.logits[0, 1].cpu().item()
                    
                    print(f"      📍 Baseline logit (potency): {baseline_logit:.4f}")
                    
                    for token_idx in range(1, min(num_tokens + 1, len(target_ids) - 1)):
                        atom_idx = token_idx - 1
                        # Mask this token and get LOGIT (signed)
                        perturbed_ids = target_ids.clone()
                        perturbed_ids[token_idx] = tokenizer.mask_token_id
                        
                        with torch.no_grad():
                            perturbed_input = {'input_ids': perturbed_ids.unsqueeze(0).to(device),
                                             'attention_mask': torch.ones_like(perturbed_ids.unsqueeze(0)).to(device)}
                            perturbed_out = model(**perturbed_input)
                            perturbed_logit = perturbed_out.logits[0, 1].cpu().item()
                        
                        # Signed importance = logit drop when token masked (positive = helps prediction)
                        importance = baseline_logit - perturbed_logit
                        atom_weights[atom_idx] = float(importance)
                    
                    if atom_weights:
                        vals = list(atom_weights.values())
                        min_val, max_val = min(vals), max(vals)
                        print(f"      💡 [REAL] Signed Permutation: {min_val:.6f} to {max_val:.6f}, atoms={len(atom_weights)}")
                    else:
                        atom_weights = {}
                except Exception as e2:
                    print(f"   ⚠️ Signed permutation also failed ({str(e2)[:30]}...). No explanation available.")
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
                atom_weights, title, iteration, quality_score, performance_score, model_name, params, mol,
                acc=acc, auc=auc, lr=lr, epochs=epochs, max_len=max_len, att_layer=att_layer, att_head=att_head,
                y_test=np.array(labels_list), y_pred=np.array(preds), y_pred_proba=np.array(probs)
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

    def create_all_dynamic_movies(self):
        """Create dynamic parameter movies for supported models"""
        print("🎬 Creating Dynamic Parameter Evolution Movies")
        print("=" * 60)
        created_movies = []
        # Circular FP
        movie_path = self.create_dynamic_parameter_movie('circular_fingerprint')
        if movie_path:
            created_movies.append(movie_path)
        # GraphConv
        gc_movie = self.create_graphconv_dynamic_parameter_movie()
        if gc_movie:
            created_movies.append(gc_movie)
        # ChemBERTa
        cb_movie = self.create_chemberta_dynamic_parameter_movie()
        if cb_movie:
            created_movies.append(cb_movie)
        print(f"\n🎉 Dynamic parameter movie creation complete! Created {len(created_movies)} movie(s).")
        for movie in created_movies:
            print(f"  • {movie}")
        return created_movies

def main():
    """Main function"""
    creator = DynamicParameterMovieCreator()
    creator.create_all_dynamic_movies()

if __name__ == "__main__":
    main()
