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
        
        # Define parameter exploration spaces for circular fingerprints (Morgan)
        self.parameter_spaces = {
            'circular_fingerprint': {
                'radius': [1, 2, 3, 4, 2, 3],
                'nBits': [1024, 2048, 4096, 2048, 1024, 2048]
            }
        }
        
        # Placeholders (no longer used for simulation); computed dynamically now
        self.quality_evolution = {'circular_fingerprint': []}
        self.performance_evolution = {'circular_fingerprint': []}

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

    # ------------------------ Drawing ------------------------
    def draw_molecule_with_dynamic_parameters(self, contributions, title, iteration, 
                                            quality_score, performance_score, model_type, current_params, mol):
        """Draw molecule with dynamically changing parameters (layout preserved)"""
        if mol is None:
            return None
        
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 400)
        
        # Normalize contributions for coloring
        num_atoms = mol.GetNumAtoms()
        contrib_array = np.zeros(num_atoms, dtype=float)
        for idx, w in contributions.items():
            if 0 <= idx < num_atoms:
                contrib_array[idx] = w
        
        if num_atoms > 0:
            max_abs = max(1e-9, float(np.max(np.abs(contrib_array))))
            normalized_contribs = [0.5 + 0.5 * (c / max_abs) for c in contrib_array]  # map [-1,1] -> [0,1]
        else:
            normalized_contribs = []
        
        # Set atom colors based on contributions
        atom_colors = {}
        for i, contrib in enumerate(normalized_contribs):
            # Positive -> blue scale, Negative -> red scale, Neutral -> gray
            if contrib >= 0.55:  # positive
                strength = (contrib - 0.55) / 0.45
                atom_colors[i] = (0.2 * (1 - strength), 0.2 * (1 - strength), 1.0 * (0.6 + 0.4 * strength))
            elif contrib <= 0.45:  # negative
                strength = (0.45 - contrib) / 0.45
                atom_colors[i] = (1.0 * (0.6 + 0.4 * strength), 0.2 * (1 - strength), 0.2 * (1 - strength))
            else:
                atom_colors[i] = (0.85, 0.85, 0.85)
        
        drawer.SetFontSize(14)
        drawer.DrawMolecule(mol, highlightAtoms=list(range(num_atoms)), highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        canvas = Image.new('RGB', (900, 650), 'white')
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
        
        # Metrics
        draw.text((350, 460), f"Iteration: {iteration}", fill='black', font=metric_font)
        draw.text((20, 480), f"Quality: {quality_score:.3f}", fill='blue', font=small_font)
        draw.text((150, 480), f"Performance: {performance_score:.3f}", fill='green', font=small_font)
        draw.text((300, 480), f"Combined: {0.6*performance_score + 0.4*quality_score:.3f}", 
                 fill='purple', font=small_font)
        
        # Current parameters box
        param_y_start = 500
        draw.text((20, param_y_start), "🔧 Current Parameters:", fill='black', font=small_font)
        y_offset = param_y_start + 20
        for name, val in current_params.items():
            draw.text((25, y_offset), f"• {name}: {val}", fill='darkblue', font=param_font)
            y_offset += 12
        
        # Parameter evolution tracking (right side)
        param_box_x = 600
        draw.rectangle([param_box_x-5, 80, param_box_x+290, 450], outline='black', width=1)
        draw.text((param_box_x, 85), "📊 Parameter Evolution History:", fill='black', font=small_font)
        
        history_y = 105
        hist_len = min(len(self.performance_evolution['circular_fingerprint']), 4)
        start_idx = max(0, len(self.performance_evolution['circular_fingerprint']) - hist_len)
        for idx in range(start_idx, start_idx + hist_len):
            draw.text((param_box_x, history_y), f"Iteration {idx}:", fill='black', font=param_font)
            history_y += 12
            params = self.get_parameters_for_iteration('circular_fingerprint', idx)
            for pname, pval in params.items():
                draw.text((param_box_x + 10, history_y), f"• {pname}: {pval}", fill='darkgreen', font=param_font)
                history_y += 11
            q = self.quality_evolution['circular_fingerprint'][idx]
            draw.text((param_box_x + 10, history_y), f"→ Quality: {q:.3f}", fill='blue', font=param_font)
            history_y += 20
        
        # Strategy (kept)
        strategy_y = 350
        draw.text((param_box_x, strategy_y), "🎯 Agentic Strategy:", fill='black', font=small_font)
        strategy_info = [
            "• Exploring radius 1-4 for coverage",
            "• Testing nBits 1024-4096 for detail",
            "• TPOT optimizes downstream model"
        ]
        strat_y = strategy_y + 15
        for info in strategy_info:
            draw.text((param_box_x, strat_y), info, fill='purple', font=param_font)
            strat_y += 12
        
        # Legend
        legend_x = 600
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
        
        return canvas

    # ------------------------ Parameter helpers ------------------------
    def get_parameters_for_iteration(self, model_type, iteration):
        params = {}
        for param_name, values in self.parameter_spaces[model_type].items():
            params[param_name] = values[iteration % len(values)]
        return params

    # ------------------------ Training & Movie Creation ------------------------
    def create_dynamic_parameter_movie(self, model_name):
        """Create movie showing parameter changes with real TPOT training"""
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
            
            # Train TPOT
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
                print("❌ TPOT not installed. Please install with: pip install tpot")
                return None
            except Exception as e:
                print(f"❌ TPOT training failed at iteration {iteration}: {e}")
                continue
            
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
                atom_weights, title, iteration, quality_score, performance_score, model_name, params, mol
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

    def create_all_dynamic_movies(self):
        """Create dynamic parameter movie for circular fingerprint only"""
        print("🎬 Creating Dynamic Parameter Evolution Movies")
        print("=" * 60)
        created_movies = []
        movie_path = self.create_dynamic_parameter_movie('circular_fingerprint')
        if movie_path:
            created_movies.append(movie_path)
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
