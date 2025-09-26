#!/usr/bin/env python3
"""
Dynamic Parameter Evolution Movie Creator (SHAP-local highlighting)
==================================================================

Creates a GIF identical in layout to the LIME-based/SHAP-mixed version, but uses
SHAP values both for structure highlighting (local to the target molecule) and for
fragment-quality / quantitative analysis. The previous GIFs remain untouched.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

# Reuse renderer, metrics, collision utilities, etc.
from create_dynamic_parameter_movies import DynamicParameterMovieCreator

class ShapLocalDynamicParameterMovieCreator(DynamicParameterMovieCreator):
    def _compute_shap_values(self, model, X_background: np.ndarray, X_eval: np.ndarray):
        """Return per-sample SHAP values for class 1 as (n_samples, n_features).
        Falls back to None if SHAP isn't available/supported.
        """
        try:
            import shap
        except Exception:
            return None

        # Limit background size for speed
        bg = X_background[: min(100, len(X_background))] if X_background is not None and len(X_background) > 0 else X_eval[: min(50, len(X_eval))]
        # Try auto explainer first
        try:
            explainer = shap.Explainer(model, bg)
            explanation = explainer(X_eval)
            values = getattr(explanation, "values", None)
            if values is None:
                return None
            arr = np.array(values)
            if arr.ndim == 3 and arr.shape[-1] >= 2:
                arr = arr[:, :, 1]  # class 1
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            return arr
        except Exception:
            pass
        # Fallback: LinearExplainer
        try:
            import shap
            explainer = shap.LinearExplainer(model, bg)
            vals = explainer.shap_values(X_eval)
            if isinstance(vals, list) and len(vals) >= 2:
                return np.array(vals[1])
            return np.array(vals)
        except Exception:
            return None

    def create_dynamic_parameter_movie_shap_local(self, model_name: str = 'circular_fingerprint'):
        if model_name != 'circular_fingerprint':
            raise ValueError("This pipeline currently supports only 'circular_fingerprint'.")

        print(f"🎬 Creating SHAP-local dynamic parameter movie for {model_name}")
        frames = []
        script_dir = Path(__file__).resolve().parent
        frame_dir = script_dir / "frames"
        frame_dir.mkdir(exist_ok=True)

        # Load and sample once
        df = self.load_and_sample_data()
        smiles = df['cleanedMol'].values
        labels = df['classLabel'].values

        # Reset evolutions
        self.quality_evolution['circular_fingerprint'] = []
        self.performance_evolution['circular_fingerprint'] = []

        from sklearn.model_selection import train_test_split
        for iteration in range(self.max_iters):
            params = self.get_parameters_for_iteration(model_name, iteration)
            radius = int(params['radius'])
            nBits = int(params['nBits'])
            useFeatures = bool(params.get('useFeatures', False))
            useChirality = bool(params.get('useChirality', False))

            print(f"\n🔧 [SHAP-LOCAL] Iteration {iteration} params: radius={radius}, nBits={nBits}, useFeatures={useFeatures}, useChirality={useChirality}")

            # Collision stats for current FP config
            bit_to_frag_counts, bit_active_counts, total_frags = self._compute_bit_fragment_stats(
                smiles_list=smiles, radius=radius, nBits=nBits, useFeatures=useFeatures, useChirality=useChirality
            )
            collision_stats = self._compute_collision_metrics(bit_to_frag_counts, bit_active_counts, total_frags, nBits)

            # Features and split
            X = self.featurize(smiles, radius, nBits, useFeatures, useChirality)
            y = labels
            X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
                X, y, smiles, test_size=0.2, random_state=42, stratify=y
            )

            # Train model (TPOT if enabled, else LogisticRegression fallback)
            pipeline = None
            try:
                if self.use_tpot:
                    from tpot import TPOTClassifier
                    tpot = TPOTClassifier(generations=2, population_size=8, cv=3, scoring='accuracy', random_state=42, verbosity=2, n_jobs=1, max_time_mins=1)
                    tpot.fit(X_train, y_train)
                    pipeline = tpot.fitted_pipeline_
                else:
                    raise ImportError("USE_TPOT=0")
            except Exception:
                pipeline = self._train_sklearn_fallback(X_train, y_train)

            # Evaluate and make bottom panels
            from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
            try:
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                try:
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                except Exception:
                    y_proba, auc = None, np.nan
                cm = confusion_matrix(y_test, y_pred)
                self.performance_evolution[model_name].append(float(acc))
                rocpr_img, auc_val, ap_val = self._plot_roc_pr(y_test, y_proba)
                cm_img = self._plot_confusion_matrix(cm)
                acc_img = self._plot_accuracy_evolution(model_name)
                self.last_auc[model_name] = auc_val
                self.last_auprc[model_name] = ap_val
                print(f"   ✅ Acc={acc:.3f}, AUC={auc if not np.isnan(auc) else 'NA'}")
            except Exception:
                acc = 0.0
                self.performance_evolution[model_name].append(float(acc))
                y_proba = None
                rocpr_img, auc_val, ap_val = self._plot_roc_pr(np.array([0, 1]), np.array([0.0, 1.0]))
                cm_img = self._plot_confusion_matrix(np.array([[0, 0], [0, 0]]))
                acc_img = self._plot_accuracy_evolution(model_name)
                self.last_auc[model_name] = auc_val
                self.last_auprc[model_name] = ap_val

            # Choose consistent target molecule
            if self.target_smiles is None:
                self.target_smiles = str(smiles_test[0])
                print(f"   🎯 Target molecule selected for visualization: {self.target_smiles}")
            from rdkit import Chem
            target_mol = Chem.MolFromSmiles(self.target_smiles)

            # Compute SHAP values for test set
            shap_vals = self._compute_shap_values(pipeline, X_train, X_test)
            if shap_vals is None:
                # Fallback to permutation-importance-like weights (keeps pipeline running)
                signed_imps = self.compute_signed_feature_importance(pipeline, X_test, y_test)
                shap_vals = np.tile(signed_imps.reshape(1, -1), (len(X_test), 1))

            # Local SHAP for the target instance used for highlighting
            target_idx = 0
            x_target = self.featurize([self.target_smiles], radius, nBits, useFeatures, useChirality)[0]
            active_indices = np.where(x_target > 0)[0]
            feature_weights_target = {}
            if len(active_indices) > 0:
                vals_t = shap_vals[target_idx]
                pairs = [(int(j), float(vals_t[j])) for j in active_indices]
                pairs.sort(key=lambda t: abs(t[1]), reverse=True)
                top_k = min(100, len(pairs))
                feature_weights_target = {j: w for j, w in pairs[:top_k] if abs(w) > 0}
            # Collision-adjusted attribution (D)
            feature_weights_target = self._apply_collision_adjusted_attributions(
                feature_weights_target, collision_stats['per_bit_fragment_counts']
            )

            # Map to atom weights
            atom_weights = {}
            if target_mol is not None and feature_weights_target:
                atom_weights = self.map_fragment_weights_to_atoms(
                    target_mol, feature_weights_target, radius=radius, nBits=nBits,
                    useFeatures=useFeatures, useChirality=useChirality
                )

            # Quality score (coverage + magnitude)
            if target_mol is not None and atom_weights:
                aw = np.array(list(atom_weights.values()), dtype=float)
                coverage = len(atom_weights) / max(1, target_mol.GetNumAtoms())
                magnitude = float(np.mean(np.abs(aw) / (np.max(np.abs(aw)) + 1e-9)))
                quality_score = float(0.5 * coverage + 0.5 * magnitude)
            else:
                quality_score = 0.0
            self.quality_evolution[model_name].append(quality_score)

            # Dataset-level advanced metrics using SHAP
            dataset_metrics = []
            for i, smi in enumerate(smiles_test):
                m_i = Chem.MolFromSmiles(smi)
                if m_i is None:
                    continue
                x_i = self.featurize([smi], radius, nBits, useFeatures, useChirality)[0]
                active_i = np.where(x_i > 0)[0]
                if active_i.size == 0:
                    continue
                vals_i = shap_vals[i]
                fw_i = {int(j): float(vals_i[j]) for j in active_i if abs(vals_i[j]) > 0}
                fw_i = self._apply_collision_adjusted_attributions(fw_i, collision_stats['per_bit_fragment_counts'])
                atom_w_i = self.map_fragment_weights_to_atoms(
                    m_i, fw_i, radius=radius, nBits=nBits, useFeatures=useFeatures, useChirality=useChirality
                )
                def _proba_fn_local(smiles_s: str):
                    try:
                        Xs = self.featurize([smiles_s], radius, nBits, useFeatures, useChirality)
                        return float(pipeline.predict_proba(Xs)[0, 1])
                    except Exception:
                        return 0.0
                m_dict = self._compute_advanced_quality(atom_w_i, m_i, proba_fn=_proba_fn_local)
                dataset_metrics.append(m_dict)
            dataset_adv = self._reduce_metrics_list(dataset_metrics)

            # Target advanced metrics + collision stats
            def _circ_proba_fn(smiles_s: str):
                try:
                    Xs = self.featurize([smiles_s], radius, nBits, useFeatures, useChirality)
                    return float(pipeline.predict_proba(Xs)[0, 1])
                except Exception:
                    return 0.0
            advanced_metrics = self._compute_advanced_quality(atom_weights, target_mol, proba_fn=_circ_proba_fn) or {}
            advanced_metrics['COLLISION_RATE'] = collision_stats.get('collision_rate_mean', 0.0)
            advanced_metrics['AVG_FRAGMENTS_PER_ACTIVE_BIT'] = collision_stats.get('avg_frags_per_active_bit', 0.0)
            advanced_metrics['BIT_ENTROPY'] = collision_stats.get('bit_entropy_mean', 0.0)

            # Draw frame (identical layout)
            title = "Dynamic Parameter Optimization - Circular Fingerprint"
            canvas = self.draw_molecule_with_dynamic_parameters(
                atom_weights, title, iteration, quality_score, float(acc), model_name, params, target_mol,
                auc_val=self.last_auc[model_name], auprc_val=self.last_auprc[model_name],
                acc_img=acc_img, cm_img=cm_img, rocpr_img=rocpr_img,
                advanced_metrics=advanced_metrics, dataset_adv_metrics=dataset_adv
            )
            if canvas is not None:
                frame_path = frame_dir / f"{model_name}_dynamic_frame_shaplocal_{iteration:02d}.png"
                canvas.save(frame_path)
                frames.append(str(frame_path))

        if not frames:
            print(f"❌ No frames created for {model_name} (SHAP-local)")
            return None

        # Assemble GIF
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        fig, ax = plt.subplots(figsize=(6 * self.scale, 4 * self.scale), dpi=200)
        ax.axis('off')
        fig.subplots_adjust(top=0.90, bottom=0.10)
        def animate(frame_idx):
            ax.clear(); ax.axis('off')
            if frame_idx < len(frames):
                img = plt.imread(frames[frame_idx]); ax.imshow(img)
                progress = (frame_idx + 1) / len(frames)
                for t in list(fig.texts):
                    t.remove()
                fig.text(0.985, 0.02, f"Agentic Progress: {progress:.0%} ({frame_idx+1}/{len(frames)})",
                         ha='right', va='bottom', fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.95))
        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=4000, blit=False, repeat=True)
        movie_path = (Path(__file__).resolve().parent / f"{model_name}_dynamic_parameters_shap_local.gif")
        print(f"💾 Saving SHAP-local movie to {movie_path}")
        ani.save(str(movie_path), writer='pillow', fps=1.0)
        plt.close()

        # Cleanup frames
        for fp in frames:
            try:
                Path(fp).unlink()
            except Exception:
                pass
        try:
            frame_dir.rmdir()
        except Exception:
            pass
        return movie_path

if __name__ == "__main__":
    m = ShapLocalDynamicParameterMovieCreator()
    p = m.create_dynamic_parameter_movie_shap_local('circular_fingerprint')
    print("✅ Saved:", p)
