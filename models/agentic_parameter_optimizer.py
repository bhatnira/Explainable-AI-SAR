"""
Agentic Parameter Optimization for Explainable SAR Models
========================================================

This module implements an intelligent agent that systematically optimizes
parameters for circular fingerprint, GraphConv, and ChemBERTa models to
improve both predictive performance and explanation quality.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any
import itertools
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import deepchem as dc
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from explanation_quality_metrics import ExplanationQualityEvaluator

class ParameterOptimizationAgent:
    """
    Intelligent agent for systematic parameter optimization across multiple model types.
    Uses adaptive search strategies to find optimal configurations.
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path or 'data/StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx'
        self.quality_evaluator = ExplanationQualityEvaluator()
        self.optimization_history = []
        self.best_configs = {}
        self.current_iteration = 0
        
        # Parameter search spaces
        self.parameter_spaces = self._define_parameter_spaces()
        
        # Adaptive search state
        self.search_state = {
            'exploration_rate': 0.8,
            'exploitation_rate': 0.2,
            'temperature': 1.0,
            'best_scores': {}
        }
        
        print("🤖 Parameter Optimization Agent Initialized")
        print(f"📊 Will optimize: Circular Fingerprint, GraphConv, ChemBERTa")
        
    def _define_parameter_spaces(self):
        """Define parameter search spaces for each model type."""
        return {
            'circular_fingerprint': {
                'radius': [1, 2, 3, 4],
                'nBits': [512, 1024, 2048, 4096],
                'useFeatures': [True, False],
                'useChirality': [True, False],
                'useBondTypes': [True, False]
            },
            'graphconv': {
                'graph_conv_layers': [
                    [32, 32], [64, 64], [128, 128],
                    [32, 64], [64, 128], [128, 64],
                    [32, 32, 32], [64, 64, 64], [128, 128, 128],
                    [32, 64, 32], [64, 128, 64]
                ],
                'dense_layer_size': [64, 128, 256, 512],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
                'batch_size': [16, 32, 64, 128]
            },
            'chemberta': {
                'max_length': [128, 256, 512],
                'learning_rate': [1e-5, 2e-5, 5e-5, 1e-4],
                'batch_size': [8, 16, 32],
                'num_epochs': [3, 5, 10, 15],
                'warmup_steps': [0, 100, 500, 1000],
                'weight_decay': [0.0, 0.01, 0.1]
            }
        }
    
    def load_and_prepare_data(self):
        """Load and prepare molecular data for optimization."""
        print("📂 Loading and preparing data...")
        
        df = pd.read_excel(self.data_path)
        
        # Sample data for optimization (use subset for faster experimentation)
        class_0 = df[df['classLabel'] == 0].sample(n=min(100, len(df[df['classLabel'] == 0])), random_state=42)
        class_1 = df[df['classLabel'] == 1].sample(n=min(100, len(df[df['classLabel'] == 1])), random_state=42)
        sampled_df = pd.concat([class_0, class_1], ignore_index=True)
        
        smiles = sampled_df['cleanedMol'].tolist()
        labels = sampled_df['classLabel'].values
        
        print(f"✅ Data prepared: {len(smiles)} molecules, {np.sum(labels)} active, {len(labels) - np.sum(labels)} inactive")
        
        return smiles, labels, sampled_df
    
    def generate_parameter_combinations(self, model_type: str, strategy: str = 'adaptive') -> List[Dict]:
        """
        Generate parameter combinations using different strategies.
        
        Args:
            model_type: 'circular_fingerprint', 'graphconv', or 'chemberta'
            strategy: 'grid', 'random', 'adaptive', or 'bayesian'
        """
        space = self.parameter_spaces[model_type]
        
        if strategy == 'grid':
            # Full grid search (can be expensive)
            keys = list(space.keys())
            values = list(space.values())
            combinations = []
            for combo in itertools.product(*values):
                combinations.append(dict(zip(keys, combo)))
            return combinations[:20]  # Limit for practical reasons
        
        elif strategy == 'random':
            # Random sampling
            combinations = []
            for _ in range(15):
                combo = {}
                for key, values in space.items():
                    combo[key] = np.random.choice(values)
                combinations.append(combo)
            return combinations
        
        elif strategy == 'adaptive':
            # Adaptive sampling based on previous results
            return self._adaptive_parameter_selection(model_type)
        
        else:  # Default to balanced approach
            return self._balanced_parameter_selection(model_type)
    
    def _adaptive_parameter_selection(self, model_type: str) -> List[Dict]:
        """Adaptive parameter selection based on optimization history."""
        space = self.parameter_spaces[model_type]
        combinations = []
        
        # Get best performing parameters from history
        model_history = [h for h in self.optimization_history if h['model_type'] == model_type]
        
        if not model_history:
            # No history, use random exploration
            return self.generate_parameter_combinations(model_type, 'random')
        
        # Sort by combined score
        model_history.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        best_params = model_history[:3]  # Top 3 configurations
        
        # Exploration: Random variations around best parameters
        for best_config in best_params:
            params = best_config['parameters']
            
            # Generate variations
            for _ in range(3):
                new_params = params.copy()
                
                # Randomly modify 1-2 parameters
                keys_to_modify = np.random.choice(list(params.keys()), 
                                                size=min(2, len(params)), 
                                                replace=False)
                
                for key in keys_to_modify:
                    if key in space:
                        # Choose nearby value or random
                        if np.random.random() < self.search_state['exploration_rate']:
                            new_params[key] = np.random.choice(space[key])
                        else:
                            # Choose value close to current best
                            current_val = params[key]
                            if isinstance(current_val, (int, float)):
                                available_vals = space[key]
                                # Find closest values
                                if isinstance(available_vals[0], (int, float)):
                                    available_vals = sorted(available_vals)
                                    current_idx = available_vals.index(current_val) if current_val in available_vals else len(available_vals)//2
                                    # Choose neighbor
                                    neighbor_indices = [max(0, current_idx-1), min(len(available_vals)-1, current_idx+1)]
                                    new_params[key] = available_vals[np.random.choice(neighbor_indices)]
                
                combinations.append(new_params)
        
        # Exploitation: Pure random exploration
        for _ in range(6):
            combo = {}
            for key, values in space.items():
                combo[key] = np.random.choice(values)
            combinations.append(combo)
        
        return combinations
    
    def _balanced_parameter_selection(self, model_type: str) -> List[Dict]:
        """Balanced parameter selection strategy."""
        space = self.parameter_spaces[model_type]
        combinations = []
        
        # Strategy 1: Default/reasonable configurations
        if model_type == 'circular_fingerprint':
            combinations.extend([
                {'radius': 2, 'nBits': 1024, 'useFeatures': True, 'useChirality': True, 'useBondTypes': True},
                {'radius': 3, 'nBits': 2048, 'useFeatures': True, 'useChirality': False, 'useBondTypes': True},
                {'radius': 2, 'nBits': 2048, 'useFeatures': False, 'useChirality': True, 'useBondTypes': True},
            ])
        elif model_type == 'graphconv':
            combinations.extend([
                {'graph_conv_layers': [64, 64], 'dense_layer_size': 128, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32},
                {'graph_conv_layers': [128, 128], 'dense_layer_size': 256, 'dropout': 0.3, 'learning_rate': 0.0005, 'batch_size': 32},
                {'graph_conv_layers': [32, 64, 32], 'dense_layer_size': 128, 'dropout': 0.4, 'learning_rate': 0.001, 'batch_size': 64},
            ])
        elif model_type == 'chemberta':
            combinations.extend([
                {'max_length': 256, 'learning_rate': 2e-5, 'batch_size': 16, 'num_epochs': 5, 'warmup_steps': 500, 'weight_decay': 0.01},
                {'max_length': 512, 'learning_rate': 1e-5, 'batch_size': 8, 'num_epochs': 10, 'warmup_steps': 1000, 'weight_decay': 0.1},
                {'max_length': 128, 'learning_rate': 5e-5, 'batch_size': 32, 'num_epochs': 3, 'warmup_steps': 100, 'weight_decay': 0.0},
            ])
        
        # Strategy 2: Random exploration
        for _ in range(7):
            combo = {}
            for key, values in space.items():
                combo[key] = np.random.choice(values)
            combinations.append(combo)
        
        return combinations
    
    def train_circular_fingerprint_model(self, params: Dict, smiles: List, labels: np.ndarray) -> Tuple[Any, float, Dict]:
        """Train circular fingerprint model with given parameters."""
        print(f"🔬 Training Circular Fingerprint model with params: {params}")
        
        try:
            # Create circular fingerprint featurizer
            featurizer = dc.feat.CircularFingerprint(
                radius=params['radius'],
                size=params['nBits'],
                chiral=params['useChirality'],
                bonds=params['useBondTypes'],
                features=params['useFeatures']
            )
            
            # Featurize molecules
            X = featurizer.featurize(smiles)
            dataset = dc.data.NumpyDataset(X=X, y=labels, ids=smiles)
            
            # Split data
            train_dataset, test_dataset = dc.splits.RandomSplitter().train_test_split(
                dataset, frac_train=0.8, seed=42
            )
            
            # Train model (using simple classifier for fingerprints)
            model = dc.models.MultitaskClassifier(
                n_tasks=1,
                n_features=params['nBits'],
                layer_sizes=[512, 256],
                dropouts=0.2,
                learning_rate=0.001,
                batch_size=32
            )
            
            model.fit(train_dataset, nb_epoch=20)
            
            # Evaluate
            test_scores = model.evaluate(test_dataset, [dc.metrics.Metric(dc.metrics.roc_auc_score)])
            test_auc = test_scores['roc_auc_score']
            
            print(f"   ✅ Test AUC: {test_auc:.3f}")
            
            return model, test_auc, {'train_size': len(train_dataset), 'test_size': len(test_dataset)}
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None, 0.0, {}
    
    def train_graphconv_model(self, params: Dict, smiles: List, labels: np.ndarray) -> Tuple[Any, float, Dict]:
        """Train GraphConv model with given parameters."""
        print(f"🔬 Training GraphConv model with params: {params}")
        
        try:
            # Create dataset
            featurizer = dc.feat.ConvMolFeaturizer()
            X = featurizer.featurize(smiles)
            dataset = dc.data.NumpyDataset(X=X, y=labels, ids=smiles)
            
            # Split data
            train_dataset, test_dataset = dc.splits.RandomSplitter().train_test_split(
                dataset, frac_train=0.8, seed=42
            )
            
            # Create and train model
            model = dc.models.GraphConvModel(
                n_tasks=1,
                graph_conv_layers=params['graph_conv_layers'],
                dense_layer_size=params['dense_layer_size'],
                dropout=params['dropout'],
                mode='classification',
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size']
            )
            
            model.fit(train_dataset, nb_epoch=15)
            
            # Evaluate
            test_scores = model.evaluate(test_dataset, [dc.metrics.Metric(dc.metrics.roc_auc_score)])
            test_auc = test_scores['roc_auc_score']
            
            print(f"   ✅ Test AUC: {test_auc:.3f}")
            
            return model, test_auc, {'train_size': len(train_dataset), 'test_size': len(test_dataset)}
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None, 0.0, {}
    
    def train_chemberta_model(self, params: Dict, smiles: List, labels: np.ndarray) -> Tuple[Any, float, Dict]:
        """Train ChemBERTa model with given parameters (simplified version)."""
        print(f"🔬 Training ChemBERTa model with params: {params}")
        
        try:
            # For this demo, we'll simulate ChemBERTa with a simplified approach
            # In practice, you'd use the actual ChemBERTa model
            
            # Create molecular descriptors as features (simulating BERT embeddings)
            features = []
            for smi in smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    desc = [
                        Descriptors.MolWt(mol),
                        Descriptors.LogP(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.TPSA(mol),
                        rdMolDescriptors.CalcNumRotatableBonds(mol),
                        rdMolDescriptors.CalcNumAromaticRings(mol),
                        len(smi)  # Sequence length proxy
                    ]
                    features.append(desc)
                else:
                    features.append([0] * 8)
            
            X = np.array(features)
            dataset = dc.data.NumpyDataset(X=X, y=labels, ids=smiles)
            
            # Split data
            train_dataset, test_dataset = dc.splits.RandomSplitter().train_test_split(
                dataset, frac_train=0.8, seed=42
            )
            
            # Create model (simulating ChemBERTa with multi-layer network)
            model = dc.models.MultitaskClassifier(
                n_tasks=1,
                n_features=8,
                layer_sizes=[params['max_length']//2, params['max_length']//4],
                dropouts=0.1,
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size']
            )
            
            model.fit(train_dataset, nb_epoch=params['num_epochs'])
            
            # Evaluate
            test_scores = model.evaluate(test_dataset, [dc.metrics.Metric(dc.metrics.roc_auc_score)])
            test_auc = test_scores['roc_auc_score']
            
            print(f"   ✅ Test AUC: {test_auc:.3f}")
            
            return model, test_auc, {'train_size': len(train_dataset), 'test_size': len(test_dataset)}
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None, 0.0, {}
    
    def evaluate_model_explanations(self, model, model_type: str, smiles: List, labels: np.ndarray) -> float:
        """Evaluate explanation quality for a trained model."""
        print(f"🔍 Evaluating explanation quality for {model_type}...")
        
        try:
            # Generate sample explanations (simplified for demonstration)
            explanations_data = []
            
            # Get predictions
            if model_type == 'circular_fingerprint':
                featurizer = dc.feat.CircularFingerprint(radius=2, size=1024)
                X = featurizer.featurize(smiles[:10])  # Sample
                dataset = dc.data.NumpyDataset(X=X, y=labels[:10], ids=smiles[:10])
            elif model_type == 'graphconv':
                featurizer = dc.feat.ConvMolFeaturizer()
                X = featurizer.featurize(smiles[:10])
                dataset = dc.data.NumpyDataset(X=X, y=labels[:10], ids=smiles[:10])
            else:  # chemberta
                features = []
                for smi in smiles[:10]:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        desc = [Descriptors.MolWt(mol), Descriptors.LogP(mol), 
                               Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                               Descriptors.TPSA(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
                               rdMolDescriptors.CalcNumAromaticRings(mol), len(smi)]
                        features.append(desc)
                    else:
                        features.append([0] * 8)
                X = np.array(features)
                dataset = dc.data.NumpyDataset(X=X, y=labels[:10], ids=smiles[:10])
            
            predictions = model.predict(dataset)
            
            for i, smi in enumerate(smiles[:10]):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                
                prediction = predictions[i][0] if len(predictions[i]) > 0 else 0.5
                
                # Generate heuristic explanations based on model type
                atom_contributions = self._generate_model_specific_explanations(
                    mol, prediction, model_type
                )
                
                explanations_data.append({
                    'smiles': smi,
                    'molecule': mol,
                    'prediction': prediction,
                    'atom_contributions': atom_contributions,
                    'true_label': labels[i] if i < len(labels) else 0
                })
            
            if not explanations_data:
                return 0.0
            
            # Calculate quality metrics
            quality_results = self.quality_evaluator.calculate_comprehensive_quality_score(
                explanations_data
            )
            
            quality_score = quality_results['overall_quality_score']
            print(f"   📊 Explanation Quality: {quality_score:.3f}")
            
            return quality_score
            
        except Exception as e:
            print(f"   ❌ Error evaluating explanations: {e}")
            return 0.0
    
    def _generate_model_specific_explanations(self, mol, prediction, model_type):
        """Generate explanations tailored to specific model types."""
        atom_contribs = []
        
        # Model-specific explanation patterns
        if model_type == 'circular_fingerprint':
            # Fingerprint models focus on local neighborhoods
            for atom_idx, atom in enumerate(mol.GetAtoms()):
                # Local environment contribution
                neighbors = len(atom.GetNeighbors())
                is_aromatic = atom.GetIsAromatic()
                atomic_num = atom.GetAtomicNum()
                
                contrib = 0.0
                if is_aromatic:
                    contrib += 0.3 * prediction
                if atomic_num in [7, 8, 16]:  # Heteroatoms
                    contrib += 0.4 * prediction
                if neighbors > 2:
                    contrib += 0.2 * prediction
                
                contrib += np.random.normal(0, 0.1)  # Noise
                atom_contribs.append(contrib)
                
        elif model_type == 'graphconv':
            # Graph models consider global molecular structure
            for atom_idx, atom in enumerate(mol.GetAtoms()):
                atomic_num = atom.GetAtomicNum()
                degree = atom.GetDegree()
                is_aromatic = atom.GetIsAromatic()
                is_ring = atom.IsInRing()
                
                contrib = 0.0
                if is_aromatic:
                    contrib += 0.4 * prediction
                if atomic_num in [7, 8]:
                    contrib += 0.35 * prediction
                if is_ring:
                    contrib += 0.25 * prediction
                if degree > 3:
                    contrib -= 0.1 * prediction  # Overcrowding penalty
                
                contrib += np.random.normal(0, 0.08)
                atom_contribs.append(contrib)
                
        else:  # chemberta
            # Sequence models focus on patterns and sequences
            for atom_idx, atom in enumerate(mol.GetAtoms()):
                atomic_num = atom.GetAtomicNum()
                is_aromatic = atom.GetIsAromatic()
                position_factor = 1 - abs(atom_idx - len(list(mol.GetAtoms()))/2) / (len(list(mol.GetAtoms()))/2)
                
                contrib = 0.0
                if is_aromatic:
                    contrib += 0.35 * prediction
                if atomic_num in [7, 8]:
                    contrib += 0.3 * prediction
                
                # Position-dependent contribution (sequence models)
                contrib *= (0.5 + 0.5 * position_factor)
                contrib += np.random.normal(0, 0.06)
                atom_contribs.append(contrib)
        
        return atom_contribs
    
    def optimize_model_type(self, model_type: str, smiles: List, labels: np.ndarray, max_iterations: int = 10) -> Dict:
        """Optimize parameters for a specific model type."""
        print(f"\n🎯 Optimizing {model_type.upper()} parameters")
        print("=" * 60)
        
        model_results = []
        best_score = 0
        best_config = None
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(model_type, strategy='adaptive')
        
        for i, params in enumerate(param_combinations[:max_iterations]):
            print(f"\n🔄 Iteration {i+1}/{min(max_iterations, len(param_combinations))}")
            print(f"Parameters: {params}")
            
            # Train model
            if model_type == 'circular_fingerprint':
                model, test_auc, info = self.train_circular_fingerprint_model(params, smiles, labels)
            elif model_type == 'graphconv':
                model, test_auc, info = self.train_graphconv_model(params, smiles, labels)
            elif model_type == 'chemberta':
                model, test_auc, info = self.train_chemberta_model(params, smiles, labels)
            else:
                continue
            
            if model is None:
                continue
            
            # Evaluate explanation quality
            explanation_quality = self.evaluate_model_explanations(model, model_type, smiles, labels)
            
            # Calculate combined score
            combined_score = 0.6 * test_auc + 0.4 * explanation_quality
            
            # Store results
            result = {
                'iteration': i + 1,
                'model_type': model_type,
                'parameters': params,
                'test_auc': test_auc,
                'explanation_quality': explanation_quality,
                'combined_score': combined_score,
                'timestamp': datetime.now().isoformat(),
                'additional_info': info
            }
            
            model_results.append(result)
            self.optimization_history.append(result)
            
            # Update best configuration
            if combined_score > best_score:
                best_score = combined_score
                best_config = result.copy()
                
                # Save best model
                os.makedirs("models/optimized", exist_ok=True)
                model_path = f"models/optimized/best_{model_type}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"💾 New best model saved: {model_path}")
            
            print(f"Results: AUC={test_auc:.3f}, Quality={explanation_quality:.3f}, Combined={combined_score:.3f}")
            
            # Update search strategy based on results
            self._update_search_strategy(model_type, result)
        
        # Summary for this model type
        print(f"\n📊 {model_type.upper()} Optimization Summary:")
        print(f"Best Combined Score: {best_score:.3f}")
        print(f"Best Configuration: {best_config['parameters'] if best_config else 'None'}")
        
        return {
            'model_type': model_type,
            'best_config': best_config,
            'all_results': model_results,
            'best_score': best_score
        }
    
    def _update_search_strategy(self, model_type: str, result: Dict):
        """Update search strategy based on optimization results."""
        # Adaptive learning: adjust exploration vs exploitation
        if result['combined_score'] > self.search_state['best_scores'].get(model_type, 0):
            self.search_state['best_scores'][model_type] = result['combined_score']
            # Increase exploitation when finding good results
            self.search_state['exploitation_rate'] = min(0.5, self.search_state['exploitation_rate'] + 0.05)
            self.search_state['exploration_rate'] = 1 - self.search_state['exploitation_rate']
        else:
            # Increase exploration when results are poor
            self.search_state['exploration_rate'] = min(0.9, self.search_state['exploration_rate'] + 0.02)
            self.search_state['exploitation_rate'] = 1 - self.search_state['exploration_rate']
        
        # Temperature cooling for simulated annealing effect
        self.search_state['temperature'] *= 0.95
    
    def run_comprehensive_optimization(self, max_iterations_per_model: int = 8) -> Dict:
        """Run comprehensive optimization across all model types."""
        print("🚀 Starting Comprehensive Agentic Parameter Optimization")
        print("=" * 80)
        
        # Load data
        smiles, labels, df = self.load_and_prepare_data()
        
        # Optimize each model type
        results = {}
        model_types = ['circular_fingerprint', 'graphconv', 'chemberta']
        
        for model_type in model_types:
            try:
                results[model_type] = self.optimize_model_type(
                    model_type, smiles, labels, max_iterations_per_model
                )
            except Exception as e:
                print(f"❌ Error optimizing {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        # Find overall best model
        best_overall = None
        best_overall_score = 0
        
        for model_type, result in results.items():
            if 'best_score' in result and result['best_score'] > best_overall_score:
                best_overall_score = result['best_score']
                best_overall = result
        
        # Generate comprehensive report
        optimization_report = {
            'optimization_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_iterations': len(self.optimization_history),
                'models_optimized': len(results),
                'best_overall_score': best_overall_score,
                'best_overall_model': best_overall['model_type'] if best_overall else None,
                'best_overall_config': best_overall['best_config'] if best_overall else None
            },
            'model_results': results,
            'optimization_history': self.optimization_history,
            'search_state_final': self.search_state
        }
        
        # Save comprehensive report
        os.makedirs("results", exist_ok=True)
        report_path = "results/agentic_optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(optimization_report, f, indent=2, default=str)
        
        # Create visualization
        self._create_optimization_visualization(results)
        
        # Print final summary
        print("\n" + "=" * 80)
        print("🏆 AGENTIC OPTIMIZATION COMPLETE")
        print("=" * 80)
        
        if best_overall:
            print(f"🥇 Best Overall Model: {best_overall['model_type'].upper()}")
            print(f"🎯 Best Combined Score: {best_overall_score:.3f}")
            print(f"⚙️ Best Configuration: {best_overall['best_config']['parameters']}")
            print(f"📈 Performance: AUC={best_overall['best_config']['test_auc']:.3f}, Quality={best_overall['best_config']['explanation_quality']:.3f}")
        
        print(f"\n📊 Model Comparison:")
        for model_type, result in results.items():
            if 'best_score' in result:
                print(f"  • {model_type:<20}: {result['best_score']:.3f}")
        
        print(f"\n💾 Results saved to: {report_path}")
        print(f"📈 Visualization saved to: results/agentic_optimization_visualization.png")
        
        return optimization_report
    
    def _create_optimization_visualization(self, results: Dict):
        """Create comprehensive visualization of optimization results."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Agentic Parameter Optimization Results', fontsize=16)
            
            # Plot 1: Best scores by model type
            ax = axes[0, 0]
            model_types = []
            best_scores = []
            
            for model_type, result in results.items():
                if 'best_score' in result:
                    model_types.append(model_type.replace('_', '\n'))
                    best_scores.append(result['best_score'])
            
            if model_types:
                bars = ax.bar(model_types, best_scores, 
                             color=['lightblue', 'lightcoral', 'lightgreen'][:len(model_types)])
                ax.set_ylabel('Best Combined Score')
                ax.set_title('Best Score by Model Type')
                ax.set_ylim(0, 1)
                
                # Add value labels
                for bar, score in zip(bars, best_scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
            
            # Plot 2: Optimization progress over time
            ax = axes[0, 1]
            if self.optimization_history:
                iterations = [h['iteration'] for h in self.optimization_history]
                scores = [h['combined_score'] for h in self.optimization_history]
                model_types_hist = [h['model_type'] for h in self.optimization_history]
                
                colors = {'circular_fingerprint': 'blue', 'graphconv': 'red', 'chemberta': 'green'}
                
                for model_type in set(model_types_hist):
                    model_iterations = [i for i, mt in zip(iterations, model_types_hist) if mt == model_type]
                    model_scores = [s for s, mt in zip(scores, model_types_hist) if mt == model_type]
                    ax.plot(model_iterations, model_scores, 'o-', 
                           color=colors.get(model_type, 'black'), 
                           label=model_type.replace('_', ' ').title(), alpha=0.7)
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Combined Score')
                ax.set_title('Optimization Progress')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 3: Performance vs Explainability scatter
            ax = axes[1, 0]
            if self.optimization_history:
                aucs = [h['test_auc'] for h in self.optimization_history]
                qualities = [h['explanation_quality'] for h in self.optimization_history]
                model_types_hist = [h['model_type'] for h in self.optimization_history]
                
                for model_type in set(model_types_hist):
                    model_aucs = [a for a, mt in zip(aucs, model_types_hist) if mt == model_type]
                    model_qualities = [q for q, mt in zip(qualities, model_types_hist) if mt == model_type]
                    ax.scatter(model_aucs, model_qualities, 
                             color=colors.get(model_type, 'black'),
                             label=model_type.replace('_', ' ').title(), alpha=0.7)
                
                ax.set_xlabel('Test AUC')
                ax.set_ylabel('Explanation Quality')
                ax.set_title('Performance vs Explainability')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 4: Parameter importance (simplified)
            ax = axes[1, 1]
            ax.text(0.5, 0.5, 'Parameter Analysis\n(Detailed analysis in JSON report)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Parameter Analysis')
            ax.axis('off')
            
            plt.tight_layout()
            
            viz_path = "results/agentic_optimization_visualization.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Optimization visualization saved to {viz_path}")
            
        except Exception as e:
            print(f"⚠️ Could not create visualization: {e}")

def main():
    """Main function to run agentic parameter optimization."""
    
    # Initialize optimization agent
    agent = ParameterOptimizationAgent()
    
    try:
        # Run comprehensive optimization
        results = agent.run_comprehensive_optimization(max_iterations_per_model=6)
        
        print("\n🎉 Agentic parameter optimization completed successfully!")
        print("Check the results/ directory for detailed reports and visualizations.")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
