"""
Explanation Quality Evaluator for GraphConv Model
================================================

This script integrates with the existing GraphConv model to evaluate
the quality of molecular explanations and provide optimization feedback.
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import DeepChem and other dependencies
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from explanation_quality_metrics import ExplanationQualityEvaluator, save_quality_report

def load_model_and_data():
    """Load the trained GraphConv model and test data."""
    print("📂 Loading GraphConv model and data...")
    
    # Load data
    data_path = 'data/StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx'
    df = pd.read_excel(data_path)
    
    # Sample data (same as in training)
    class_0 = df[df['classLabel'] == 0].sample(n=50, random_state=42)
    class_1 = df[df['classLabel'] == 1].sample(n=50, random_state=42)
    sampled_df = pd.concat([class_0, class_1], ignore_index=True)
    
    smiles = sampled_df['cleanedMol'].tolist()
    labels = sampled_df['classLabel'].values
    
    # Create dataset
    featurizer = dc.feat.ConvMolFeaturizer()
    X = featurizer.featurize(smiles)
    dataset = dc.data.NumpyDataset(X=X, y=labels, ids=smiles)
    
    # Split data (same as training)
    train_dataset, test_dataset = dc.splits.RandomSplitter().train_test_split(
        dataset, frac_train=0.8, seed=42
    )
    
    # Try to load existing model
    model_path = "models/graphconv_model.pkl"
    if os.path.exists(model_path):
        print("✅ Loading existing GraphConv model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        print("❌ GraphConv model not found! Please train the model first.")
        return None, None, None
    
    return model, test_dataset, smiles

def generate_explanations_for_evaluation(model, dataset, num_samples=20):
    """Generate explanations for a subset of molecules for evaluation."""
    print(f"🔍 Generating explanations for {num_samples} molecules...")
    
    explanations_data = []
    
    # Get predictions
    predictions = model.predict(dataset)
    
    for i in range(min(num_samples, len(dataset))):
        smiles = dataset.ids[i]
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            continue
        
        prediction = predictions[i][0] if len(predictions[i]) > 0 else 0.5
        
        # Generate atom contributions (using heuristic method)
        atom_contributions = generate_heuristic_contributions(mol, prediction)
        
        explanations_data.append({
            'smiles': smiles,
            'molecule': mol,
            'prediction': prediction,
            'atom_contributions': atom_contributions,
            'true_label': dataset.y[i] if hasattr(dataset, 'y') else 0
        })
    
    return explanations_data

def generate_heuristic_contributions(mol, prediction):
    """
    Generate heuristic atom contributions based on chemical knowledge.
    This is a placeholder - in practice, you'd use gradient-based methods.
    """
    atom_contribs = []
    
    # Calculate molecular properties for context
    mol_logp = Crippen.MolLogP(mol)
    mol_weight = Descriptors.MolWt(mol)
    
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()
        is_aromatic = atom.GetIsAromatic()
        is_ring = atom.IsInRing()
        formal_charge = atom.GetFormalCharge()
        
        # Base contribution factors
        pos_contrib = 0
        if is_aromatic:
            pos_contrib += 0.4  # Aromatic atoms often important
        if atomic_num in [7, 8]:  # N, O atoms
            pos_contrib += 0.3
        if is_ring:
            pos_contrib += 0.2
        if atomic_num == 16:  # Sulfur
            pos_contrib += 0.25
            
        # Negative contribution factors
        neg_contrib = 0
        if atomic_num == 17:  # Chlorine
            neg_contrib += 0.3
        if degree > 4:  # Overcrowded atoms
            neg_contrib += 0.2
        if formal_charge != 0:
            neg_contrib += abs(formal_charge) * 0.2
        if atomic_num == 9:  # Fluorine (can be negative in some contexts)
            neg_contrib += 0.1
            
        # Calculate base contribution
        base_contrib = (pos_contrib - neg_contrib) * prediction
        
        # Add molecular context
        if mol_logp > 5:  # Very lipophilic - some atoms should be negative
            if atomic_num == 6 and not is_aromatic:  # Aliphatic carbon
                base_contrib -= 0.1
        
        # Add controlled randomness for diversity
        context_factor = (mol_logp / 5.0) * (1 - 2 * (atom_idx % 2))
        noise = np.random.normal(0, 0.08)  # Controlled noise
        
        final_contrib = base_contrib + context_factor * 0.15 + noise
        
        # Scale based on prediction strength
        final_contrib *= (0.2 + 0.8 * abs(prediction - 0.5) * 2)
        
        atom_contribs.append(final_contrib)
    
    return atom_contribs

def evaluate_explanation_quality():
    """Main function to evaluate explanation quality."""
    print("🏆 Starting Explanation Quality Evaluation")
    print("=" * 50)
    
    # Load model and data
    model, test_dataset, smiles = load_model_and_data()
    if model is None:
        return
    
    # Generate explanations
    explanations_data = generate_explanations_for_evaluation(model, test_dataset, num_samples=25)
    
    if not explanations_data:
        print("❌ No explanations generated!")
        return
    
    print(f"✅ Generated explanations for {len(explanations_data)} molecules")
    
    # Initialize quality evaluator
    evaluator = ExplanationQualityEvaluator()
    
    # Calculate comprehensive quality score
    quality_results = evaluator.calculate_comprehensive_quality_score(
        explanations_data, model=model
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 EXPLANATION QUALITY RESULTS")
    print("=" * 50)
    
    print(f"\n🎯 Overall Quality Score: {quality_results['overall_quality_score']:.3f}")
    
    print("\n📋 Component Scores:")
    for component, score in quality_results['component_scores'].items():
        print(f"  • {component.replace('_', ' ').title()}: {score:.3f}")
    
    print("\n⚖️ Component Weights:")
    for component, weight in quality_results['weights'].items():
        print(f"  • {component.replace('_', ' ').title()}: {weight:.2f}")
    
    # Detailed metrics
    detailed = quality_results['detailed_metrics']
    
    print("\n🔍 Detailed Analysis:")
    
    # Consistency metrics
    if 'consistency' in detailed:
        consistency = detailed['consistency']
        print(f"  📊 Consistency: {consistency['mean_consistency']:.3f} ± {consistency['std_consistency']:.3f}")
        print(f"     - Number of similarity groups: {consistency['num_groups']}")
    
    # SAR alignment
    if 'sar_alignment' in detailed:
        sar = detailed['sar_alignment']
        print(f"  🧬 SAR Alignment: {sar['overall_sar_alignment']:.3f}")
        print(f"     - Patterns found: {sar['num_patterns_found']}")
        
        if 'pattern_scores' in sar:
            print("     - Pattern-specific scores:")
            for pattern, scores in sar['pattern_scores'].items():
                print(f"       • {pattern}: {scores['alignment_score']:.3f} ({scores['num_matches']} matches)")
    
    # Chemical intuition
    if 'chemical_intuition' in detailed:
        intuition = detailed['chemical_intuition']
        print(f"  ⚗️ Chemical Intuition: {intuition['mean_intuition_score']:.3f} ± {intuition['std_intuition_score']:.3f}")
    
    # Selectivity
    if 'selectivity' in detailed:
        selectivity = detailed['selectivity']
        print(f"  🎯 Selectivity: {selectivity['mean_selectivity']:.3f} ± {selectivity['std_selectivity']:.3f}")
    
    # Stability (if available)
    if 'stability' in detailed and detailed['stability']:
        stability = detailed['stability']
        print(f"  🔄 Stability: {stability['mean_stability']:.3f} ± {stability['std_stability']:.3f}")
    
    # Generate recommendations
    recommendations = generate_optimization_recommendations(quality_results)
    
    print("\n" + "=" * 50)
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Priority: {rec['priority']}")
        print(f"   {rec['description']}")
        if 'implementation' in rec:
            print(f"   Implementation: {rec['implementation']}")
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save quality report
    report_path = os.path.join(output_dir, "explanation_quality_report.json")
    save_quality_report(quality_results, report_path)
    
    # Create and save visualization
    try:
        viz_path = os.path.join(output_dir, "explanation_quality_metrics.png")
        fig = evaluator.visualize_quality_metrics(quality_results, save_path=viz_path)
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
    
    # Save recommendations
    rec_path = os.path.join(output_dir, "optimization_recommendations.json")
    with open(rec_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"💾 Recommendations saved to {rec_path}")
    
    print(f"\n✅ Evaluation complete! Results saved to {output_dir}/")
    
    return quality_results, recommendations

def generate_optimization_recommendations(quality_results):
    """Generate specific recommendations for improving explanation quality."""
    recommendations = []
    
    component_scores = quality_results['component_scores']
    overall_score = quality_results['overall_quality_score']
    
    # Overall score assessment
    if overall_score < 0.5:
        recommendations.append({
            'title': 'Major Model Revision Needed',
            'priority': 'HIGH',
            'description': 'Overall explanation quality is low. Consider fundamental changes to model architecture or training approach.',
            'implementation': 'Try different model architectures, loss functions, or training procedures that explicitly optimize for explainability.'
        })
    elif overall_score < 0.7:
        recommendations.append({
            'title': 'Moderate Improvements Needed',
            'priority': 'MEDIUM',
            'description': 'Explanation quality is moderate. Focus on specific weak components.',
            'implementation': 'Target the lowest-scoring components with specific improvements.'
        })
    
    # Component-specific recommendations
    if component_scores.get('consistency', 0) < 0.6:
        recommendations.append({
            'title': 'Improve Explanation Consistency',
            'priority': 'HIGH',
            'description': 'Explanations vary significantly for similar molecules. This suggests the model lacks robustness.',
            'implementation': 'Add consistency regularization terms to loss function or use ensemble methods to improve stability.'
        })
    
    if component_scores.get('sar_alignment', 0) < 0.6:
        recommendations.append({
            'title': 'Enhance SAR Alignment',
            'priority': 'HIGH',
            'description': 'Explanations do not align well with known structure-activity relationships.',
            'implementation': 'Incorporate domain knowledge through feature engineering, knowledge-guided training, or constraint-based learning.'
        })
    
    if component_scores.get('chemical_intuition', 0) < 0.6:
        recommendations.append({
            'title': 'Improve Chemical Intuition',
            'priority': 'MEDIUM',
            'description': 'Explanations do not follow chemical intuition and domain knowledge.',
            'implementation': 'Use chemical feature engineering, incorporate pharmacophore information, or add chemical knowledge constraints.'
        })
    
    if component_scores.get('selectivity', 0) < 0.5:
        recommendations.append({
            'title': 'Increase Explanation Selectivity',
            'priority': 'MEDIUM',
            'description': 'Explanations are not selective enough - too many atoms highlighted.',
            'implementation': 'Add sparsity regularization to explanation method or use attention mechanisms with sharpening.'
        })
    
    if component_scores.get('stability', 0) < 0.7 and 'stability' in component_scores:
        recommendations.append({
            'title': 'Improve Explanation Stability',
            'priority': 'MEDIUM',
            'description': 'Explanations are not stable under small perturbations.',
            'implementation': 'Use noise injection during training, ensemble methods, or smooth gradient techniques.'
        })
    
    # Specific technical recommendations
    recommendations.append({
        'title': 'Implement Gradient-Based Explanations',
        'priority': 'HIGH',
        'description': 'Current heuristic explanations should be replaced with proper gradient-based methods.',
        'implementation': 'Implement integrated gradients, GradCAM for graphs, or attention mechanisms for more accurate attributions.'
    })
    
    recommendations.append({
        'title': 'Add Explanation-Aware Training',
        'priority': 'MEDIUM',
        'description': 'Train the model to produce better explanations by incorporating explanation quality in the loss.',
        'implementation': 'Add explanation consistency, sparsity, or domain knowledge terms to the training objective.'
    })
    
    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
    
    return recommendations

if __name__ == "__main__":
    # Run the evaluation
    try:
        quality_results, recommendations = evaluate_explanation_quality()
        print("\n🎉 Explanation quality evaluation completed successfully!")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
