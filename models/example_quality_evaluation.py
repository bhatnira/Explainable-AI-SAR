"""
Simple Example: Using Explanation Quality Metrics
================================================

This script provides a practical example of how to evaluate and improve
explanation quality for your molecular property prediction models.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

def calculate_basic_explanation_metrics(explanations_data):
    """
    Calculate basic explanation quality metrics without heavy dependencies.
    
    Args:
        explanations_data: List of dicts with keys:
            - smiles: SMILES string
            - prediction: Model prediction (0-1)
            - atom_contributions: List of float contributions per atom
            - true_label: True activity label (0 or 1)
    
    Returns:
        dict: Quality metrics
    """
    
    print("📊 Calculating explanation quality metrics...")
    
    metrics = {
        'consistency': 0.0,
        'selectivity': 0.0,
        'chemical_plausibility': 0.0,
        'prediction_agreement': 0.0
    }
    
    if not explanations_data:
        return metrics
    
    # 1. Selectivity Metric
    # Good explanations should highlight only important atoms, not everything
    selectivity_scores = []
    
    for data in explanations_data:
        contribs = np.array(data['atom_contributions'])
        
        # Calculate sparsity (fraction of near-zero contributions)
        threshold = 0.1 * np.std(contribs) if np.std(contribs) > 0 else 0.05
        sparse_fraction = np.mean(np.abs(contribs) < threshold)
        
        # Calculate concentration (are contributions focused?)
        if len(contribs) > 1:
            abs_contribs = np.abs(contribs)
            if np.sum(abs_contribs) > 0:
                normalized = abs_contribs / np.sum(abs_contribs)
                entropy = -np.sum(normalized * np.log(normalized + 1e-10))
                max_entropy = np.log(len(contribs))
                concentration = 1 - (entropy / max_entropy)
            else:
                concentration = 0
        else:
            concentration = 1
        
        selectivity = (sparse_fraction + concentration) / 2
        selectivity_scores.append(selectivity)
    
    metrics['selectivity'] = np.mean(selectivity_scores)
    
    # 2. Chemical Plausibility
    # Check if explanations align with basic chemical intuition
    plausibility_scores = []
    
    for data in explanations_data:
        smiles = data['smiles']
        contribs = data['atom_contributions']
        prediction = data['prediction']
        
        # Basic chemical knowledge checks
        plausibility = assess_chemical_plausibility(smiles, contribs, prediction)
        plausibility_scores.append(plausibility)
    
    metrics['chemical_plausibility'] = np.mean(plausibility_scores)
    
    # 3. Prediction Agreement
    # Explanations should align with predictions (positive prediction = more positive contributions)
    agreement_scores = []
    
    for data in explanations_data:
        contribs = np.array(data['atom_contributions'])
        prediction = data['prediction']
        
        # Calculate overall contribution sentiment
        mean_contrib = np.mean(contribs)
        
        # Agreement: positive prediction should have positive mean contribution
        if prediction > 0.5:  # Predicted active
            agreement = max(0, mean_contrib)  # Reward positive contributions
        else:  # Predicted inactive
            agreement = max(0, -mean_contrib)  # Reward negative contributions
        
        # Normalize to 0-1
        agreement = min(1.0, agreement * 2)
        agreement_scores.append(agreement)
    
    metrics['prediction_agreement'] = np.mean(agreement_scores)
    
    # 4. Consistency (simplified version)
    # Similar molecules should have similar explanation patterns
    consistency_scores = []
    
    # Group by prediction similarity (simplified clustering)
    active_explanations = [d for d in explanations_data if d['prediction'] > 0.5]
    inactive_explanations = [d for d in explanations_data if d['prediction'] <= 0.5]
    
    for group in [active_explanations, inactive_explanations]:
        if len(group) > 1:
            # Calculate pairwise consistency within group
            group_consistency = calculate_group_consistency(group)
            consistency_scores.append(group_consistency)
    
    metrics['consistency'] = np.mean(consistency_scores) if consistency_scores else 0.0
    
    # Calculate overall quality score
    metrics['overall_quality'] = np.mean([
        metrics['consistency'] * 0.25,
        metrics['selectivity'] * 0.25,
        metrics['chemical_plausibility'] * 0.30,
        metrics['prediction_agreement'] * 0.20
    ])
    
    return metrics

def assess_chemical_plausibility(smiles, atom_contributions, prediction):
    """
    Assess chemical plausibility of explanations using basic rules.
    This is a simplified version that doesn't require RDKit.
    """
    
    # Basic heuristics based on SMILES analysis
    plausibility_score = 0.5  # Start with neutral score
    
    # Check for aromatic rings (indicated by lowercase letters in SMILES)
    aromatic_chars = sum(1 for c in smiles if c.islower() and c.isalpha())
    has_aromatics = aromatic_chars > 0
    
    # Check for heteroatoms
    heteroatoms = sum(1 for c in smiles if c in 'NOS')
    has_heteroatoms = heteroatoms > 0
    
    # Check for halogens
    halogens = sum(1 for c in smiles if c in 'FClBrI')
    has_halogens = halogens > 0
    
    contribs = np.array(atom_contributions)
    
    # Rule 1: If molecule has aromatic features and prediction is positive,
    # contributions should generally be positive (aromatic rings often contribute to activity)
    if has_aromatics and prediction > 0.6:
        if np.mean(contribs) > 0:
            plausibility_score += 0.2
        else:
            plausibility_score -= 0.1
    
    # Rule 2: Heteroatoms should have significant contributions (not ignored)
    if has_heteroatoms:
        # Assuming heteroatoms are distributed throughout the molecule,
        # at least some contributions should be significant
        significant_contribs = np.sum(np.abs(contribs) > 0.1)
        if significant_contribs > len(contribs) * 0.3:  # At least 30% significant
            plausibility_score += 0.15
        else:
            plausibility_score -= 0.1
    
    # Rule 3: Very lipophilic molecules (many carbons) might need some negative contributions
    carbon_heavy = sum(1 for c in smiles if c == 'C') > len(smiles) * 0.6
    if carbon_heavy and prediction < 0.4:
        if np.mean(contribs) < 0:
            plausibility_score += 0.1
    
    # Rule 4: Explanations shouldn't be too extreme (all positive or all negative)
    if len(contribs) > 3:
        positive_fraction = np.mean(contribs > 0)
        if 0.2 <= positive_fraction <= 0.8:  # Balanced explanations are more plausible
            plausibility_score += 0.15
    
    return max(0, min(1, plausibility_score))

def calculate_group_consistency(group):
    """Calculate consistency within a group of similar explanations."""
    if len(group) < 2:
        return 0.0
    
    correlations = []
    
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            contrib1 = np.array(group[i]['atom_contributions'])
            contrib2 = np.array(group[j]['atom_contributions'])
            
            # Align to same length (simple truncation)
            min_len = min(len(contrib1), len(contrib2))
            contrib1 = contrib1[:min_len]
            contrib2 = contrib2[:min_len]
            
            # Calculate correlation
            if len(contrib1) > 1 and np.std(contrib1) > 0 and np.std(contrib2) > 0:
                correlation = np.corrcoef(contrib1, contrib2)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
    
    return np.mean(correlations) if correlations else 0.0

def generate_improvement_recommendations(metrics):
    """Generate specific recommendations based on metric scores."""
    
    recommendations = []
    
    # Overall assessment
    overall = metrics['overall_quality']
    if overall < 0.4:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Poor Overall Quality',
            'description': f'Overall explanation quality is low ({overall:.2f}). Major improvements needed.',
            'suggestions': [
                'Consider fundamentally different explanation methods',
                'Implement gradient-based attribution techniques',
                'Add explanation-aware training objectives'
            ]
        })
    elif overall < 0.7:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': 'Moderate Quality',
            'description': f'Explanation quality is moderate ({overall:.2f}). Room for improvement.',
            'suggestions': [
                'Focus on the lowest-scoring components',
                'Fine-tune explanation parameters',
                'Consider ensemble explanation methods'
            ]
        })
    
    # Component-specific recommendations
    if metrics['selectivity'] < 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Poor Selectivity',
            'description': 'Explanations highlight too many atoms or are not focused enough.',
            'suggestions': [
                'Add sparsity constraints to explanation method',
                'Use attention mechanisms or similar focusing techniques',
                'Post-process explanations to highlight only top contributors'
            ]
        })
    
    if metrics['chemical_plausibility'] < 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Low Chemical Plausibility',
            'description': 'Explanations don\'t align with chemical knowledge.',
            'suggestions': [
                'Incorporate domain knowledge into explanation generation',
                'Use chemistry-aware explanation methods',
                'Validate explanations against known SAR patterns'
            ]
        })
    
    if metrics['consistency'] < 0.5:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': 'Low Consistency',
            'description': 'Similar molecules have inconsistent explanations.',
            'suggestions': [
                'Add consistency regularization to training',
                'Use ensemble methods to improve stability',
                'Implement molecular alignment for fairer comparison'
            ]
        })
    
    if metrics['prediction_agreement'] < 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Poor Prediction Agreement',
            'description': 'Explanations don\'t align with model predictions.',
            'suggestions': [
                'Check explanation method implementation',
                'Ensure explanations are generated correctly',
                'Consider different attribution techniques'
            ]
        })
    
    return recommendations

def create_example_explanation_data():
    """Create example explanation data for demonstration."""
    
    # Example data with different quality levels
    examples = [
        {
            'smiles': 'CCc1ccccc1',  # Ethylbenzene
            'prediction': 0.75,
            'atom_contributions': [0.1, 0.15, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1],  # Focused on aromatic ring
            'true_label': 1,
            'description': 'Good example: aromatic atoms have high contributions'
        },
        {
            'smiles': 'CCCCCO',  # Pentanol
            'prediction': 0.25,
            'atom_contributions': [-0.1, -0.05, -0.08, -0.12, -0.15, 0.2],  # Hydroxyl positive, alkyl negative
            'true_label': 0,
            'description': 'Good example: polar atom positive, non-polar negative for inactive compound'
        },
        {
            'smiles': 'c1ccccc1N',  # Aniline
            'prediction': 0.8,
            'atom_contributions': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],  # All equal - poor selectivity
            'true_label': 1,
            'description': 'Poor example: no selectivity, all atoms equally important'
        },
        {
            'smiles': 'CCCCc1ccccc1',  # Butylbenzene
            'prediction': 0.85,
            'atom_contributions': [-0.3, -0.25, -0.2, -0.15, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1],  # Mixed pattern
            'true_label': 1,
            'description': 'Mixed example: aromatic positive, alkyl negative despite positive prediction'
        }
    ]
    
    return examples

def run_example_evaluation():
    """Run an example evaluation to demonstrate the metrics."""
    
    print("🔬 Explanation Quality Metrics - Example Evaluation")
    print("=" * 60)
    
    # Get example data
    example_data = create_example_explanation_data()
    
    print(f"📝 Evaluating {len(example_data)} example explanations:")
    for i, data in enumerate(example_data, 1):
        print(f"  {i}. {data['smiles']} - {data['description']}")
    
    # Calculate metrics
    metrics = calculate_basic_explanation_metrics(example_data)
    
    print(f"\n📊 Quality Metrics Results:")
    print(f"{'='*40}")
    
    for metric_name, score in metrics.items():
        if metric_name != 'overall_quality':
            status = "✅ Good" if score > 0.7 else "⚠️ Moderate" if score > 0.4 else "❌ Poor"
            print(f"{metric_name.replace('_', ' ').title():.<25} {score:.3f} {status}")
    
    print(f"{'='*40}")
    overall = metrics['overall_quality']
    overall_status = "✅ Good" if overall > 0.7 else "⚠️ Moderate" if overall > 0.4 else "❌ Poor"
    print(f"{'Overall Quality':.<25} {overall:.3f} {overall_status}")
    
    # Generate recommendations
    recommendations = generate_improvement_recommendations(metrics)
    
    if recommendations:
        print(f"\n💡 Improvement Recommendations:")
        print(f"{'='*40}")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['issue']} (Priority: {rec['priority']})")
            print(f"   {rec['description']}")
            print(f"   Suggestions:")
            for suggestion in rec['suggestions']:
                print(f"   • {suggestion}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    results = {
        'metrics': metrics,
        'recommendations': recommendations,
        'example_data': [
            {k: v for k, v in data.items() if k != 'description'}  # Remove description for cleaner JSON
            for data in example_data
        ]
    }
    
    with open("results/example_quality_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to results/example_quality_evaluation.json")
    
    return metrics, recommendations

def main():
    """Main function to run the example."""
    try:
        metrics, recommendations = run_example_evaluation()
        
        print(f"\n🎉 Example evaluation completed!")
        print(f"Overall quality score: {metrics['overall_quality']:.3f}")
        print(f"Number of recommendations: {len(recommendations)}")
        
        print(f"\n📚 Next Steps:")
        print(f"1. Replace example data with your actual model explanations")
        print(f"2. Implement the suggested improvements")
        print(f"3. Re-evaluate to track progress")
        print(f"4. Use the optimize_for_explainability.py script for systematic optimization")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
