"""
Explanation Quality Metrics for SAR Analysis
============================================

This module provides comprehensive metrics to evaluate the quality of molecular explanations
and determine how well they represent underlying structure-activity relationships (SAR).
These metrics can be used to optimize models for better explainability.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, rdMolChemicalFeatures
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

class ExplanationQualityEvaluator:
    """
    Comprehensive evaluator for molecular explanation quality.
    
    This class implements multiple metrics to assess how well explanations
    capture true structure-activity relationships.
    """
    
    def __init__(self):
        self.pharmacophore_factory = rdMolChemicalFeatures.BuildFeatureFactory(
            'BaseFeatures.fdef'
        )
        self.known_sar_patterns = self._load_known_sar_patterns()
        
    def _load_known_sar_patterns(self):
        """Load known SAR patterns for validation."""
        # Common pharmacophore patterns and their expected contributions
        return {
            'aromatic_rings': {
                'pattern': '[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1',
                'expected_contribution': 'positive',
                'description': 'Aromatic rings often contribute to binding'
            },
            'hydroxyl_groups': {
                'pattern': '[OH]',
                'expected_contribution': 'context_dependent',
                'description': 'Hydroxyl groups can form hydrogen bonds'
            },
            'nitrogen_heterocycles': {
                'pattern': '[#7]1[#6][#6][#6][#6][#6]1',
                'expected_contribution': 'positive',
                'description': 'Nitrogen heterocycles often important for activity'
            },
            'carboxyl_groups': {
                'pattern': 'C(=O)O',
                'expected_contribution': 'context_dependent',
                'description': 'Carboxyl groups can affect solubility and binding'
            },
            'halogens': {
                'pattern': '[F,Cl,Br,I]',
                'expected_contribution': 'context_dependent',
                'description': 'Halogens can affect lipophilicity and binding'
            }
        }
    
    def calculate_consistency_metrics(self, explanations_data):
        """
        Calculate consistency metrics across similar molecules.
        
        Args:
            explanations_data: List of dicts containing:
                - smiles: SMILES string
                - prediction: Model prediction
                - atom_contributions: List of atom contribution scores
                - molecule: RDKit molecule object
        
        Returns:
            dict: Consistency metrics
        """
        print("📊 Calculating consistency metrics...")
        
        # Group molecules by structural similarity
        similarity_groups = self._group_by_similarity(explanations_data)
        
        consistency_scores = []
        for group in similarity_groups:
            if len(group) < 2:
                continue
                
            # Calculate pairwise consistency within group
            group_consistency = self._calculate_group_consistency(group)
            consistency_scores.append(group_consistency)
        
        return {
            'mean_consistency': np.mean(consistency_scores) if consistency_scores else 0,
            'std_consistency': np.std(consistency_scores) if consistency_scores else 0,
            'num_groups': len(similarity_groups),
            'consistency_distribution': consistency_scores
        }
    
    def calculate_sar_alignment_metrics(self, explanations_data):
        """
        Calculate how well explanations align with known SAR principles.
        
        Args:
            explanations_data: List of explanation dictionaries
            
        Returns:
            dict: SAR alignment metrics
        """
        print("🧬 Calculating SAR alignment metrics...")
        
        alignment_scores = []
        pattern_matches = defaultdict(list)
        
        for data in explanations_data:
            mol = data['molecule']
            atom_contribs = data['atom_contributions']
            
            # Check each known SAR pattern
            for pattern_name, pattern_info in self.known_sar_patterns.items():
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern_info['pattern']))
                
                if matches:
                    # Calculate average contribution for matched atoms
                    pattern_contrib = np.mean([
                        atom_contribs[atom_idx] for match in matches for atom_idx in match
                    ])
                    
                    pattern_matches[pattern_name].append({
                        'contribution': pattern_contrib,
                        'expected': pattern_info['expected_contribution'],
                        'smiles': data['smiles']
                    })
        
        # Calculate alignment scores
        sar_scores = {}
        for pattern_name, matches in pattern_matches.items():
            if len(matches) > 1:
                contribs = [m['contribution'] for m in matches]
                expected = self.known_sar_patterns[pattern_name]['expected_contribution']
                
                if expected == 'positive':
                    # Expect positive contributions
                    alignment = np.mean([c > 0 for c in contribs])
                elif expected == 'negative':
                    # Expect negative contributions
                    alignment = np.mean([c < 0 for c in contribs])
                else:  # context_dependent
                    # Expect varied but consistent contributions
                    alignment = 1.0 - (np.std(contribs) / (np.abs(np.mean(contribs)) + 1e-6))
                
                sar_scores[pattern_name] = {
                    'alignment_score': alignment,
                    'num_matches': len(matches),
                    'mean_contribution': np.mean(contribs)
                }
        
        return {
            'overall_sar_alignment': np.mean(list(s['alignment_score'] for s in sar_scores.values())) if sar_scores else 0,
            'pattern_scores': sar_scores,
            'num_patterns_found': len(sar_scores)
        }
    
    def calculate_chemical_intuition_metrics(self, explanations_data):
        """
        Calculate metrics based on chemical intuition and domain knowledge.
        
        Args:
            explanations_data: List of explanation dictionaries
            
        Returns:
            dict: Chemical intuition metrics
        """
        print("⚗️ Calculating chemical intuition metrics...")
        
        intuition_scores = []
        
        for data in explanations_data:
            mol = data['molecule']
            atom_contribs = data['atom_contributions']
            prediction = data['prediction']
            
            # Calculate various chemical properties
            mol_properties = self._calculate_molecular_properties(mol)
            
            # Score based on chemical intuition
            intuition_score = self._score_chemical_intuition(
                mol, atom_contribs, prediction, mol_properties
            )
            intuition_scores.append(intuition_score)
        
        return {
            'mean_intuition_score': np.mean(intuition_scores),
            'std_intuition_score': np.std(intuition_scores),
            'intuition_distribution': intuition_scores
        }
    
    def calculate_stability_metrics(self, model, molecules, num_perturbations=10):
        """
        Calculate explanation stability under small perturbations.
        
        Args:
            model: The trained model
            molecules: List of molecules to test
            num_perturbations: Number of perturbations per molecule
            
        Returns:
            dict: Stability metrics
        """
        print("🔄 Calculating stability metrics...")
        
        stability_scores = []
        
        for mol_data in molecules[:5]:  # Test on subset for efficiency
            mol = mol_data['molecule']
            original_contribs = mol_data['atom_contributions']
            
            # Generate perturbations (small changes in molecular representation)
            perturbation_contribs = []
            
            for _ in range(num_perturbations):
                # Add small noise to features or use dropout-like perturbations
                perturbed_contribs = self._get_perturbed_explanation(
                    model, mol, perturbation_strength=0.1
                )
                perturbation_contribs.append(perturbed_contribs)
            
            # Calculate stability as correlation with original
            correlations = [
                pearsonr(original_contribs, pert_contrib)[0] 
                for pert_contrib in perturbation_contribs
                if len(pert_contrib) == len(original_contribs)
            ]
            
            if correlations:
                stability_scores.append(np.mean(correlations))
        
        return {
            'mean_stability': np.mean(stability_scores) if stability_scores else 0,
            'std_stability': np.std(stability_scores) if stability_scores else 0,
            'stability_distribution': stability_scores
        }
    
    def calculate_selectivity_metrics(self, explanations_data):
        """
        Calculate how selective explanations are (not highlighting everything).
        
        Args:
            explanations_data: List of explanation dictionaries
            
        Returns:
            dict: Selectivity metrics
        """
        print("🎯 Calculating selectivity metrics...")
        
        selectivity_scores = []
        
        for data in explanations_data:
            atom_contribs = np.array(data['atom_contributions'])
            
            # Calculate sparsity (what fraction of atoms have near-zero contribution)
            threshold = 0.1 * np.std(atom_contribs)
            sparse_fraction = np.mean(np.abs(atom_contribs) < threshold)
            
            # Calculate concentration (are contributions focused on few atoms)
            contrib_entropy = -np.sum(
                np.abs(atom_contribs) / np.sum(np.abs(atom_contribs)) * 
                np.log(np.abs(atom_contribs) / np.sum(np.abs(atom_contribs)) + 1e-10)
            )
            max_entropy = np.log(len(atom_contribs))
            concentration = 1 - (contrib_entropy / max_entropy)
            
            selectivity_score = (sparse_fraction + concentration) / 2
            selectivity_scores.append(selectivity_score)
        
        return {
            'mean_selectivity': np.mean(selectivity_scores),
            'std_selectivity': np.std(selectivity_scores),
            'selectivity_distribution': selectivity_scores
        }
    
    def calculate_comprehensive_quality_score(self, explanations_data, model=None):
        """
        Calculate a comprehensive quality score combining all metrics.
        
        Args:
            explanations_data: List of explanation dictionaries
            model: Optional model for stability testing
            
        Returns:
            dict: Comprehensive quality assessment
        """
        print("🏆 Calculating comprehensive explanation quality score...")
        
        # Calculate all individual metrics
        consistency = self.calculate_consistency_metrics(explanations_data)
        sar_alignment = self.calculate_sar_alignment_metrics(explanations_data)
        chemical_intuition = self.calculate_chemical_intuition_metrics(explanations_data)
        selectivity = self.calculate_selectivity_metrics(explanations_data)
        
        # Calculate stability if model is provided
        stability = {}
        if model is not None:
            stability = self.calculate_stability_metrics(model, explanations_data)
        
        # Combine scores with weights
        weights = {
            'consistency': 0.25,
            'sar_alignment': 0.30,
            'chemical_intuition': 0.25,
            'selectivity': 0.20
        }
        
        if stability:
            weights['stability'] = 0.15
            # Renormalize
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        component_scores = {
            'consistency': consistency['mean_consistency'],
            'sar_alignment': sar_alignment['overall_sar_alignment'],
            'chemical_intuition': chemical_intuition['mean_intuition_score'],
            'selectivity': selectivity['mean_selectivity']
        }
        
        if stability:
            component_scores['stability'] = stability['mean_stability']
        
        overall_score = sum(
            weights[component] * score 
            for component, score in component_scores.items()
        )
        
        return {
            'overall_quality_score': overall_score,
            'component_scores': component_scores,
            'weights': weights,
            'detailed_metrics': {
                'consistency': consistency,
                'sar_alignment': sar_alignment,
                'chemical_intuition': chemical_intuition,
                'selectivity': selectivity,
                'stability': stability
            }
        }
    
    def _group_by_similarity(self, explanations_data, similarity_threshold=0.7):
        """Group molecules by structural similarity."""
        from rdkit import DataStructs
        from rdkit.Chem import rdMolDescriptors
        
        # Calculate fingerprints
        fps = []
        for data in explanations_data:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(data['molecule'], 2)
            fps.append((fp, data))
        
        # Group by similarity
        groups = []
        used = set()
        
        for i, (fp1, data1) in enumerate(fps):
            if i in used:
                continue
                
            group = [data1]
            used.add(i)
            
            for j, (fp2, data2) in enumerate(fps[i+1:], i+1):
                if j in used:
                    continue
                    
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                if similarity >= similarity_threshold:
                    group.append(data2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_group_consistency(self, group):
        """Calculate consistency within a group of similar molecules."""
        if len(group) < 2:
            return 0.0
        
        correlations = []
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                contrib1 = group[i]['atom_contributions']
                contrib2 = group[j]['atom_contributions']
                
                # Align contributions by substructure matching
                aligned_contrib1, aligned_contrib2 = self._align_contributions(
                    group[i]['molecule'], group[j]['molecule'], contrib1, contrib2
                )
                
                if len(aligned_contrib1) > 1 and len(aligned_contrib2) > 1:
                    corr, _ = pearsonr(aligned_contrib1, aligned_contrib2)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _align_contributions(self, mol1, mol2, contrib1, contrib2):
        """Align contributions between similar molecules."""
        # Simple alignment - could be improved with more sophisticated matching
        min_len = min(len(contrib1), len(contrib2))
        return contrib1[:min_len], contrib2[:min_len]
    
    def _calculate_molecular_properties(self, mol):
        """Calculate various molecular properties."""
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'hbd': rdMolDescriptors.CalcNumHBD(mol),
            'hba': rdMolDescriptors.CalcNumHBA(mol),
            'tpsa': rdMolDescriptors.CalcTPSA(mol),
            'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol)
        }
    
    def _score_chemical_intuition(self, mol, atom_contribs, prediction, properties):
        """Score based on chemical intuition."""
        score = 0.0
        
        # Check if aromatic atoms have reasonable contributions
        aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
        if aromatic_atoms:
            aromatic_contribs = [atom_contribs[i] for i in aromatic_atoms]
            # Aromatic atoms should generally have positive contribution for active compounds
            if prediction > 0.5:  # Active compound
                score += 0.3 * np.mean([c > 0 for c in aromatic_contribs])
        
        # Check heteroatom contributions
        hetero_atoms = [atom.GetIdx() for atom in mol.GetAtoms() 
                       if atom.GetAtomicNum() not in [1, 6]]  # Not H or C
        if hetero_atoms:
            hetero_contribs = [atom_contribs[i] for i in hetero_atoms]
            # Heteroatoms should have significant (non-zero) contributions
            score += 0.3 * np.mean([abs(c) > 0.1 for c in hetero_contribs])
        
        # Check if highly lipophilic regions have appropriate contributions
        if properties['logp'] > 3:  # Lipophilic compound
            # Should have some negative contributions to balance lipophilicity
            score += 0.2 * np.mean([c < 0 for c in atom_contribs])
        
        # Check if polar regions (near heteroatoms) have appropriate contributions
        polar_score = self._evaluate_polar_contributions(mol, atom_contribs)
        score += 0.2 * polar_score
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _evaluate_polar_contributions(self, mol, atom_contribs):
        """Evaluate contributions of polar regions."""
        # Simple heuristic: atoms bonded to heteroatoms should have significant contributions
        polar_atoms = []
        for bond in mol.GetBonds():
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if atom1.GetAtomicNum() in [7, 8, 16] or atom2.GetAtomicNum() in [7, 8, 16]:  # N, O, S
                polar_atoms.extend([atom1.GetIdx(), atom2.GetIdx()])
        
        if polar_atoms:
            polar_contribs = [abs(atom_contribs[i]) for i in set(polar_atoms)]
            return np.mean([c > 0.05 for c in polar_contribs])
        
        return 0.5  # Neutral score if no polar atoms found
    
    def _get_perturbed_explanation(self, model, mol, perturbation_strength=0.1):
        """Generate explanation with small perturbations (placeholder)."""
        # This would need to be implemented based on the specific model type
        # For now, return a perturbed version of a typical explanation
        num_atoms = mol.GetNumAtoms()
        base_contribs = np.random.normal(0, 0.2, num_atoms)
        noise = np.random.normal(0, perturbation_strength, num_atoms)
        return base_contribs + noise
    
    def visualize_quality_metrics(self, quality_results, save_path=None):
        """Visualize explanation quality metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Explanation Quality Assessment', fontsize=16)
        
        # Component scores
        ax = axes[0, 0]
        components = list(quality_results['component_scores'].keys())
        scores = list(quality_results['component_scores'].values())
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(components)]
        
        bars = ax.bar(components, scores, color=colors)
        ax.set_ylabel('Score')
        ax.set_title('Component Scores')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Overall quality score
        ax = axes[0, 1]
        overall_score = quality_results['overall_quality_score']
        ax.pie([overall_score, 1-overall_score], 
               labels=['Quality Score', 'Room for Improvement'],
               colors=['lightgreen', 'lightcoral'],
               startangle=90)
        ax.set_title(f'Overall Quality: {overall_score:.3f}')
        
        # Consistency distribution
        if 'consistency' in quality_results['detailed_metrics']:
            consistency_dist = quality_results['detailed_metrics']['consistency'].get('consistency_distribution', [])
            if consistency_dist:
                ax = axes[0, 2]
                ax.hist(consistency_dist, bins=10, alpha=0.7, color='skyblue')
                ax.set_xlabel('Consistency Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Consistency Distribution')
        
        # SAR alignment by pattern
        sar_data = quality_results['detailed_metrics'].get('sar_alignment', {})
        pattern_scores = sar_data.get('pattern_scores', {})
        if pattern_scores:
            ax = axes[1, 0]
            patterns = list(pattern_scores.keys())
            alignment_scores = [pattern_scores[p]['alignment_score'] for p in patterns]
            
            ax.barh(patterns, alignment_scores, color='lightcoral')
            ax.set_xlabel('SAR Alignment Score')
            ax.set_title('SAR Pattern Alignment')
            ax.set_xlim(0, 1)
        
        # Chemical intuition distribution
        intuition_data = quality_results['detailed_metrics'].get('chemical_intuition', {})
        intuition_dist = intuition_data.get('intuition_distribution', [])
        if intuition_dist:
            ax = axes[1, 1]
            ax.hist(intuition_dist, bins=10, alpha=0.7, color='lightgreen')
            ax.set_xlabel('Chemical Intuition Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Chemical Intuition Distribution')
        
        # Selectivity distribution
        selectivity_data = quality_results['detailed_metrics'].get('selectivity', {})
        selectivity_dist = selectivity_data.get('selectivity_distribution', [])
        if selectivity_dist:
            ax = axes[1, 2]
            ax.hist(selectivity_dist, bins=10, alpha=0.7, color='gold')
            ax.set_xlabel('Selectivity Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Selectivity Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Quality metrics visualization saved to {save_path}")
        
        plt.show()
        return fig

def save_quality_report(quality_results, filepath):
    """Save comprehensive quality report to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    serializable_results = convert_numpy(quality_results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"📄 Quality report saved to {filepath}")

if __name__ == "__main__":
    print("🔬 Explanation Quality Metrics Module")
    print("This module provides comprehensive metrics for evaluating molecular explanation quality.")
    print("Import and use ExplanationQualityEvaluator to assess your model explanations.")
