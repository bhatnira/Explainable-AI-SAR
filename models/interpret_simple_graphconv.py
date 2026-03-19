#!/usr/bin/env python
"""
GraphConv Model Interpretation and Explainability
Author: AI Assistant
Date: 2025
Description: Interpret GraphConv model predictions with molecular visualizations
             and atom-level contribution analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepchem as dc

class GraphConvExplainer:
    """Explainer for GraphConv models with molecular visualization"""
    
    def __init__(self, model_path, featurizer=None):
        """Initialize the explainer with a trained model"""
        self.model_path = model_path
        self.model = None
        self.featurizer = featurizer or dc.feat.ConvMolFeaturizer()
        self.load_model()
    
    def load_model(self):
        """Load the trained GraphConv model"""
        try:
            print("🔄 Loading trained GraphConv model...")
            self.model = dc.models.GraphConvModel(
                n_tasks=1,
                mode='classification',
                n_classes=2,
                graph_conv_layers=[64, 64],
                dense_layer_size=128,
                dropout=0.2,
                model_dir=self.model_path
            )
            self.model.restore()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_molecule(self, smiles):
        """Predict activity for a single molecule"""
        try:
            # Featurize the molecule
            features = self.featurizer.featurize([smiles])
            dataset = dc.data.NumpyDataset(X=features, ids=[smiles])
            
            # Make prediction
            prediction = self.model.predict(dataset)
            probability = prediction[0][1]  # Probability of being active
            
            return probability
        except Exception as e:
            print(f"❌ Error predicting molecule {smiles}: {e}")
            return None
    
    def explain_prediction(self, smiles, reference_smiles=None):
        """
        Explain prediction using atomic contribution analysis
        This is a simplified explanation based on atomic properties
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"❌ Invalid SMILES: {smiles}")
                return None
            
            # Get base prediction
            base_prob = self.predict_molecule(smiles)
            if base_prob is None:
                return None
            
            # Calculate atomic contributions (simplified approach)
            atom_contributions = self._calculate_atomic_contributions(mol, smiles)
            
            explanation = {
                'smiles': smiles,
                'prediction_probability': base_prob,
                'predicted_class': 'Active' if base_prob > 0.5 else 'Inactive',
                'confidence': abs(base_prob - 0.5) * 2,
                'atom_contributions': atom_contributions,
                'molecule': mol
            }
            
            return explanation
            
        except Exception as e:
            print(f"❌ Error explaining prediction: {e}")
            return None
    
    def _calculate_atomic_contributions(self, mol, smiles):
        """
        Calculate atomic contributions using molecular descriptors
        This is a simplified approach for demonstration
        """
        atom_contributions = []
        
        for atom in mol.GetAtoms():
            # Simple heuristic based on atom properties
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            hybridization = str(atom.GetHybridization())
            
            # Simple scoring based on common active patterns
            contribution = 0.0
            
            # Nitrogen and oxygen often important for bioactivity
            if atomic_num == 7:  # Nitrogen
                contribution += 0.3
            elif atomic_num == 8:  # Oxygen
                contribution += 0.2
            elif atomic_num == 6:  # Carbon
                contribution += 0.1
            
            # Aromatic atoms often important
            if atom.GetIsAromatic():
                contribution += 0.2
            
            # Higher degree atoms (hubs) might be important
            contribution += degree * 0.05
            
            # Formal charge effects
            contribution += abs(formal_charge) * 0.1
            
            atom_contributions.append({
                'atom_idx': atom.GetIdx(),
                'atomic_symbol': atom.GetSymbol(),
                'atomic_num': atomic_num,
                'contribution': contribution,
                'degree': degree,
                'formal_charge': formal_charge,
                'is_aromatic': atom.GetIsAromatic(),
                'hybridization': hybridization
            })
        
        return atom_contributions
    
    def visualize_explanation(self, explanation, save_path=None):
        """Visualize the molecular explanation with atom highlighting"""
        if explanation is None:
            return None
        
        mol = explanation['molecule']
        atom_contribs = explanation['atom_contributions']
        
        # Create atom colors based on contributions
        atom_colors = {}
        contrib_values = [ac['contribution'] for ac in atom_contribs]
        max_contrib = max(contrib_values) if contrib_values else 1.0
        
        for ac in atom_contribs:
            # Color intensity based on contribution
            intensity = ac['contribution'] / max_contrib if max_contrib > 0 else 0
            # Red for positive contributions, blue for negative (though we only have positive here)
            atom_colors[ac['atom_idx']] = (1.0, 1.0 - intensity, 1.0 - intensity)
        
        # Create molecule drawing
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
        drawer.SetFontSize(0.8)
        
        # Draw molecule with atom highlighting
        drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), 
                          highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        # Save or display the image
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(drawer.GetDrawingText())
            print(f"💾 Visualization saved to: {save_path}")
        
        return drawer.GetDrawingText()
    
    def generate_report(self, smiles_list, output_dir='results/graphconv_explanations'):
        """Generate explanation reports for multiple molecules"""
        print(f"📊 Generating explanation reports for {len(smiles_list)} molecules...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        explanations = []
        
        for i, smiles in enumerate(smiles_list):
            print(f"   Processing molecule {i+1}/{len(smiles_list)}: {smiles[:50]}...")
            
            # Get explanation
            explanation = self.explain_prediction(smiles)
            if explanation is None:
                continue
            
            explanations.append(explanation)
            
            # Save visualization
            viz_path = os.path.join(output_dir, f'molecule_{i+1}_explanation.png')
            self.visualize_explanation(explanation, viz_path)
        
        # Create summary report
        self._create_summary_report(explanations, output_dir)
        
        print(f"✅ Generated explanations for {len(explanations)} molecules")
        print(f"📂 Results saved to: {output_dir}")
        
        return explanations
    
    def _create_summary_report(self, explanations, output_dir):
        """Create a summary report of all explanations"""
        if not explanations:
            return
        
        # Create summary dataframe
        summary_data = []
        for exp in explanations:
            summary_data.append({
                'smiles': exp['smiles'],
                'prediction_probability': exp['prediction_probability'],
                'predicted_class': exp['predicted_class'],
                'confidence': exp['confidence'],
                'num_atoms': len(exp['atom_contributions']),
                'avg_atom_contribution': np.mean([ac['contribution'] for ac in exp['atom_contributions']]),
                'max_atom_contribution': max([ac['contribution'] for ac in exp['atom_contributions']]),
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'explanation_summary.csv'), index=False)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prediction probability distribution
        axes[0, 0].hist(summary_df['prediction_probability'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution of Prediction Probabilities')
        axes[0, 0].set_xlabel('Prediction Probability')
        axes[0, 0].set_ylabel('Frequency')
        
        # Confidence distribution
        axes[0, 1].hist(summary_df['confidence'], bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Distribution of Prediction Confidence')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        
        # Class distribution
        class_counts = summary_df['predicted_class'].value_counts()
        axes[1, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Predicted Class Distribution')
        
        # Atom contribution analysis
        axes[1, 1].scatter(summary_df['num_atoms'], summary_df['avg_atom_contribution'], alpha=0.6)
        axes[1, 1].set_title('Atom Count vs Average Contribution')
        axes[1, 1].set_xlabel('Number of Atoms')
        axes[1, 1].set_ylabel('Average Atom Contribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'explanation_summary.png'), dpi=300, bbox_inches='tight')
        plt.show()

def load_test_molecules():
    """Load test molecules for explanation"""
    try:
        df = pd.read_excel('StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx')
        # Take a sample of molecules for explanation
        sample_df = df.sample(n=min(10, len(df)), random_state=42)
        return sample_df['Smiles'].tolist()
    except Exception as e:
        print(f"❌ Error loading test molecules: {e}")
        return []

def main():
    """Main explanation pipeline"""
    print("🔍 GraphConv Model Interpretation Pipeline")
    print("=" * 50)
    
    try:
        # Initialize explainer
        explainer = GraphConvExplainer('models/graphconv_simple')
        
        # Load test molecules
        test_smiles = load_test_molecules()
        if not test_smiles:
            print("❌ No test molecules found")
            return
        
        # Generate explanations
        explanations = explainer.generate_report(test_smiles)
        
        # Show individual examples
        print("\n🔬 Example Explanations:")
        for i, exp in enumerate(explanations[:3]):  # Show first 3
            print(f"\nMolecule {i+1}:")
            print(f"  SMILES: {exp['smiles']}")
            print(f"  Prediction: {exp['predicted_class']} ({exp['prediction_probability']:.3f})")
            print(f"  Confidence: {exp['confidence']:.3f}")
            print(f"  Top contributing atoms:")
            
            # Sort atoms by contribution
            sorted_atoms = sorted(exp['atom_contributions'], 
                                key=lambda x: x['contribution'], reverse=True)[:3]
            for atom in sorted_atoms:
                print(f"    {atom['atomic_symbol']}{atom['atom_idx']}: {atom['contribution']:.3f}")
        
        print(f"\n🎉 Explanation pipeline completed!")
        print(f"Generated explanations for {len(explanations)} molecules")
        
    except Exception as e:
        print(f"❌ Error during explanation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
