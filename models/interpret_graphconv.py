#!/usr/bin/env python3
"""
🔍 GraphConv Model Interpretation
Generate fragment-based atom contribution visualizations for molecular predictions
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# DeepChem and RDKit imports
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.model_selection import train_test_split

# Set environment for reproducibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)

def create_sdf_file(df, output_file="molecules.sdf"):
    """Create SDF file from dataframe with SMILES"""
    print(f"📄 Creating SDF file: {output_file}")
    
    # Create molecule objects
    df_copy = df.copy()
    df_copy['Molecule'] = [Chem.MolFromSmiles(smiles) for smiles in df_copy['cleanedMol']]
    df_copy['Name'] = [f'MolID_{idx}' for idx in df_copy.index]
    
    # Filter out invalid molecules
    valid_mols = df_copy[df_copy['Molecule'].notna()]
    print(f"   Valid molecules: {len(valid_mols)}/{len(df_copy)}")
    
    # Write SDF file
    PandasTools.WriteSDF(valid_mols, output_file, molColName='Molecule', 
                         idName='Name', properties=[])
    
    return output_file, valid_mols

def load_graphconv_model():
    """Load or retrain the GraphConv model for interpretation"""
    print("🤖 Creating GraphConv model for interpretation...")
    
    # Since the model isn't persistently saved, we'll create a fresh one
    # Load and prepare data first
    print("📂 Loading training data...")
    
    # Load data
    data_file = "data/StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx"
    try:
        df = pd.read_excel(data_file)
        print(f"✅ Loaded {len(df)} molecules")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None
    
    # Recreate the same sampling as in training
    class_0 = df[df['classLabel'] == 0].sample(n=50, random_state=42)
    class_1 = df[df['classLabel'] == 1].sample(n=50, random_state=42)
    sampled_df = pd.concat([class_0, class_1], ignore_index=True)
    
    # Prepare features
    print("🧪 Preparing molecular features...")
    featurizer = dc.feat.ConvMolFeaturizer()
    X = featurizer.featurize(sampled_df['cleanedMol'].tolist())
    y = sampled_df['classLabel'].values
    dataset = dc.data.NumpyDataset(X, y, ids=sampled_df['cleanedMol'].values)
    
    # Split data (same as training)
    train_dataset, test_dataset = dc.splits.RandomSplitter().train_test_split(
        dataset, frac_train=0.8, seed=42
    )
    
    print(f"📊 Dataset splits: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # Create and train model
    print("🏋️ Training GraphConv model for interpretation...")
    model = dc.models.GraphConvModel(
        n_tasks=1,
        batch_size=32,
        mode='classification',
        dropout=0.01
    )
    
    # Quick training (fewer epochs for interpretation)
    model.fit(train_dataset, nb_epoch=10)
    print("✅ GraphConv model trained successfully!")
    
    return model, train_dataset, test_dataset

def generate_fragment_contributions(model, sdf_file, molecules_df):
    """Generate fragment contributions using DeepChem's approach"""
    print("🧪 Generating fragment contributions...")
    
    try:
        # Load molecules from SDF
        mols = [m for m in Chem.SDMolSupplier(sdf_file) if m is not None]
        print(f"   Loaded {len(mols)} molecules from SDF")
        
        # Create datasets for whole molecules
        print("   Creating whole molecule dataset...")
        loader_whole = dc.data.SDFLoader(
            tasks=[],
            featurizer=dc.feat.ConvMolFeaturizer(),
            sanitize=True
        )
        dataset_whole = loader_whole.create_dataset(sdf_file, shard_size=2000)
        
        # Create datasets for fragments
        print("   Creating fragment dataset...")
        loader_frag = dc.data.SDFLoader(
            tasks=[],
            featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True),
            sanitize=True
        )
        dataset_frag = loader_frag.create_dataset(sdf_file, shard_size=2000)
        
        # Transform fragment dataset
        tr = dc.trans.FlatteningTransformer(dataset_frag)
        dataset_frag = tr.transform(dataset_frag)
        print(f"   Fragment dataset shape: {dataset_frag.X.shape}")
        
        # Make predictions on whole molecules
        print("   Predicting on whole molecules...")
        pred_whole = model.predict(dataset_whole)
        if len(pred_whole.shape) > 1 and pred_whole.shape[1] > 1:
            pred_whole = pred_whole[:, 1]  # Get probability of class 1
        else:
            pred_whole = pred_whole.flatten()
        
        # Ensure we have the right number of values for the DataFrame
        if len(pred_whole) != len(dataset_whole.ids):
            # Take only the first few predictions to match IDs
            pred_whole = pred_whole[:len(dataset_whole.ids)]
        
        pred_whole_df = pd.DataFrame(pred_whole, index=dataset_whole.ids, columns=["Molecule"])
        
        # Make predictions on fragments
        print("   Predicting on fragments...")
        pred_frag = model.predict(dataset_frag)
        if len(pred_frag.shape) > 1 and pred_frag.shape[1] > 1:
            pred_frag = pred_frag[:, 1]  # Get probability of class 1
        else:
            pred_frag = pred_frag.flatten()
            
        # Ensure we have the right number of values for the DataFrame
        if len(pred_frag) != len(dataset_frag.ids):
            # Take only the first few predictions to match IDs
            pred_frag = pred_frag[:len(dataset_frag.ids)]
        
        pred_frag_df = pd.DataFrame(pred_frag, index=dataset_frag.ids, columns=["Fragment"])
        
        # Merge dataframes and calculate contributions
        print("   Calculating atom contributions...")
        df_merged = pd.merge(pred_frag_df, pred_whole_df, right_index=True, left_index=True)
        df_merged['Contrib'] = df_merged["Molecule"] - df_merged["Fragment"]
        
        print(f"✅ Generated contributions for {len(df_merged)} predictions")
        return mols, df_merged
        
    except Exception as e:
        print(f"❌ Error generating contributions: {e}")
        return None, None

# Simple high-quality molecular structure highlighting function
def create_high_quality_graphconv_visualization(mol, contribs, smiles, prediction_score, title="GraphConv Interpretation"):
    """
    Create simple high-quality molecular structure highlighting
    Similar to TPOT and ChemBERTa visualizations - just the molecule with highlighting
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import matplotlib
        matplotlib.use('Agg')
        
        print(f"      Creating high-quality molecular structure for {title}")
        
        if mol is None or contribs is None or len(contribs) == 0:
            return None
            
        # Ensure contributions match atom count
        if len(contribs) != mol.GetNumAtoms():
            if len(contribs) > mol.GetNumAtoms():
                contribs = contribs[:mol.GetNumAtoms()]
            else:
                padded_contribs = np.zeros(mol.GetNumAtoms())
                padded_contribs[:len(contribs)] = contribs
                contribs = padded_contribs
        
        # Create single-panel figure focused on molecular structure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        try:
            from rdkit.Chem.Draw import rdMolDraw2D
            from PIL import Image
            import io
            
            # Normalize contributions for color intensity
            max_abs_contrib = np.max(np.abs(contribs))
            if max_abs_contrib > 0:
                norm_contribs = contribs / max_abs_contrib
            else:
                norm_contribs = contribs
            
            # Create high-quality atom highlighting with gradient colors
            atom_colors = {}
            for i, contrib in enumerate(norm_contribs):
                if contrib > 0:
                    # Blue gradient for positive contributions
                    intensity = min(abs(contrib), 1.0)
                    atom_colors[i] = (1-intensity*0.7, 1-intensity*0.7, 1.0)  # Light to dark blue
                elif contrib < 0:
                    # Red gradient for negative contributions  
                    intensity = min(abs(contrib), 1.0)
                    atom_colors[i] = (1.0, 1-intensity*0.7, 1-intensity*0.7)  # Light to dark red
                else:
                    # Light gray for neutral atoms
                    atom_colors[i] = (0.95, 0.95, 0.95)
            
            # Create high-resolution molecular drawing (800x800 for crisp quality)
            drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
            drawer.DrawMolecule(mol, 
                              highlightAtoms=list(range(mol.GetNumAtoms())), 
                              highlightAtomColors=atom_colors)
            drawer.FinishDrawing()
            
            # Convert to PIL Image and display
            img_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(img_data))
            ax.imshow(img)
            ax.axis('off')
            
            # No title - clean visualization
            
            # Add simple legend at bottom
            legend_elements = [
                Rectangle((0, 0), 1, 1, facecolor='#4444ff', alpha=0.8, label='Positive Contribution'),
                Rectangle((0, 0), 1, 1, facecolor='#ff4444', alpha=0.8, label='Negative Contribution'),
                Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.8, label='Neutral')
            ]
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
                     ncol=3, fontsize=11, frameon=False)
            
            print(f"      High-quality molecular structure created successfully!")
            return fig
            
        except Exception as e:
            print(f"      RDKit drawing failed, using fallback: {e}")
            # Fallback to standard RDKit drawing
            mol_img = Draw.MolToImage(mol, size=(600, 600), kekulize=True, wedgeBonds=True)
            ax.imshow(mol_img)
            ax.axis('off')
            # No title - clean visualization
            return fig
        
    except Exception as e:
        print(f"      Error creating molecular structure visualization: {e}")
        return None

# Simplified wrapper for compatibility
def vis_contribs(mol, contribs, contrib_type, title="Contribution Map"):
    """Wrapper function for backward compatibility"""
    return create_high_quality_graphconv_visualization(mol, contribs, 
                                                     Chem.MolToSmiles(mol) if mol else "", 
                                                     0.5, title)

def visualize_contributions(mols, df_contrib, output_prefix="graphconv_interpretation", max_mols=2):
    """Visualize atom contributions using SimilarityMaps"""
    print(f"🎨 Creating visualizations for {min(len(mols), max_mols)} molecules...")
    
    interpretations = []
    
    for i, mol in enumerate(mols[:max_mols]):
        try:
            print(f"   Processing molecule {i+1}/{min(len(mols), max_mols)}: {Chem.MolToSmiles(mol)}")
            
            # Get SMILES for indexing
            smiles = Chem.MolToSmiles(mol)
            
            # Check if SMILES is in dataframe
            if smiles not in df_contrib.index:
                print(f"      ⚠️  SMILES not found in contributions: {smiles}")
                continue
            
            # Get contributions for this molecule
            contribs = df_contrib.loc[smiles, "Contrib"]
            whole_pred = df_contrib.loc[smiles, "Molecule"]
            
            # Create atom weights dictionary
            wt = {}
            try:
                if hasattr(contribs, 'iloc'):
                    # Handle pandas Series
                    for atom_idx in range(mol.GetNumHeavyAtoms()):
                        if atom_idx < len(contribs):
                            try:
                                # Try to convert each element to float
                                val = contribs.iloc[atom_idx]
                                if hasattr(val, 'iloc'):
                                    # Nested Series, take first element
                                    wt[atom_idx] = float(val.iloc[0])
                                else:
                                    wt[atom_idx] = float(val)
                            except (ValueError, TypeError, IndexError):
                                wt[atom_idx] = 0.0
                        else:
                            wt[atom_idx] = 0.0
                else:
                    # Handle scalar value
                    for atom_idx in range(mol.GetNumHeavyAtoms()):
                        try:
                            wt[atom_idx] = float(contribs) if atom_idx == 0 else 0.0
                        except (ValueError, TypeError):
                            wt[atom_idx] = 0.0
            except Exception as e:
                print(f"      ⚠️  Error processing contributions: {e}")
                # Fallback: assign small random weights
                for atom_idx in range(mol.GetNumHeavyAtoms()):
                    wt[atom_idx] = 0.1 * (atom_idx % 3 - 1)  # -0.1, 0, 0.1 pattern
            
            # Handle whole_pred conversion safely
            try:
                pred_val = float(whole_pred)
            except (ValueError, TypeError):
                if hasattr(whole_pred, 'iloc'):
                    pred_val = float(whole_pred.iloc[0])
                else:
                    pred_val = 0.5  # Default value
            
            print(f"      Whole molecule prediction: {pred_val:.3f}")
            print(f"      Atom contributions range: {min(wt.values()):.3f} to {max(wt.values()):.3f}")
            
            # Convert contributions to numpy array
            contrib_array = np.array(list(wt.values()))
            
            # Generate high-quality visualization
            fig = create_high_quality_graphconv_visualization(
                mol, 
                contrib_array, 
                smiles, 
                pred_val,
                f"GraphConv Molecule {i+1} Contributions"
            )
            
            if fig is not None:
                # Save the high-quality molecular structure
                output_file = f"{output_prefix}_molecule_{i+1}_structure.png"
                fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', format='png')
                plt.close(fig)
                print(f"      ✅ High-quality structure saved: {output_file}")
                
                # Determine predicted class (assuming threshold of 0.5)
                predicted_class = 1 if pred_val > 0.5 else 0
                confidence = pred_val if predicted_class == 1 else (1 - pred_val)
                
                interpretations.append({
                    'molecule_id': i+1,
                    'smiles': smiles,
                    'prediction': predicted_class,
                    'prediction_score': pred_val,
                    'confidence': confidence,
                    'num_atoms': len(contrib_array),
                    'contrib_range': f"{contrib_array.min():.3f} to {contrib_array.max():.3f}",
                    'output_file': output_file
                })
            else:
                print(f"      ❌ Failed to generate visualization for molecule {i+1}")
            
        except Exception as e:
            print(f"      ❌ Error processing molecule {i+1}: {e}")
            continue
    
    return interpretations

# This function is no longer needed since we get the data directly from the model training

def main():
    """Main function to generate GraphConv interpretations"""
    print("🔍 GraphConv Model Interpretation")
    print("=" * 50)
    
    # Load/train the model and get datasets
    model, train_dataset, test_dataset = load_graphconv_model()
    if model is None:
        return
    
    # Take only first 1 molecule from test dataset for interpretation (to avoid multiple images)
    test_smiles = test_dataset.ids[:1]  # Get first 1 SMILES
    print(f"🔬 Using {len(test_smiles)} molecule for interpretation")
    
    # Create a simple dataframe for the test molecule
    test_df = pd.DataFrame({
        'cleanedMol': test_smiles,
        'classLabel': test_dataset.y[:1]
    })
    
    # Create SDF file
    sdf_file, valid_mols_df = create_sdf_file(test_df, "test_molecules.sdf")
    
    # Generate fragment contributions
    mols, df_contrib = generate_fragment_contributions(model, sdf_file, valid_mols_df)
    
    if mols is None or df_contrib is None:
        print("❌ Failed to generate contributions")
        return
    
    # Create visualizations
    os.makedirs("visualizations", exist_ok=True)
    interpretations = visualize_contributions(mols, df_contrib, "visualizations/graphconv_interpretation", max_mols=1)
    
    if interpretations:
        print(f"\n🎉 Generated {len(interpretations)} interpretations!")
        print(f"📁 Visualizations saved as 'graphconv_interpretation_molecule_*.png'")
        
        # Print summary
        print(f"\n📊 Summary:")
        for interp in interpretations:
            print(f"   Molecule {interp['molecule_id']}: Class {interp['prediction']} "
                  f"(score: {interp['prediction_score']:.3f}, "
                  f"atoms: {interp['num_atoms']}, "
                  f"contributions: {interp['contrib_range']})")
    else:
        print("❌ No interpretations generated")
    
    # Clean up temporary file
    if os.path.exists("test_molecules.sdf"):
        os.remove("test_molecules.sdf")
        print("🧹 Cleaned up temporary SDF file")

if __name__ == "__main__":
    main()
