#!/usr/bin/env python3
"""
Prepare QSAR dataset for GIF generation by computing fingerprints, PCA, and t-SNE.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

def main():
    print("Loading QSAR dataset...")
    qsar_df = pd.read_csv('Tzu-QSAR-only/data/processed/QSAR_potency_20um.csv')
    
    print(f"Processing {len(qsar_df)} compounds...")
    
    # Prepare dataset for GIF generation
    gif_ready_data = {
        'Molecule ChEMBL ID': qsar_df['Identifier'].values,
        'Smiles': qsar_df['Canonical SMILES'].values,
        'IC50': qsar_df['IC50 uM'].values,
        'classLabel': qsar_df['potency_label'].values,  # 1=potent, 0=not potent
        'IsValidSMILES': [],
        'Morgan_FP': [],
        'Molecule': [],
        'Fingerprint': [],
        'MolecularWeight': [],
        'cleanedMol': []
    }
    
    # Compute fingerprints and molecular properties
    print("Computing molecular fingerprints and properties...")
    fps = []
    mols = []
    weights = []
    valid_smiles = []
    
    for i, smiles in enumerate(qsar_df['Canonical SMILES'].values):
        if i % 20 == 0:
            print(f"  Processed {i}/{len(qsar_df)}...")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Morgan Fingerprint (2048 bits, radius 2)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
                mols.append(mol)
                weights.append(Descriptors.MolWt(mol))
                valid_smiles.append(True)
            else:
                fps.append(None)
                mols.append(None)
                weights.append(None)
                valid_smiles.append(False)
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            fps.append(None)
            mols.append(None)
            weights.append(None)
            valid_smiles.append(False)
    
    gif_ready_data['IsValidSMILES'] = valid_smiles
    gif_ready_data['Morgan_FP'] = fps
    gif_ready_data['Molecule'] = mols
    gif_ready_data['Fingerprint'] = fps  # Same as Morgan_FP
    gif_ready_data['MolecularWeight'] = weights
    
    # Filter to only valid SMILES
    valid_mask = np.array(valid_smiles)
    print(f"\nValid SMILES: {sum(valid_mask)}/{len(valid_smiles)}")
    
    # Convert fingerprints to arrays for PCA
    print("Computing PCA projections...")
    valid_fps = np.array([list(fp) for fp, valid in zip(fps, valid_smiles) if valid])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(valid_fps)
    
    pca1_vals = [None] * len(valid_smiles)
    pca2_vals = [None] * len(valid_smiles)
    pca_idx = 0
    for i, valid in enumerate(valid_smiles):
        if valid:
            pca1_vals[i] = pca_result[pca_idx, 0]
            pca2_vals[i] = pca_result[pca_idx, 1]
            pca_idx += 1
    
    gif_ready_data['PCA1'] = pca1_vals
    gif_ready_data['PCA2'] = pca2_vals
    
    # Compute t-SNE (sampled if too many points)
    print("Computing t-SNE projections...")
    sample_size = min(100, len(valid_fps))  # Limit to 100 for speed
    
    if len(valid_fps) > sample_size:
        sample_indices_into_valid = np.random.choice(len(valid_fps), sample_size, replace=False)
        sample_indices_into_valid = np.sort(sample_indices_into_valid)
        sample_fps = valid_fps[sample_indices_into_valid]
    else:
        sample_indices_into_valid = np.arange(len(valid_fps))
        sample_fps = valid_fps
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_fps)-1))
    tsne_result = tsne.fit_transform(sample_fps)
    
    tsne1_vals = [None] * len(valid_smiles)
    tsne2_vals = [None] * len(valid_smiles)
    # Map back from valid_fps indices to original array indices
    valid_indices = np.where(valid_mask)[0]
    for i, fp_idx in enumerate(sample_indices_into_valid):
        orig_idx = valid_indices[fp_idx]
        tsne1_vals[orig_idx] = tsne_result[i, 0]
        tsne2_vals[orig_idx] = tsne_result[i, 1]
    
    gif_ready_data['tSNE1'] = tsne1_vals
    gif_ready_data['tSNE2'] = tsne2_vals
    gif_ready_data['Frequency'] = [1] * len(valid_smiles)
    
    # Create final DataFrame
    gif_df = pd.DataFrame(gif_ready_data)
    
    # Save the dataset
    output_file = 'data/QSAR_potency_20um_for_GIF.xlsx'
    gif_df.to_excel(output_file, index=False, sheet_name='QSAR_Potency')
    print(f"\n✅ Saved GIF-ready dataset to: {output_file}")
    print(f"Shape: {gif_df.shape}")
    print(f"\nDataset columns:")
    print(gif_df.columns.tolist())
    print(f"\nClass distribution:")
    print(f"  Potent (1): {sum(gif_df['classLabel'] == 1)}")
    print(f"  Not Potent (0): {sum(gif_df['classLabel'] == 0)}")

if __name__ == '__main__':
    main()
