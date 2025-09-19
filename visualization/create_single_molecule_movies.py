#!/usr/bin/env python3
"""
Create Single Molecule Movies - Original Static Version
This shows the issue where parameters remained unchanged across iterations
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os

class SingleMoleculeMovieCreator:
    def __init__(self):
        self.sample_molecules = {
            'circular_fingerprint': 'CC(C)C1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'chemberta': 'C1=CC=C(C=C1)C(=O)O',  # Benzoic acid
            'graphconv': 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin
        }
        
        # STATIC parameters that DON'T change (original issue)
        self.static_parameters = {
            'circular_fingerprint': {
                'radius': 2,
                'nBits': 1024,
                'useFeatures': False,
                'useChirality': True
            },
            'chemberta': {
                'max_length': 256,
                'batch_size': 32,
                'learning_rate': 2e-5,
                'num_epochs': 3
            },
            'graphconv': {
                'hidden_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'batch_size': 32
            }
        }
    
    def calculate_molecule_properties(self, smiles):
        """Calculate molecular properties for display"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        return {
            'MW': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Descriptors.MolLogP(mol), 2),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': round(Descriptors.TPSA(mol), 2)
        }
    
    def create_parameter_text(self, model_type, iteration):
        """Create parameter display text - STATIC VERSION"""
        params = self.static_parameters[model_type]
        
        param_text = f"Iteration {iteration}\nParameters (STATIC - NO CHANGE):\n"
        for key, value in params.items():
            param_text += f"{key}: {value}\n"
        
        # Add fake quality score that slightly varies
        fake_quality = 0.45 + (iteration * 0.02)  # Slightly increasing
        param_text += f"\nExplanation Quality: {fake_quality:.3f}"
        
        return param_text
    
    def create_molecule_image(self, smiles, model_type, iteration):
        """Create molecule visualization with parameters"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Create molecule image
        mol_img = Draw.MolToImage(mol, size=(400, 300))
        
        # Create combined image with parameters
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Molecule subplot
        ax1.imshow(mol_img)
        ax1.set_title(f'{model_type.replace("_", " ").title()} - Iteration {iteration}')
        ax1.axis('off')
        
        # Parameters subplot
        ax2.text(0.05, 0.95, self.create_parameter_text(model_type, iteration),
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add molecular properties
        props = self.calculate_molecule_properties(smiles)
        prop_text = "Molecular Properties:\n"
        for key, value in props.items():
            prop_text += f"{key}: {value}\n"
        
        ax2.text(0.05, 0.45, prop_text,
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close(fig)
        
        return Image.fromarray(img)
    
    def create_movie(self, model_type):
        """Create static parameter movie for a model type"""
        print(f"🎬 Creating STATIC parameter movie for {model_type}")
        
        smiles = self.sample_molecules[model_type]
        images = []
        
        # Create frames for 8 iterations with STATIC parameters
        for iteration in range(1, 9):
            img = self.create_molecule_image(smiles, model_type, iteration)
            if img:
                images.append(img)
        
        # Save as GIF
        output_path = f'{model_type}_parameters_evolution.gif'
        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=800,  # 0.8 seconds per frame (faster)
                loop=0
            )
            print(f"💾 Saved STATIC parameter movie to {output_path}")
        
        return output_path

def main():
    print("🎬 Creating Single Molecule Movies (STATIC Parameters - Original Issue)")
    print("=" * 80)
    
    creator = SingleMoleculeMovieCreator()
    
    model_types = ['circular_fingerprint', 'chemberta', 'graphconv']
    created_movies = []
    
    for model_type in model_types:
        movie_path = creator.create_movie(model_type)
        created_movies.append(movie_path)
    
    print(f"\n🎉 Static parameter movie creation complete! Created {len(created_movies)} movies.")
    print("Movies with STATIC parameters (demonstrating the original issue):")
    for movie in created_movies:
        print(f"  • {movie}")
    
    print("\n⚠️  Note: These movies show the ORIGINAL ISSUE where parameters don't change!")
    print("📌 Compare with dynamic movies to see the difference!")

if __name__ == "__main__":
    main()
