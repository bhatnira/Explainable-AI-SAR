#!/usr/bin/env python3
"""
Molecular Explanation Evolution Movie Creator
===========================================

Creates an animated movie showing how molecular explanations evolve
through each iteration of the agentic optimization process.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import random
from PIL import Image, ImageDraw, ImageFont
import io
import base64

class MolecularExplanationMovieCreator:
    def __init__(self):
        self.colors = {
            'circular_fingerprint': '#FF6B6B',
            'tpot': '#4ECDC4'
        }
        
        # Sample molecules for demonstration
        self.sample_molecules = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
            "CC(=O)OC1=CC=CC=C1C(=O)O",       # Aspirin
            "C1=CC=C(C=C1)CCN",               # Phenethylamine
            "CC1=CC=C(C=C1)C(C)(C)C",         # tert-Butylbenzene
            "CC(C)(C)C1=CC=C(C=C1)O",         # BHT
            "C1=CC=C2C(=C1)C=CC=C2",          # Naphthalene
            "CC1=CC=CC=C1N",                  # Toluidine
        ]
        
    def load_optimization_data(self):
        """Load the agentic optimization results"""
        results_path = Path("../results/agentic_optimization_results.json")
        
        if not results_path.exists():
            print("❌ Results file not found.")
            return None
            
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def simulate_molecular_explanation(self, smiles, iteration, model_type, quality_score, performance_score):
        """Simulate molecular explanation based on iteration progress"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Generate atom contributions based on iteration and scores
        num_atoms = mol.GetNumAtoms()
        
        # Base contributions influenced by iteration progress
        base_contributions = np.random.normal(0, 0.3, num_atoms)
        
        # Quality influences how well-defined the contributions are
        quality_factor = quality_score * 2  # Scale to make it more visible
        clarity_boost = quality_factor * 0.5
        
        # Performance influences the magnitude of contributions
        perf_factor = performance_score * 1.5
        
        # Iteration influences refinement (later iterations = more refined)
        iteration_factor = min(iteration / 10.0, 1.0)  # Normalize to 0-1
        
        # Generate refined contributions
        contributions = []
        for i in range(num_atoms):
            # Start with base contribution
            contrib = base_contributions[i]
            
            # Add quality-based clarity (less random, more structured)
            contrib += np.random.normal(0, 0.1) * (1 - clarity_boost)
            
            # Add performance-based magnitude
            contrib *= perf_factor
            
            # Add iteration-based refinement
            if iteration > 2:  # Later iterations show more structure
                atom = mol.GetAtomWithIdx(i)
                if atom.GetSymbol() in ['N', 'O']:  # Important atoms
                    contrib += 0.3 * iteration_factor
                elif atom.GetSymbol() == 'C' and len(atom.GetNeighbors()) > 2:  # Branched carbons
                    contrib += 0.2 * iteration_factor
            
            contributions.append(contrib)
        
        return contributions
    
    def draw_molecule_with_contributions(self, smiles, contributions, title, iteration, 
                                       quality_score, performance_score, model_type):
        """Draw molecule with atom contributions colored"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
        
        # Normalize contributions for coloring
        if contributions:
            min_contrib = min(contributions)
            max_contrib = max(contributions)
            contrib_range = max_contrib - min_contrib
            
            if contrib_range > 0:
                normalized_contribs = [(c - min_contrib) / contrib_range for c in contributions]
            else:
                normalized_contribs = [0.5] * len(contributions)
        else:
            normalized_contribs = [0.5] * mol.GetNumAtoms()
        
        # Set atom colors based on contributions
        atom_colors = {}
        for i, contrib in enumerate(normalized_contribs):
            if contrib > 0.7:  # Strong positive contribution
                atom_colors[i] = (0.2, 0.2, 1.0)  # Blue
            elif contrib > 0.6:
                atom_colors[i] = (0.4, 0.4, 1.0)  # Light blue
            elif contrib < 0.3:  # Negative contribution
                atom_colors[i] = (1.0, 0.2, 0.2)  # Red
            elif contrib < 0.4:
                atom_colors[i] = (1.0, 0.4, 0.4)  # Light red
            else:  # Neutral
                atom_colors[i] = (0.8, 0.8, 0.8)  # Gray
        
        # Draw molecule
        drawer.SetFontSize(12)
        drawer.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), 
                           highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        # Get image data
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        
        # Add title and metrics
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
            small_font = ImageFont.truetype("Arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title
        draw.text((10, 10), title, fill='black', font=font)
        draw.text((10, 30), f"Iteration {iteration}", fill='black', font=small_font)
        draw.text((10, 45), f"Quality: {quality_score:.3f}", fill='blue', font=small_font)
        draw.text((10, 60), f"Performance: {performance_score:.3f}", fill='green', font=small_font)
        draw.text((10, 75), f"Model: {model_type}", fill=self.colors.get(model_type, 'black'), font=small_font)
        
        return img
    
    def create_iteration_frame(self, model_data, iteration_idx, model_type):
        """Create a frame showing molecular explanations for a specific iteration"""
        if iteration_idx >= len(model_data['all_results']):
            return None
            
        result = model_data['all_results'][iteration_idx]
        
        # Create figure with subplots for multiple molecules
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{model_type.replace('_', ' ').title()} - Iteration {result['iteration']}\n"
                    f"Quality: {result['explanation_quality']:.3f}, "
                    f"Performance: {result['performance']:.3f}, "
                    f"Combined: {result['combined_score']:.3f}", 
                    fontsize=16, fontweight='bold')
        
        # Select 4 representative molecules
        selected_molecules = random.sample(self.sample_molecules, 4)
        
        for idx, (ax, smiles) in enumerate(zip(axes.flat, selected_molecules)):
            # Generate molecular explanation for this iteration
            contributions = self.simulate_molecular_explanation(
                smiles, result['iteration'], model_type, 
                result['explanation_quality'], result['performance']
            )
            
            if contributions:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Create molecule image
                    mol_img = self.draw_molecule_with_contributions(
                        smiles, contributions, f"Molecule {idx+1}", 
                        result['iteration'], result['explanation_quality'], 
                        result['performance'], model_type
                    )
                    
                    if mol_img:
                        ax.imshow(mol_img)
                        ax.axis('off')
                    else:
                        ax.text(0.5, 0.5, f"Molecule {idx+1}\nSMILES: {smiles[:20]}...", 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                else:
                    ax.text(0.5, 0.5, f"Invalid molecule {idx+1}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"No explanation\nfor molecule {idx+1}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save frame
        frame_path = Path(f"frames/{model_type}_iteration_{result['iteration']:02d}.png")
        frame_path.parent.mkdir(exist_ok=True)
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return frame_path
    
    def create_progress_visualization_frame(self, all_model_data, current_iteration):
        """Create a frame showing overall progress across all models"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Score evolution
        for model_name, model_data in all_model_data.items():
            if 'all_results' not in model_data:
                continue
                
            iterations = []
            qualities = []
            performances = []
            combined = []
            
            for result in model_data['all_results'][:current_iteration+1]:
                iterations.append(result['iteration'])
                qualities.append(result['explanation_quality'])
                performances.append(result['performance'])
                combined.append(result['combined_score'])
            
            if iterations:
                ax1.plot(iterations, combined, 'o-', 
                        color=self.colors.get(model_name, 'gray'),
                        label=model_name.replace('_', ' ').title(), linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Combined Score')
        ax1.set_title('Score Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quality vs Performance scatter
        for model_name, model_data in all_model_data.items():
            if 'all_results' not in model_data:
                continue
                
            qualities = []
            performances = []
            
            for result in model_data['all_results'][:current_iteration+1]:
                qualities.append(result['explanation_quality'])
                performances.append(result['performance'])
            
            if qualities:
                ax2.scatter(qualities, performances, 
                           color=self.colors.get(model_name, 'gray'),
                           label=model_name.replace('_', ' ').title(), 
                           alpha=0.7, s=60)
        
        ax2.set_xlabel('Explanation Quality')
        ax2.set_ylabel('Performance')
        ax2.set_title('Quality vs Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Current best scores
        current_best = {}
        for model_name, model_data in all_model_data.items():
            if 'all_results' not in model_data:
                continue
                
            best_score = 0
            for result in model_data['all_results'][:current_iteration+1]:
                if result['combined_score'] > best_score:
                    best_score = result['combined_score']
            
            current_best[model_name] = best_score
        
        if current_best:
            models = list(current_best.keys())
            scores = list(current_best.values())
            colors = [self.colors.get(model, 'gray') for model in models]
            
            bars = ax3.bar([m.replace('_', '\n') for m in models], scores, color=colors, alpha=0.7)
            ax3.set_ylabel('Best Combined Score')
            ax3.set_title(f'Best Scores (up to iteration {current_iteration})')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save frame
        frame_path = Path(f"frames/progress_iteration_{current_iteration:02d}.png")
        frame_path.parent.mkdir(exist_ok=True)
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return frame_path
    
    def create_individual_model_movie(self, model_name, model_data):
        """Create movie for individual model"""
        print(f"🎬 Creating movie for {model_name}")
        
        if 'all_results' not in model_data:
            print(f"❌ No results data for {model_name}")
            return None
        
        frames = []
        
        # Create frames for each iteration
        for i in range(len(model_data['all_results'])):
            frame_path = self.create_iteration_frame(model_data, i, model_name)
            if frame_path:
                frames.append(str(frame_path))
        
        if not frames:
            print(f"❌ No frames created for {model_name}")
            return None
        
        # Create movie using matplotlib animation
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        def animate(frame_idx):
            ax.clear()
            ax.axis('off')
            
            if frame_idx < len(frames):
                img = plt.imread(frames[frame_idx])
                ax.imshow(img)
                ax.set_title(f"{model_name.replace('_', ' ').title()} - Evolution Frame {frame_idx+1}/{len(frames)}", 
                            fontsize=16, pad=20)
            
            return [ax]
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                     interval=1500, blit=False, repeat=True)
        
        # Save as gif
        movie_path = Path(f"movies/{model_name}_evolution.gif")
        movie_path.parent.mkdir(exist_ok=True)
        
        print(f"💾 Saving {model_name} movie to {movie_path}")
        ani.save(movie_path, writer='pillow', fps=0.67)  # ~1.5 seconds per frame
        plt.close()
        
        return movie_path
    
    def create_overall_progress_movie(self, results):
        """Create movie showing overall progress across all models"""
        print("🎬 Creating overall progress movie")
        
        model_data = results['model_results']
        
        # Find maximum iterations across all models
        max_iterations = 0
        for model_name, data in model_data.items():
            if 'all_results' in data:
                max_iterations = max(max_iterations, len(data['all_results']))
        
        if max_iterations == 0:
            print("❌ No iteration data found")
            return None
        
        frames = []
        
        # Create progress frames
        for i in range(max_iterations):
            frame_path = self.create_progress_visualization_frame(model_data, i)
            if frame_path:
                frames.append(str(frame_path))
        
        if not frames:
            print("❌ No progress frames created")
            return None
        
        # Create animation
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.axis('off')
        
        def animate(frame_idx):
            ax.clear()
            ax.axis('off')
            
            if frame_idx < len(frames):
                img = plt.imread(frames[frame_idx])
                ax.imshow(img)
                ax.set_title(f"Agentic Optimization Progress - Iteration {frame_idx+1}/{len(frames)}", 
                            fontsize=18, pad=20)
            
            return [ax]
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                     interval=2000, blit=False, repeat=True)
        
        # Save as gif
        movie_path = Path("movies/overall_progress_evolution.gif")
        movie_path.parent.mkdir(exist_ok=True)
        
        print(f"💾 Saving overall progress movie to {movie_path}")
        ani.save(movie_path, writer='pillow', fps=0.5)  # 2 seconds per frame
        plt.close()
        
        return movie_path
    
    def create_all_movies(self):
        """Create all movies"""
        print("🎬 Creating Molecular Explanation Evolution Movies")
        print("=" * 60)
        
        # Load data
        results = self.load_optimization_data()
        if results is None:
            return
        
        # Create directories
        Path("frames").mkdir(exist_ok=True)
        Path("movies").mkdir(exist_ok=True)
        
        created_movies = []
        
        # Create individual model movies
        for model_name, model_data in results['model_results'].items():
            movie_path = self.create_individual_model_movie(model_name, model_data)
            if movie_path:
                created_movies.append(movie_path)
        
        # Create overall progress movie
        progress_movie = self.create_overall_progress_movie(results)
        if progress_movie:
            created_movies.append(progress_movie)
        
        # Create summary report
        self.create_movie_summary_report(created_movies, results)
        
        print(f"\n🎉 Movie creation complete! Created {len(created_movies)} movies.")
        print("Check the movies/ directory for the animations.")
        
        return created_movies
    
    def create_movie_summary_report(self, movie_paths, results):
        """Create a summary report of created movies"""
        report = []
        report.append("🎬 MOLECULAR EXPLANATION EVOLUTION MOVIES REPORT")
        report.append("=" * 65)
        report.append("")
        
        report.append("📁 CREATED MOVIES:")
        report.append("-" * 20)
        for movie_path in movie_paths:
            report.append(f"• {movie_path}")
        report.append("")
        
        report.append("🎯 MOVIE DESCRIPTIONS:")
        report.append("-" * 25)
        
        for model_name, model_data in results['model_results'].items():
            if 'all_results' in model_data:
                report.append(f"\n📊 {model_name.upper()}_evolution.gif:")
                report.append(f"   Shows molecular explanation evolution over {len(model_data['all_results'])} iterations")
                report.append(f"   Demonstrates how explanation quality improved from {model_data['all_results'][0]['explanation_quality']:.3f}")
                report.append(f"   to {model_data['best_configuration']['explanation_quality']:.3f}")
                report.append(f"   Each frame shows 4 representative molecules with atom contributions")
        
        report.append(f"\n📈 overall_progress_evolution.gif:")
        report.append("   Shows overall optimization progress across all models")
        report.append("   Tracks score evolution, quality vs performance, and best scores")
        report.append("   Demonstrates the agentic learning process in action")
        
        report.append("\n💡 INTERPRETATION GUIDE:")
        report.append("-" * 25)
        report.append("🔵 Blue atoms: Positive contribution to prediction")
        report.append("🔴 Red atoms: Negative contribution to prediction") 
        report.append("⚪ Gray atoms: Neutral/minimal contribution")
        report.append("📈 Later iterations show more refined, structured explanations")
        report.append("🏆 Higher quality scores correlate with clearer atom contributions")
        
        report.append("\n🎬 TECHNICAL DETAILS:")
        report.append("-" * 20)
        report.append("• Frame rate: ~0.67 fps (1.5 seconds per iteration)")
        report.append("• Format: Animated GIF")
        report.append("• Resolution: 150 DPI")
        report.append("• Molecular visualizations: RDKit-based")
        report.append("• Atom contributions: Simulated based on iteration progress")
        
        # Save report
        report_text = "\n".join(report)
        report_path = Path("movies/movie_summary_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Movie summary report saved to: {report_path}")
        print("\n" + report_text)

def main():
    """Main function"""
    creator = MolecularExplanationMovieCreator()
    creator.create_all_movies()

if __name__ == "__main__":
    main()
