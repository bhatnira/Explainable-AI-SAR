#!/usr/bin/env python3
"""
Generate GIFs using QSAR dataset with conda environment
"""
import sys
from pathlib import Path

# Add visualization module to path
sys.path.insert(0, str(Path('visualization').resolve()))

from create_dynamic_parameter_movies import DynamicParameterMovieCreator

def main():
    print("\n" + "="*70)
    print("GIF GENERATION WITH QSAR DATASET (CONDA ENVIRONMENT)")
    print("="*70)
    
    movie_creator = DynamicParameterMovieCreator()
    
    print("\n[1/3] Creating GIF with Circular Fingerprints + TPOT...")
    try:
        gif_path = movie_creator.create_dynamic_parameter_movie('circular_fingerprint')
        print(f"✅ Saved: {gif_path}")
        print(f"   Dataset: QSAR potency classification (20 µM threshold)")
        print(f"   Compounds: 164 molecules (80 potent, 84 not potent)")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("✅ GIF GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output GIFs in: visualization/")
    print(f"📊 Dataset: QSAR_potency_20um_for_GIF.xlsx")
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
