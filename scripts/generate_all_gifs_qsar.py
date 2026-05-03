#!/usr/bin/env python3
"""
Generate ALL GIFs using QSAR dataset (Series C target compound)
This script generates GIFs using:
- Dataset: QSAR potency classification (20 µM threshold)
- Target: ROY-0000220-001 from Series C (Spiro, IC50=0.78 µM, highly potent)
- Compounds: 164 TB drug candidates (80 potent, 84 not potent)
"""
import sys
from pathlib import Path

# Add visualization module to path
sys.path.insert(0, str(Path('visualization').resolve()))

def generate_all_gifs():
    print("\n" + "="*80)
    print("GENERATING ALL GIFS WITH QSAR DATASET (SERIES C TARGET)")
    print("="*80)
    
    results = {}
    
    # 1. Circular Fingerprint with TPOT
    print("\n[1/4] Circular Fingerprint + TPOT AutoML")
    print("-" * 80)
    try:
        from create_dynamic_parameter_movies import DynamicParameterMovieCreator
        creator = DynamicParameterMovieCreator()
        gif_path = creator.create_dynamic_parameter_movie('circular_fingerprint')
        print(f"✅ Saved: {gif_path}")
        results['circular'] = gif_path
    except Exception as e:
        print(f"❌ Error: {e}")
        results['circular'] = None
    
    # 2. SHAP-based GIF
    print("\n[2/4] SHAP-based Attribution (Dataset SHAP + LIME highlighting)")
    print("-" * 80)
    try:
        from create_dynamic_parameter_movies_shap import ShapDynamicParameterMovieCreator
        creator = ShapDynamicParameterMovieCreator()
        gif_path = creator.create_dynamic_parameter_movie('circular_fingerprint')
        print(f"✅ Saved: {gif_path}")
        results['shap'] = gif_path
    except Exception as e:
        print(f"❌ Error: {e}")
        results['shap'] = None
    
    # 3. SHAP-local GIF
    print("\n[3/4] SHAP-Local Attribution (Pure SHAP local explanations)")
    print("-" * 80)
    try:
        from create_dynamic_parameter_movies_shap_local import ShapLocalDynamicParameterMovieCreator
        creator = ShapLocalDynamicParameterMovieCreator()
        gif_path = creator.create_dynamic_parameter_movie('circular_fingerprint')
        print(f"✅ Saved: {gif_path}")
        results['shap_local'] = gif_path
    except Exception as e:
        print(f"❌ Error: {e}")
        results['shap_local'] = None
    
    # 4. SHAP-LIME Ensemble GIF
    print("\n[4/4] SHAP-LIME Ensemble Attribution (Blended SHAP + LIME)")
    print("-" * 80)
    try:
        from create_dynamic_parameter_movies_shap_ensemble import ShapLimeEnsembleMovieCreator
        creator = ShapLimeEnsembleMovieCreator()
        gif_path = creator.create_dynamic_parameter_movie('circular_fingerprint')
        print(f"✅ Saved: {gif_path}")
        results['shap_ensemble'] = gif_path
    except Exception as e:
        print(f"❌ Error: {e}")
        results['shap_ensemble'] = None
    
    # Summary
    print("\n" + "="*80)
    print("✅ GIF GENERATION SUMMARY")
    print("="*80)
    print(f"\nDataset: QSAR potency classification (164 TB compounds)")
    print(f"Target: ROY-0000220-001 (Series C - Spiro, IC50=0.78 µM, highly potent)")
    print(f"Classes: 80 potent (< 20 µM), 84 not potent (≥ 20 µM)")
    
    print(f"\nGenerated GIFs:")
    for method, path in results.items():
        status = "✅" if path else "❌"
        print(f"  {status} {method.replace('_', ' ').title()}: {path if path else 'FAILED'}")
    
    success_count = sum(1 for p in results.values() if p is not None)
    print(f"\n📊 Success: {success_count}/4 GIFs generated")
    print(f"📁 All GIFs in: visualization/")
    print("\n" + "="*80 + "\n")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(generate_all_gifs())
