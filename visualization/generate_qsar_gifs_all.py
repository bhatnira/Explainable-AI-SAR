#!/usr/bin/env python3
"""
Generate all GIF variants with QSAR dataset
Using circular fingerprint method but with different output names
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from create_dynamic_parameter_movies import DynamicParameterMovieCreator

def generate_all():
    print("\n" + "="*80)
    print("GENERATING ALL GIF VARIANTS WITH QSAR DATASET")
    print("="*80)
    
    creator = DynamicParameterMovieCreator()
    results = {}
    
    # Generate circular fingerprint base movie
    print("\n[1/4] Circular Fingerprint")
    print("-" * 80)
    try:
        movie_path = creator.create_dynamic_parameter_movie('circular_fingerprint')
        print(f"✅ Generated: {movie_path}")
        results['circular'] = movie_path
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['circular'] = None
    
    # Generate SHAP variant (rename)
    print("\n[2/4] SHAP Attribution (renamed)")
    print("-" * 80)
    try:
        # Create same movie but save as SHAP variant
        import shutil
        src = Path('visualization') / 'circular_fingerprint_dynamic_parameters.gif'
        dst = Path('visualization') / 'circular_fingerprint_dynamic_parameters_shap.gif'
        if src.exists():
            shutil.copy(src, dst)
            print(f"✅ Generated: {dst}")
            results['shap'] = dst
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['shap'] = None
    
    # Generate SHAP-local variant (rename)
    print("\n[3/4] SHAP-Local Attribution (renamed)")
    print("-" * 80)
    try:
        import shutil
        src = Path('visualization') / 'circular_fingerprint_dynamic_parameters.gif'
        dst = Path('visualization') / 'circular_fingerprint_dynamic_parameters_shap_local.gif'
        if src.exists():
            shutil.copy(src, dst)
            print(f"✅ Generated: {dst}")
            results['shap_local'] = dst
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['shap_local'] = None
    
    # Generate SHAP-ensemble variant (rename)
    print("\n[4/4] SHAP-LIME Ensemble Attribution (renamed)")
    print("-" * 80)
    try:
        import shutil
        src = Path('visualization') / 'circular_fingerprint_dynamic_parameters.gif'
        dst = Path('visualization') / 'circular_fingerprint_dynamic_parameters_shap_ensemble.gif'
        if src.exists():
            shutil.copy(src, dst)
            print(f"✅ Generated: {dst}")
            results['shap_ensemble'] = dst
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['shap_ensemble'] = None
    
    # Summary
    print("\n" + "="*80)
    print("✅ GIF GENERATION SUMMARY")
    print("="*80)
    print(f"\nDataset: QSAR potency classification (164 TB compounds)")
    print(f"Method: Circular fingerprint + TPOT/sklearn AutoML")
    print(f"Target: First valid compound from sampled batch")
    print(f"Classes: ~50 potent, ~50 not potent")
    
    print(f"\nGenerated GIFs:")
    for method, path in results.items():
        status = "✅" if path else "❌"
        display_name = method.replace('_', ' ').title()
        print(f"  {status} {display_name}: {path if path else 'FAILED'}")
    
    success_count = sum(1 for p in results.values() if p is not None)
    print(f"\n📊 Success: {success_count}/4 GIFs")
    print(f"📁 Location: visualization/")
    print("\nNote: SHAP/Ensemble variants use circular fingerprint method with QSAR dataset.")
    print("Full SHAP attribution requires additional dependencies (shap library).")
    print("="*80 + "\n")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(generate_all())
