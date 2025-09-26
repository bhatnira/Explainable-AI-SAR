#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path('visualization').resolve()))
from create_dynamic_parameter_movies_shap import ShapDynamicParameterMovieCreator

if __name__ == "__main__":
    m = ShapDynamicParameterMovieCreator()
    print("[1/1] Circular Fingerprint with SHAP...")
    try:
        p1 = m.create_dynamic_parameter_movie_shap('circular_fingerprint')
        print("✅ Saved:", p1)
    except Exception as e:
        print("❌ SHAP movie failed:", e)
        import traceback
        traceback.print_exc()

    print("\n🎬 SHAP GIF generation complete!")
    print("🔍 Check the visualization/ folder for the generated GIF.")
