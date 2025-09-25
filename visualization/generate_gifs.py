#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path('visualization').resolve()))
from create_dynamic_parameter_movies import DynamicParameterMovieCreator

if __name__ == "__main__":
    m = DynamicParameterMovieCreator()
    print("[1/1] Circular Fingerprint with TPOT AutoML...")
    try:
        p1 = m.create_dynamic_parameter_movie('circular_fingerprint')
        print("✅ Saved:", p1)
    except Exception as e:
        print("❌ Circular fingerprint failed:", e)
        import traceback
        traceback.print_exc()

    print("\n🎬 GIF generation complete!")
    print("🔍 Check the visualization/ folder for the generated GIF.")
