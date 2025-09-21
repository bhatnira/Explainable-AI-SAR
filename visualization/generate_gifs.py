#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path('visualization').resolve()))
from create_dynamic_parameter_movies import DynamicParameterMovieCreator

if __name__ == "__main__":
    m = DynamicParameterMovieCreator()
    print("[1/3] Circular Fingerprint...")
    try:
        p1 = m.create_dynamic_parameter_movie('circular_fingerprint')
        print("Saved:", p1)
    except Exception as e:
        print("Circular failed:", e)

    print("[2/3] GraphConv...")
    try:
        p2 = m.create_graphconv_dynamic_parameter_movie()
        print("Saved:", p2)
    except Exception as e:
        print("GraphConv failed:", e)

    print("[3/3] ChemBERTa...")
    try:
        p3 = m.create_chemberta_dynamic_parameter_movie()
        print("Saved:", p3)
    except Exception as e:
        print("ChemBERTa failed:", e)
