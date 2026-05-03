#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/nbhatta1/Documents/Explainable-AI-SAR')

from create_dynamic_parameter_movies import DynamicParameterMovieCreator

creator = DynamicParameterMovieCreator()
print("Creating ChemBERTa GIF with improved permutation-importance explanations...")
creator.create_chemberta_dynamic_parameter_movie()
print("✅ ChemBERTa GIF regenerated successfully!")
