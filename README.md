# 🧬 Explainable AI for Molecular Classification

A comprehensive toolkit for training and interpreting molecular classification models using ChemBERTa, GraphConv, and TPOT AutoML with high-quality visualizations.

## 📁 Project Structure

```
├── 📂 data/                    # Dataset files
│   └── StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx
├── 📂 models/                  # Training and interpretation scripts
│   ├── train_chemberta_simple.py      # ChemBERTa transformer model
│   ├── train_graphconv_fixed.py       # GraphConv neural network
│   ├── train_tpot_simple.py          # TPOT AutoML pipeline
│   ├── interpret_chemberta.py         # ChemBERTa attention visualization
│   ├── interpret_graphconv.py         # GraphConv molecular highlighting
│   └── interpret_tpot.py             # TPOT feature importance
├── 📂 results/                 # Model outputs and metrics
│   ├── chemberta_results.json
│   ├── graphconv_results.json
│   ├── tpot_results.json
│   ├── chemberta_output/              # ChemBERTa model artifacts
│   └── runs/                          # Training logs
├── 📂 visualizations/          # Interpretation visualizations
│   ├── chemberta_attention_molecule_*.png    # Attention maps
│   ├── graphconv_interpretation_molecule_*_structure.png  # Molecular highlighting
│   └── tpot_interpretation_molecule_*.png    # Feature importance
├── 📂 chemberta_env/          # Python virtual environment
└── 📂 cache_dir/              # Model cache files
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate the virtual environment
source chemberta_env/bin/activate

# Install required packages (already configured)
# - transformers, torch, rdkit-pypi, deepchem, tpot, matplotlib, pandas, numpy
```

### 2. Train All Models
```bash
# Train ChemBERTa transformer model
python models/train_chemberta_simple.py

# Train GraphConv neural network
python models/train_graphconv_fixed.py

# Train TPOT AutoML pipeline
python models/train_tpot_simple.py
```

### 3. Generate Interpretations
```bash
# Generate ChemBERTa attention visualizations
python models/interpret_chemberta.py

# Generate GraphConv molecular structure highlighting
python models/interpret_graphconv.py

# Generate TPOT feature importance visualizations
python models/interpret_tpot.py
```

## 🤖 Models Overview

### 1. ChemBERTa (Transformer)
- **Purpose**: Transformer-based molecular classification using SMILES strings
- **Architecture**: Pre-trained ChemBERTa with fine-tuning
- **Input**: SMILES molecular representations
- **Interpretation**: Attention weight visualization on molecular atoms
- **Performance**: ~45% accuracy, ~34% ROC AUC

### 2. GraphConv (Graph Neural Network)
- **Purpose**: Graph-based molecular classification using molecular graphs
- **Architecture**: DeepChem GraphConv with ConvMolFeaturizer
- **Input**: Molecular graph representations
- **Interpretation**: Fragment-based atom contribution analysis
- **Performance**: ~50% accuracy, ~53% ROC AUC

### 3. TPOT AutoML
- **Purpose**: Automated machine learning pipeline optimization
- **Architecture**: Evolutionary algorithm with circular fingerprints
- **Input**: Morgan circular fingerprints (2048 bits)
- **Interpretation**: LIME-based feature importance with surrogate models
- **Performance**: ~65% accuracy, ~71% ROC AUC (best performer)

## 🎨 Visualization Features

### High-Quality Molecular Visualizations
All models generate publication-ready visualizations with:
- **300 DPI resolution** for crisp output
- **Professional color schemes**: Blue (positive), Red (negative), Gray (neutral)
- **Consistent visual standards** across all models
- **Clean molecular structure highlighting**

### ChemBERTa Attention Maps
- 4 attention visualizations per run
- Atom-level attention weight highlighting
- Color-coded contribution visualization
- Professional molecular structure rendering

### GraphConv Structure Highlighting
- Clean molecular structure with atom highlighting
- Gradient color intensity based on contribution magnitude
- High-resolution RDKit molecular drawings (800x800)
- Simple, focused design matching TPOT/ChemBERTa quality

### TPOT Feature Importance
- LIME-style interpretability with Random Forest surrogate
- Signed feature importance (positive/negative contributions)
- Circular fingerprint bit analysis
- Statistical summary of important features

## 📊 Dataset Information

- **Source**: `StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx`
- **Size**: 4,077 molecules total
- **Training**: 100 balanced samples (50 per class) for demonstration
- **Split**: 80/20 train/test ratio
- **Features**: SMILES strings, molecular weights, binary classification

## 🔧 Technical Details

### Data Processing
- Balanced sampling for consistent training
- SMILES standardization and validation
- Molecular weight filtering (<800 Da)
- Robust error handling for invalid molecules

### Model Training
- **ChemBERTa**: SimpleTransformers with binary classification
- **GraphConv**: DeepChem with ConvMolFeaturizer, 30 epochs
- **TPOT**: 5 generations, population size 20, CV=3

### Interpretation Methods
- **ChemBERTa**: Transformer attention weight extraction
- **GraphConv**: Fragment-based DeepChem interpretation
- **TPOT**: LIME with Random Forest surrogate models

## 📈 Results Summary

| Model | Accuracy | ROC AUC | Training Time | Interpretation Method |
|-------|----------|---------|---------------|----------------------|
| ChemBERTa | 45% | 34% | ~30s | Attention Maps |
| GraphConv | 50% | 53% | ~3s | Fragment Analysis |
| TPOT AutoML | 65% | 71% | ~60s | Feature Importance |

## 🎯 Key Features

✅ **Three complementary ML approaches** (Transformer, GNN, AutoML)  
✅ **High-quality visualizations** with consistent design standards  
✅ **Comprehensive interpretability** for all models  
✅ **Production-ready code** with error handling  
✅ **Modular architecture** for easy extension  
✅ **Publication-quality outputs** (300 DPI, professional formatting)  

## 🔍 Usage Examples

### Individual Model Training
```bash
# Example: Train just the ChemBERTa model
python models/train_chemberta_simple.py
# Output: chemberta_results.json + model artifacts
```

### Batch Interpretation Generation
```bash
# Generate all interpretations at once
for script in models/interpret_*.py; do
    python "$script"
done
# Output: Multiple PNG visualizations in visualizations/
```

### Custom Dataset
Replace the Excel file in `data/` with your own molecular dataset following the same format:
- SMILES column for molecular structures
- Binary target variable for classification
- Molecular weight information (optional)

## 🚀 Future Enhancements

- [ ] Multi-class classification support
- [ ] Additional molecular descriptors (ECFP, MACCS keys)
- [ ] Cross-validation with multiple random seeds
- [ ] Ensemble model combining all three approaches
- [ ] Interactive web interface for visualization
- [ ] Docker containerization for reproducibility

## 📝 License

This project is part of the Explainable-AI-SAR repository for research and educational purposes.
