# 🧬 Explainable AI for Molecular Classification

A comprehensive toolkit for training and interpreting molecular classification models using TPOT AutoML with circular fingerprints and high-quality visualizations.

## 📁 Project Structure

```
├── 📂 data/                    # Dataset files
│   └── StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx
├── 📂 models/                  # Training and interpretation scripts
│   ├── train_tpot_simple.py          # TPOT AutoML pipeline
│   └── interpret_tpot.py             # TPOT feature importance
├── 📂 results/                 # Model outputs and metrics
│   ├── tpot_results.json
│   └── runs/                          # Training logs
├── 📂 visualizations/          # Interpretation visualizations
│   └── tpot_interpretation_molecule_*.png    # Feature importance
└── 📂 cache_dir/              # Model cache files
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate the virtual environment
source chemberta_env/bin/activate

# Install required packages (already configured)
# - tpot, rdkit-pypi, deepchem, matplotlib, pandas, numpy
```

### 2. Train Model
```bash
# Train TPOT AutoML pipeline
python models/train_tpot_simple.py
```

### 3. Generate Interpretations
```bash
# Generate TPOT feature importance visualizations
python models/interpret_tpot.py
```

## 🤖 Models Overview

### 1. TPOT AutoML with Circular Fingerprints
- **Purpose**: Automated machine learning pipeline optimization with fragment-based molecular representation
- **Architecture**: Evolutionary algorithm optimizing over scikit-learn pipelines with Morgan circular fingerprints
- **Input**: Morgan circular fingerprints (customizable radius and bits)
- **Interpretation**: LIME-based feature importance with surrogate models showing atom-level contributions
- **Performance**: ~65% accuracy, ~71% ROC AUC (optimized through evolutionary search)
- **Explainability**: Fragment-based explanations highlighting important molecular substructures

## 🎨 Visualization Features

### High-Quality Molecular Visualizations
The TPOT model generates publication-ready visualizations with:
- **300 DPI resolution** for crisp output
- **Professional color schemes**: Blue (positive), Red (negative), Gray (neutral)
- **Clean molecular structure highlighting**
- **Fragment-based importance mapping**

### TPOT Feature Importance Maps
- Atom-level contribution visualization using LIME
- Molecular structure highlighting based on circular fingerprint importance
- Clear indication of activating vs. deactivating molecular fragments
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
- **TPOT**: Evolutionary algorithm with circular fingerprints, configurable generations and population

### Interpretation Methods
- **TPOT**: LIME with optimized surrogate models for fragment-based explanations

## 📈 Results Summary

| Model | Accuracy | ROC AUC | Training Time | Interpretation Method |
|-------|----------|---------|---------------|----------------------|
| TPOT AutoML | 65% | 71% | ~60s | Fragment-based Feature Importance |

## 🎯 Key Features

✅ **Fragment-based molecular representation** with circular fingerprints  
✅ **Automated machine learning optimization** with evolutionary algorithms  
✅ **High-quality visualizations** with professional design standards  
✅ **Comprehensive interpretability** with LIME-based explanations  
✅ **Production-ready code** with error handling  
✅ **Modular architecture** for easy extension  
✅ **Publication-quality outputs** (300 DPI, professional formatting)  

## 🔍 Usage Examples

### Individual Model Training
```bash
# Train the TPOT AutoML model
python models/train_tpot_simple.py
# Output: tpot_results.json + model artifacts
```

### Interpretation Generation
```bash
# Generate TPOT interpretations
python models/interpret_tpot.py
# Output: PNG visualizations in visualizations/
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
