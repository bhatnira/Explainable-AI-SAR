# QSAR IC50 Analysis - Main Dataset

## Main Dataset (Input Sheet)
**Total instances with IC50 data: 177**

### IC50 Value Counts (in µM):
- **< 5 µM: 36 instances** (highly active)
- **< 10 µM: 60 instances** (very active)
- **< 30 µM: 106 instances** (active)
- **< 40 µM: 110 instances** (moderately active)

## Dataset Breakdown by Series
All QSAR series in the spreadsheet:

| Series | Sheet Name | Total IC50 Values | <5 µM | <10 µM | <30 µM | <40 µM |
|--------|-----------|------------------|-------|--------|--------|--------|
| A (Triazole) | Series A Triazole | 117 | 27 | 39 | 68 | 69 |
| B (Cysteine) | Series B Cysteine | 37 | 4 | 9 | 24 | 27 |
| C (Spiro) | Series C Spiro | 5 | 3 | 5 | 5 | 5 |
| D (Pyrrolidine) | Series D Pyrrolidine | 5 | 2 | 5 | 5 | 5 |
| E (Spiro 2) | Series E Spiro (2) | 6 | 0 | 1 | 4 | 4 |
| **Total** | | **170** | **36** | **59** | **106** | **110** |

## File Location
- **File**: `Tzu-QSAR-only/data/raw/TB Project QSAR.xlsx`
- **Main data sheet**: `Input`
- **IC50 Column**: Column F (`IC50 uM`)
- **pIC50 Column**: Column G (`PIC50`)

## Data Structure
The Excel file contains:
- **Structure**: Molecular structure (images stored in cells)
- **Identifier**: Compound ID (e.g., TB-VS0012)
- **Series**: Compound series (A, B, C, D, or E)
- **Canonical SMILES**: Two columns with molecular structure notation
- **IC50 uM**: IC50 value in micromolar units
- **PIC50**: pIC50 value (log-based IC50)

## Integration in Docker
- **Script**: `scripts/analyze_qsar_ic50.py` - Analyzes IC50 distribution
- **Services configured**:
  - `qsar-analysis` - Main analysis
  - `qsar-train` - Model training
  - `qsar-descriptors` - Descriptor computation
  - `qsar-visualize` - Visualization

Run with Docker:
```bash
docker compose up qsar-analysis
```

## Dependencies Integrated
✓ openpyxl - Excel file reading
✓ pandas - Data handling
✓ RDKit - Chemical informatics & SMILES processing
✓ umap-learn, hdbscan - Clustering
✓ scikit-learn - ML models
✓ matplotlib, plotly - Visualization
✓ All other Explainable-AI-SAR dependencies
