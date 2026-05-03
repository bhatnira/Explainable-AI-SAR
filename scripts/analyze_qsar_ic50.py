#!/usr/bin/env python3
"""
Analyze IC50 values from QSAR dataset.
"""
import sys
import os
import pandas as pd

def main():
    # Find the Excel file
    possible_paths = [
        'Tzu-QSAR-only/data/raw/TB Project QSAR.xlsx',
        '/home/nbhatta1/Documents/Explainable-AI-SAR/Tzu-QSAR-only/data/raw/TB Project QSAR.xlsx',
        '/workspace/Tzu-QSAR-only/data/raw/TB Project QSAR.xlsx',
    ]
    
    excel_file = None
    for path in possible_paths:
        if os.path.exists(path):
            excel_file = path
            break
    
    if not excel_file:
        print(f'Error: Could not find TB Project QSAR.xlsx', file=sys.stderr)
        print(f'Searched in: {possible_paths}', file=sys.stderr)
        sys.exit(1)
    
    try:
        df = pd.read_excel(excel_file)
        
        print(f'Total instances: {len(df)}')
        print(f'Columns: {df.columns.tolist()}\n')
        
        ic50_col = [col for col in df.columns if 'ic50' in col.lower()]
        if ic50_col:
            ic50_col = ic50_col[0]
            ic50_vals = pd.to_numeric(df[ic50_col], errors='coerce')
            
            print(f'IC50 Value Counts:')
            print(f'  < 5 µM: {sum(ic50_vals < 5)} instances')
            print(f'  < 10 µM: {sum(ic50_vals < 10)} instances')
            print(f'  < 30 µM: {sum(ic50_vals < 30)} instances')
            print(f'  < 40 µM: {sum(ic50_vals < 40)} instances')
            
            print(f'\nIC50 Statistics:')
            print(ic50_vals.describe())
        else:
            print('No IC50 column found')
            print(df.head())
            
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
