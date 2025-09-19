#!/usr/bin/env python3
"""
🤖 TPOT AutoML Molecular Classification
Simple training script using circular fingerprints with 100 data instances
"""

import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

def generate_circular_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Generate circular fingerprints for molecules"""
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Generate Morgan fingerprint (circular fingerprint)
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            # Convert to numpy array
            fp_array = np.array(list(fp.ToBitString()), dtype=int)
            fingerprints.append(fp_array)
        else:
            # Handle invalid SMILES with zero vector
            fingerprints.append(np.zeros(n_bits, dtype=int))
    return np.array(fingerprints)

def train_tpot_model():
    """Train TPOT AutoML model on molecular data"""
    
    print("🤖 TPOT AutoML Molecular Classification")
    print("=" * 50)
    
    # Start total timer
    total_start_time = time.time()
    
    # Load data
    data_file = "StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx"
    print(f"📂 Loading data from: {data_file}")
    
    try:
        df = pd.read_excel(data_file)
        print(f"✅ Loaded {len(df)} molecules")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Sample 100 molecules with balanced classes
    print("📊 Sampling to 100 molecules...")
    
    # Get balanced sample (50 from each class)
    class_0 = df[df['classLabel'] == 0].sample(n=50, random_state=42)
    class_1 = df[df['classLabel'] == 1].sample(n=50, random_state=42)
    
    # Combine samples
    sampled_df = pd.concat([class_0, class_1], ignore_index=True)
    print(f"📊 Final samples: {len(sampled_df)}")
    
    # Check class distribution
    class_dist = sampled_df['classLabel'].value_counts().to_dict()
    print(f"📊 Class distribution: {class_dist}")
    
    # Generate circular fingerprints
    print("🧪 Generating circular fingerprints...")
    X = generate_circular_fingerprints(sampled_df['cleanedMol'].tolist())
    y = sampled_df['classLabel'].values
    
    print(f"   Fingerprint shape: {X.shape}")
    print(f"   Features: {X.shape[1]} circular fingerprint bits")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Dataset splits:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Train TPOT model
    print("🤖 Training TPOT AutoML model...")
    
    try:
        from tpot import TPOTClassifier
        
        print("🔧 Model Configuration:")
        print("   AutoML: TPOT")
        print("   Generations: 5")
        print("   Population size: 20")
        print("   CV folds: 3")
        print("   Scoring: accuracy")
        
        print("🏋️ Starting AutoML optimization...")
        start_time = time.time()
        
        # Create TPOT classifier with reduced parameters for faster training
        tpot = TPOTClassifier(
            generations=5,
            population_size=20,
            cv=3,
            scoring='accuracy',
            random_state=42,
            verbosity=1,
            n_jobs=1,  # Single thread to avoid conflicts
            max_time_mins=10  # Limit training time
        )
        
        # Fit TPOT
        tpot.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        
        # Make predictions
        print("🔬 Evaluating model...")
        y_pred = tpot.predict(X_test)
        y_pred_proba = tpot.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n🎯 Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   Training time: {training_time:.1f}s")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Get the best pipeline
        print(f"\n🏆 Best Pipeline:")
        print(f"   {tpot.fitted_pipeline_}")
        
        # Calculate total time
        total_time = time.time() - total_start_time
        
        # Save results
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'training_time': training_time,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_time': total_time,
            'best_pipeline': str(tpot.fitted_pipeline_),
            'fingerprint_bits': X.shape[1]
        }
        
        results_file = 'tpot_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to {results_file}")
        print(f"⏱️  Total execution time: {total_time:.1f}s")
        print("🎉 TPOT training completed!")
        
        return results
        
    except ImportError:
        print("❌ TPOT not installed. Please install with: pip install tpot")
        return None
    except Exception as e:
        print(f"❌ TPOT training failed: {e}")
        print("❌ Training failed.")
        return None

if __name__ == "__main__":
    # Set environment variables for single-threaded execution
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Train model
    train_tpot_model()
