#!/usr/bin/env python3
"""
🧪 Simple GraphConv Model Training
Clean implementation for molecular classification using DeepChem
"""

import os
import pandas as pd
import time
import json

# Set environment for single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

def load_data(data_file='StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx', sample_size=100):
    """Load molecular data"""
    print(f"📂 Loading data from: {data_file}")
    
    df = pd.read_excel(data_file)
    print(f"✅ Loaded {len(df)} molecules")
    
    # Clean data
    df_clean = df[['cleanedMol', 'classLabel']].dropna()
    
    # Sample data to specified size
    if sample_size and sample_size < len(df_clean):
        # Balance classes in sample
        df_class_0 = df_clean[df_clean['classLabel'] == 0].sample(n=sample_size//2, random_state=42)
        df_class_1 = df_clean[df_clean['classLabel'] == 1].sample(n=min(sample_size//2, len(df_clean[df_clean['classLabel'] == 1])), random_state=42)
        df_clean = pd.concat([df_class_0, df_class_1], ignore_index=True)
        df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"📊 Sampled to {len(df_clean)} molecules")
    
    print(f"📊 Final samples: {len(df_clean)}")
    print(f"📊 Class distribution: {df_clean['classLabel'].value_counts().to_dict()}")
    
    return df_clean

def create_deepchem_datasets(df):
    """Create DeepChem datasets with ConvMolFeaturizer"""
    print("🧪 Creating DeepChem datasets...")
    
    try:
        import deepchem as dc
        from deepchem.feat import ConvMolFeaturizer
        from sklearn.model_selection import train_test_split
        
        # Define the ConvMolFeaturizer for graph representation
        featurizer = ConvMolFeaturizer()
        print("   Using ConvMolFeaturizer for molecular graphs")
        
        # Featurize the SMILES
        print("   Featurizing SMILES strings...")
        features = featurizer.featurize(df['cleanedMol'].tolist())
        targets = df['classLabel'].values
        
        # Create DeepChem dataset
        dataset = dc.data.NumpyDataset(features, targets)
        
        # Split data: 70% train, 15% valid, 15% test
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=0.3, random_state=42, stratify=targets
        )
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.5, random_state=42, stratify=y_train
        )
        
        # Create DeepChem datasets
        train_dataset = dc.data.NumpyDataset(X_train, y_train)
        valid_dataset = dc.data.NumpyDataset(X_valid, y_valid)
        test_dataset = dc.data.NumpyDataset(X_test, y_test)
        
        print(f"📊 Dataset splits:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Valid: {len(valid_dataset)} samples")
        print(f"   Test: {len(test_dataset)} samples")
        
        return train_dataset, valid_dataset, test_dataset
        
    except ImportError as e:
        print(f"❌ DeepChem not available: {e}")
        print("💡 Please install DeepChem: pip install deepchem")
        return None, None, None

def train_graphconv_model(train_dataset, valid_dataset, test_dataset):
    """Train GraphConv model"""
    print("🧪 Training GraphConv model...")
    
    try:
        import deepchem as dc
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        import tensorflow as tf
        from tensorflow.keras import backend as K
        
        # Clear Keras session
        K.clear_session()
        
        # Model configuration
        print("🔧 Model Configuration:")
        print("   Model: DeepChem GraphConvModel")
        print("   Batch size: 64")
        print("   Dropout: 0.01")
        print("   Epochs: 50")  # Reduced for faster training
        print("   Mode: classification")
        
        # Create GraphConv model
        n_tasks = 1
        model = dc.models.GraphConvModel(
            n_tasks, 
            batch_size=64,  # Smaller batch for 100 samples
            dropout=0.01,
            mode='classification'
        )
        
        print("🏋️ Starting training...")
        start_time = time.time()
        
        # Train model
        model.fit(train_dataset, nb_epoch=50)  # Reduced epochs
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        
        # Evaluate model
        print("🔬 Evaluating model...")
        
        # Get predictions
        train_predictions = model.predict(train_dataset)
        valid_predictions = model.predict(valid_dataset) 
        test_predictions = model.predict(test_dataset)
        
        # Calculate metrics for test set
        test_pred_classes = (test_predictions > 0.5).astype(int).flatten()
        test_true = test_dataset.y.flatten()
        
        accuracy = accuracy_score(test_true, test_pred_classes)
        roc_auc = roc_auc_score(test_true, test_predictions.flatten())
        
        print(f"\n🎯 Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   Training time: {training_time:.1f}s")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(test_true, test_pred_classes))
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'training_time': training_time,
            'train_samples': len(train_dataset),
            'valid_samples': len(valid_dataset),
            'test_samples': len(test_dataset)
        }
        
    except Exception as e:
        print(f"❌ GraphConv training failed: {e}")
        return None

def main():
    print("🧪 GraphConv Molecular Classification")
    print("=" * 50)
    
    total_start = time.time()
    
    # Load data
    df = load_data()
    
    # Create DeepChem datasets
    train_dataset, valid_dataset, test_dataset = create_deepchem_datasets(df)
    
    if train_dataset is None:
        print("❌ Failed to create datasets. Please install DeepChem and dependencies.")
        return
    
    # Train GraphConv model
    results = train_graphconv_model(train_dataset, valid_dataset, test_dataset)
    
    if results is None:
        print("❌ Training failed.")
        return
    
    # Save results
    total_time = time.time() - total_start
    results['total_time'] = total_time
    
    with open('graphconv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to graphconv_results.json")
    print(f"⏱️  Total execution time: {total_time:.1f}s")
    print("🎉 GraphConv training completed!")

if __name__ == "__main__":
    main()
