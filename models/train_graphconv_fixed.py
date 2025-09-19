#!/usr/bin/env python3
"""
🧪 Fixed GraphConv Model Training
Clean implementation for molecular classification using DeepChem
"""

import os
import pandas as pd
import time
import json

# Set environment for single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def train_graphconv_model(df):
    """Train GraphConv model"""
    print("🧪 Training GraphConv model...")
    
    try:
        import deepchem as dc
        from deepchem.feat import ConvMolFeaturizer
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow.keras import backend as K
        
        # Clear Keras session
        K.clear_session()
        
        # Define the ConvMolFeaturizer for graph representation
        featurizer = ConvMolFeaturizer()
        print("   Using ConvMolFeaturizer for molecular graphs")
        
        # Featurize the SMILES
        print("   Featurizing SMILES strings...")
        features = featurizer.featurize(df['cleanedMol'].tolist())
        targets = df['classLabel'].values
        
        # Create train/test split manually
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42, stratify=targets
        )
        
        # Create DeepChem datasets
        train_dataset = dc.data.NumpyDataset(X_train, y_train)
        test_dataset = dc.data.NumpyDataset(X_test, y_test)
        
        print(f"📊 Dataset splits:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Test: {len(test_dataset)} samples")
        
        # Model configuration
        print("🔧 Model Configuration:")
        print("   Model: DeepChem GraphConvModel")
        print("   Batch size: 32")
        print("   Dropout: 0.01")
        print("   Epochs: 30")
        print("   Mode: classification")
        
        # Create GraphConv model
        n_tasks = 1
        model = dc.models.GraphConvModel(
            n_tasks, 
            batch_size=32,
            dropout=0.01,
            mode='classification'
        )
        
        print("🏋️ Starting training...")
        start_time = time.time()
        
        # Train model
        model.fit(train_dataset, nb_epoch=30)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        
        # Evaluate model
        print("🔬 Evaluating model...")
        
        # Get predictions
        test_predictions = model.predict(test_dataset)
        
        # Debug shapes
        print(f"   Test predictions shape: {test_predictions.shape}")
        print(f"   Test true labels shape: {test_dataset.y.shape}")
        
        # Handle prediction shape - get first column for binary classification
        if len(test_predictions.shape) > 1 and test_predictions.shape[1] > 1:
            # Multi-class output, take second column (class 1 probabilities)
            test_pred_probs = test_predictions[:, 1]
        else:
            # Single output
            test_pred_probs = test_predictions.flatten()
        
        # Calculate metrics for test set
        test_pred_classes = (test_pred_probs > 0.5).astype(int)
        test_true = test_dataset.y.flatten()
        
        # Ensure same length
        min_len = min(len(test_true), len(test_pred_classes))
        test_true = test_true[:min_len]
        test_pred_classes = test_pred_classes[:min_len]
        test_pred_probs = test_pred_probs[:min_len]
        
        accuracy = accuracy_score(test_true, test_pred_classes)
        roc_auc = roc_auc_score(test_true, test_pred_probs)
        
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
    
    # Train GraphConv model
    results = train_graphconv_model(df)
    
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
