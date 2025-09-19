#!/usr/bin/env python
"""
Simple Graph Convolutional Network for Molecular Classification
Author: AI Assistant
Date: 2025
Description: Train a GraphConv model using DeepChem for molecular property prediction
             with explainability features similar to the existing models
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import deepchem with specific modules to avoid DGL issues
import deepchem.feat as dc_feat
import deepchem.data as dc_data
import deepchem.models as dc_models
import deepchem.metrics as dc_metrics
from rdkit import Chem

def load_molecular_data():
    """Load and preprocess molecular data"""
    print("📂 Loading molecular data...")
    
    # Load the Excel file
    df = pd.read_excel('data/StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx')
    
    print(f"✅ Loaded {len(df)} molecules")
    
    # Basic data validation
    df = df.dropna(subset=['Smiles', 'classLabel'])
    
    # Print dataset summary
    print(f"📊 Dataset summary:")
    print(f"   Total molecules: {len(df)}")
    print(f"   Active molecules: {sum(df['classLabel'] == 1)}")
    print(f"   Inactive molecules: {sum(df['classLabel'] == 0)}")
    
    return df

def create_molecular_datasets(df):
    """Create training and test datasets"""
    print("🧪 Creating molecular datasets...")
    
    # Split data
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['classLabel']
    )
    
    print("   Creating training dataset...")
    train_featurizer = dc_feat.ConvMolFeaturizer()
    train_features = train_featurizer.featurize(train_df['Smiles'].values)
    train_dataset = dc_data.NumpyDataset(
        X=train_features,
        y=train_df['classLabel'].values.reshape(-1, 1),
        ids=train_df['Smiles'].values
    )
    
    print("   Creating test dataset...")
    test_featurizer = dc_feat.ConvMolFeaturizer()
    test_features = test_featurizer.featurize(test_df['Smiles'].values)
    test_dataset = dc_data.NumpyDataset(
        X=test_features,
        y=test_df['classLabel'].values.reshape(-1, 1),
        ids=test_df['Smiles'].values
    )
    
    print(f"📊 Dataset splits:")
    print(f"   Training: {len(train_dataset)} molecules")
    print(f"   Test: {len(test_dataset)} molecules")
    
    return train_dataset, test_dataset, train_featurizer

def create_graphconv_model():
    """Create and configure GraphConvModel"""
    print("🤖 Creating GraphConv model...")
    
    # GraphConv model configuration
    model = dc_models.GraphConvModel(
        n_tasks=1,
        mode='classification',
        n_classes=2,
        graph_conv_layers=[64, 64],
        dense_layer_size=128,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=64,
        model_dir='models/graphconv_simple'
    )
    
    return model

def train_model(model, train_dataset, test_dataset):
    """Train the GraphConv model"""
    print("🚀 Training GraphConv model...")
    
    # Set up validation callback
    validation_callback = dc_models.ValidationCallback(
        test_dataset, 
        interval=5,
        metrics=[dc_metrics.Metric(dc_metrics.roc_auc_score)]
    )
    
    # Train the model
    model.fit(
        train_dataset, 
        nb_epoch=50,
        callbacks=[validation_callback]
    )
    
    return model

def evaluate_model(model, train_dataset, test_dataset):
    """Evaluate model performance"""
    print("📊 Evaluating model...")
    
    # Define metrics
    metric = dc_metrics.Metric(dc_metrics.roc_auc_score)
    
    # Evaluate on training set
    train_predictions = model.predict(train_dataset)
    train_score = metric.compute_metric(train_dataset.y, train_predictions)
    print(f"   Training ROC-AUC: {train_score:.4f}")
    
    # Evaluate on test set
    test_predictions = model.predict(test_dataset)
    test_score = metric.compute_metric(test_dataset.y, test_predictions)
    print(f"   Test ROC-AUC: {test_score:.4f}")
    
    # Debug prediction shapes
    print(f"Train predictions shape: {train_predictions.shape}")
    print(f"Test predictions shape: {test_predictions.shape}")
    print(f"Train labels shape: {train_dataset.y.shape}")
    print(f"Test labels shape: {test_dataset.y.shape}")
    
    # Convert predictions to binary
    # Handle multi-dimensional outputs
    if len(train_predictions.shape) == 3:  # Shape: (samples, 1, 2)
        # Take the probability of the positive class
        train_pred_probs = train_predictions[:, 0, 1]  # Probability of class 1
        test_pred_probs = test_predictions[:, 0, 1]
        train_pred_binary = (train_pred_probs > 0.5).astype(int)
        test_pred_binary = (test_pred_probs > 0.5).astype(int)
    elif len(train_predictions.shape) == 2 and train_predictions.shape[1] == 2:
        # Shape: (samples, 2) - softmax output
        train_pred_binary = (train_predictions[:, 1] > 0.5).astype(int)
        test_pred_binary = (test_predictions[:, 1] > 0.5).astype(int)
    elif len(train_predictions.shape) == 2 and train_predictions.shape[1] == 1:
        # Shape: (samples, 1) - single probability
        train_pred_binary = (train_predictions[:, 0] > 0.5).astype(int)
        test_pred_binary = (test_predictions[:, 0] > 0.5).astype(int)
    else:
        # Single output (probability)
        train_pred_binary = (train_predictions > 0.5).astype(int).flatten()
        test_pred_binary = (test_predictions > 0.5).astype(int).flatten()
    
    # Detailed classification report for test set
    print("\n📋 Test Set Classification Report:")
    y_true = test_dataset.y.flatten()
    y_pred = test_pred_binary
    
    print(f"y_true shape: {y_true.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    
    report = classification_report(
        y_true, 
        y_pred,
        target_names=['Inactive', 'Active']
    )
    print(report)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Inactive', 'Active'],
                yticklabels=['Inactive', 'Active'])
    plt.title('GraphConv Model - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/graphconv_simple_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'train_roc_auc': train_score,
        'test_roc_auc': test_score,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions
    }

def save_results(model, results, featurizer):
    """Save model and results"""
    print("💾 Saving model and results...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save model
    model.save_checkpoint(model_dir='models/graphconv_simple')
    
    # Save results
    results_df = pd.DataFrame({
        'metric': ['train_roc_auc', 'test_roc_auc'],
        'value': [results['train_roc_auc'], results['test_roc_auc']]
    })
    results_df.to_csv('results/graphconv_simple_results.csv', index=False)
    
    print("✅ Model and results saved!")
    print(f"   Model saved to: models/graphconv_simple")
    print(f"   Results saved to: results/graphconv_simple_results.csv")

def main():
    """Main training pipeline"""
    print("🔬 GraphConv Model Training Pipeline")
    print("=" * 50)
    
    try:
        # Load data
        df = load_molecular_data()
        
        # Create datasets
        train_dataset, test_dataset, featurizer = create_molecular_datasets(df)
        
        # Create model
        model = create_graphconv_model()
        
        # Train model
        trained_model = train_model(model, train_dataset, test_dataset)
        
        # Evaluate model
        results = evaluate_model(trained_model, train_dataset, test_dataset)
        
        # Save results
        save_results(trained_model, results, featurizer)
        
        print("\n🎉 Training completed successfully!")
        print(f"Final Test ROC-AUC: {results['test_roc_auc']:.4f}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
