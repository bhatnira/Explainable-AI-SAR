#!/usr/bin/env python3
"""
🧬 Simple ChemBERTa Model Training
Clean implementation for molecular classification
"""

import os
import pandas as pd
import time
import json
from transformers import AutoConfig, AutoModel
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
import io

# Set environment for single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def load_data(data_file='StandarizedSmiles_cutOFF800daltonMolecularweight.xlsx', sample_size=100):
    """Load molecular data with sampling"""
    print(f"📂 Loading data from: {data_file}")
    
    df = pd.read_excel(data_file)
    print(f"✅ Loaded {len(df)} molecules")
    
    # Clean data
    df_clean = df[['cleanedMol', 'classLabel']].dropna()
    df_clean.columns = ['text', 'labels']  # Rename for SimpleTransformers
    
    # Sample data to specified size
    if sample_size and sample_size < len(df_clean):
        # Balance classes in sample
        df_class_0 = df_clean[df_clean['labels'] == 0].sample(n=sample_size//2, random_state=42)
        df_class_1 = df_clean[df_clean['labels'] == 1].sample(n=min(sample_size//2, len(df_clean[df_clean['labels'] == 1])), random_state=42)
        df_clean = pd.concat([df_class_0, df_class_1], ignore_index=True)
        df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"📊 Sampled to {len(df_clean)} molecules")
    
    print(f"📊 Final samples: {len(df_clean)}")
    print(f"📊 Class distribution: {df_clean['labels'].value_counts().to_dict()}")
    
    return df_clean

def create_train_test_split(df, test_size=0.2):
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df['labels']
    )
    
    print(f"📊 Train samples: {len(train_df)}")
    print(f"📊 Test samples: {len(test_df)}")
    
    return train_df, test_df

def train_chemberta(train_df, test_df):
    """Train ChemBERTa model"""
    print("🧬 Training ChemBERTa model...")
    
    from simpletransformers.classification import ClassificationModel
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    
    # Model configuration
    model_args = {
        'overwrite_output_dir': True,
        'num_train_epochs': 3,
        'train_batch_size': 8,
        'eval_batch_size': 8,
        'learning_rate': 2e-5,
        'max_seq_length': 512,
        'silent': False,
        'use_cuda': False,
        'fp16': False,
        'output_dir': './chemberta_output',
        'best_model_dir': './chemberta_output/best_model',
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 100,
        'save_model_every_epoch': False,
        'save_steps': -1,
    }
    
    print("🔧 Model Configuration:")
    print(f"   Model: DeepChem/ChemBERTa-77M-MLM")
    print(f"   Epochs: {model_args['num_train_epochs']}")
    print(f"   Batch size: {model_args['train_batch_size']}")
    print(f"   Learning rate: {model_args['learning_rate']}")
    print(f"   Max sequence length: {model_args['max_seq_length']}")
    
    # Create model
    model = ClassificationModel(
        'roberta',
        'DeepChem/ChemBERTa-77M-MLM',
        num_labels=2,
        args=model_args,
        use_cuda=False
    )
    
    # Train model
    print("🏋️ Starting training...")
    start_time = time.time()
    
    model.train_model(train_df, eval_df=test_df)
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Evaluate model
    print("🔬 Evaluating model...")
    predictions, raw_outputs = model.predict(test_df['text'].tolist())
    probabilities = raw_outputs[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(test_df['labels'], predictions)
    roc_auc = roc_auc_score(test_df['labels'], probabilities)
    
    print(f"\n🎯 Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    print(f"   Training time: {training_time:.1f}s")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(test_df['labels'], predictions))
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'training_time': training_time,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }

def interpret_chemberta_attention(model, smiles, output_file="chemberta_attention.png"):
    """
    Generate attention visualization for a ChemBERTa model prediction
    
    Args:
        model: Trained ChemBERTa model
        smiles: SMILES string to visualize
        output_file: Output filename for the visualization
    """
    print(f"🔍 Generating attention visualization for: {smiles}")
    
    try:
        # Create molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ Invalid SMILES: {smiles}")
            return None
        
        # -------------------------
        # Step 1: Config for attention
        # -------------------------
        model_name_or_path = model.args.model_name
        
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.attn_implementation = "eager"
        config.output_attentions = True
        
        # -------------------------
        # Step 2: Reload RobertaModel
        # -------------------------
        hf_model = AutoModel.from_pretrained(model_name_or_path, config=config)
        
        # -------------------------
        # Step 3: Fix state_dict keys
        # -------------------------
        state_dict = model.model.state_dict()
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith("roberta."):
                new_state_dict[k[len("roberta."):]] = v
            else:
                new_state_dict[k] = v
        
        hf_model.load_state_dict(new_state_dict, strict=False)
        hf_model.eval()
        
        # -------------------------
        # Step 4: Tokenize & run model
        # -------------------------
        tokenizer = model.tokenizer
        inputs = tokenizer(smiles, return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = hf_model(**inputs)
        
        attentions = outputs.attentions
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        seq_len = len(tokens)
        
        # -------------------------
        # Step 5: Extract attention
        # -------------------------
        layer = -1  # Last layer
        head = 0    # First head
        att = attentions[layer][0, head].detach().cpu().numpy()
        
        atom_count = mol.GetNumAtoms()
        token_atom_indices = list(range(1, 1 + atom_count))
        
        if max(token_atom_indices) >= seq_len:
            print(f"⚠️  Token-to-atom mapping out of bounds. Using available tokens.")
            token_atom_indices = list(range(1, min(seq_len, 1 + atom_count)))
            atom_count = len(token_atom_indices)
        
        cls_attention = att[0, token_atom_indices]
        cls_attention /= np.sum(cls_attention)
        
        # -------------------------
        # Step 6: Prepare color & radii
        # -------------------------
        norm = mcolors.Normalize(vmin=min(cls_attention), vmax=max(cls_attention))
        cmap = cm.get_cmap('RdYlBu_r')  # Compatible with older matplotlib versions
        
        atom_weights = {i: float(cls_attention[i]) for i in range(atom_count)}
        colors = {i: tuple(cmap(norm(w))[:3]) for i, w in atom_weights.items()}
        radii = {i: 0.6 * atom_weights[i] for i in atom_weights}
        
        # -------------------------
        # Step 7: Draw high-res molecule
        # -------------------------
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)  # High resolution 800x800 px
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(atom_weights.keys()),
            highlightAtomColors=colors,
            highlightAtomRadii=radii
        )
        drawer.FinishDrawing()
        mol_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        
        # -------------------------
        # Step 8: Combine with colorbar legend
        # -------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(mol_img)
        ax.axis('off')
        ax.set_title(f'ChemBERTa Attention Visualization\nSMILES: {smiles}', 
                    fontsize=14, pad=20)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Score from [CLS] Token', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        print(f"✅ Attention visualization saved to '{output_file}'")
        return output_file
        
    except Exception as e:
        print(f"❌ Error generating attention visualization: {e}")
        return None

def main():
    print("🧬 ChemBERTa Molecular Classification")
    print("=" * 50)
    
    total_start = time.time()
    
    # Load data
    df = load_data()
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df)
    
    # Train ChemBERTa
    results = train_chemberta(train_df, test_df)
    
    # Load the trained model for interpretation
    print("\n🔍 Loading model for interpretation...")
    try:
        from simpletransformers.classification import ClassificationModel
        
        model_args = {
            'eval_batch_size': 16,
            'use_cuda': torch.cuda.is_available(),
            'silent': True
        }
        
        model = ClassificationModel(
            "roberta",
            "chemberta_output",
            args=model_args,
            use_cuda=torch.cuda.is_available()
        )
        
        # Generate attention visualization for a sample molecule
        sample_smiles = test_df.iloc[0]['text']  # First test molecule
        print(f"🎯 Sample molecule for visualization: {sample_smiles}")
        
        interpret_chemberta_attention(
            model, 
            sample_smiles, 
            "chemberta_attention_visualization.png"
        )
        
        # Also try the example molecule from the user
        example_smiles = "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC"
        print(f"🎯 Example molecule for visualization: {example_smiles}")
        
        interpret_chemberta_attention(
            model, 
            example_smiles, 
            "chemberta_attention_example.png"
        )
        
    except Exception as e:
        print(f"⚠️  Could not generate attention visualization: {e}")
    
    # Save results
    total_time = time.time() - total_start
    results['total_time'] = total_time
    
    with open('chemberta_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to chemberta_results.json")
    print(f"⏱️  Total execution time: {total_time:.1f}s")
    print("🎉 ChemBERTa training completed!")

if __name__ == "__main__":
    main()
