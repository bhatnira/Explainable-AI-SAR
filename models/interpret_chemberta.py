#!/usr/bin/env python3
"""
🔍 ChemBERTa Model Interpretation
Generate attention visualizations for molecular predictions
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
import io
from transformers import AutoConfig, AutoModel
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from simpletransformers.classification import ClassificationModel
import warnings
warnings.filterwarnings('ignore')

# Set environment for single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
        
        print(f"   Tokens: {tokens}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Atom count: {mol.GetNumAtoms()}")
        
        # -------------------------
        # Step 5: Extract attention
        # -------------------------
        layer = -1  # Last layer
        head = 0    # First head
        att = attentions[layer][0, head].detach().cpu().numpy()
        
        atom_count = mol.GetNumAtoms()
        # Map tokens to atoms (skip CLS token at index 0)
        token_atom_indices = list(range(1, min(seq_len, 1 + atom_count)))
        
        if len(token_atom_indices) == 0:
            print("❌ No valid token-to-atom mapping found")
            return None
            
        cls_attention = att[0, token_atom_indices]
        
        # Normalize attention scores
        if np.sum(cls_attention) > 0:
            cls_attention /= np.sum(cls_attention)
        else:
            cls_attention = np.ones_like(cls_attention) / len(cls_attention)
        
        print(f"   Attention scores shape: {cls_attention.shape}")
        print(f"   Attention score range: {cls_attention.min():.4f} - {cls_attention.max():.4f}")
        
        # -------------------------
        # Step 6: Prepare color & radii
        # -------------------------
        norm = mcolors.Normalize(vmin=cls_attention.min(), vmax=cls_attention.max())
        cmap = cm.get_cmap('RdYlBu_r')
        
        # Map attention scores to atoms
        atom_weights = {}
        colors = {}
        radii = {}
        
        for i in range(len(token_atom_indices)):
            if i < len(cls_attention):
                weight = float(cls_attention[i])
                atom_weights[i] = weight
                colors[i] = tuple(cmap(norm(weight))[:3])
                radii[i] = 0.3 + 0.4 * weight  # Scale radius between 0.3 and 0.7
        
        # -------------------------
        # Step 7: Draw high-res molecule
        # -------------------------
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
        
        # Set drawing options
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().addStereoAnnotation = False
        
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(atom_weights.keys()),
            highlightAtomColors=colors,
            highlightAtomRadii=radii
        )
        drawer.FinishDrawing()
        mol_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        
        # -------------------------
        # Step 8: Create visualization with colorbar
        # -------------------------
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(mol_img)
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Score from [CLS] Token', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Attention visualization saved to '{output_file}'")
        return output_file
        
    except Exception as e:
        print(f"❌ Error generating attention visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to load model and generate interpretations"""
    print("🔍 ChemBERTa Model Interpretation")
    print("=" * 50)
    
    # Check if model exists
    model_path = "chemberta_output"
    if not os.path.exists(model_path):
        print(f"❌ Model directory '{model_path}' not found.")
        print("   Please run train_chemberta_simple.py first to train the model.")
        return
    
    print(f"📂 Loading trained model from: {model_path}")
    
    try:
        # Load the trained model
        model_args = {
            'eval_batch_size': 16,
            'use_cuda': torch.cuda.is_available(),
            'silent': True
        }
        
        model = ClassificationModel(
            "roberta",
            model_path,
            args=model_args,
            use_cuda=torch.cuda.is_available()
        )
        
        print("✅ Model loaded successfully!")
        
        # Single test molecule for interpretation (to avoid multiple images)
        test_smiles = "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC"
        
        print(f"\n🧪 Processing single molecule for interpretation")
        
        # Make prediction first
        prediction = model.predict([test_smiles])
        pred_class = prediction[0][0]
        confidence = max(prediction[1][0])
        
        print(f"   Prediction: Class {pred_class} (confidence: {confidence:.3f})")
        
        # Generate attention visualization
        output_file = "visualizations/chemberta_attention.png"
        os.makedirs("visualizations", exist_ok=True)
        interpret_chemberta_attention(model, test_smiles, output_file)
        
        print(f"\n🎉 Interpretation completed!")
        print(f"📁 Visualization saved as '{output_file}'")
        
    except Exception as e:
        print(f"❌ Error loading model or generating interpretations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
