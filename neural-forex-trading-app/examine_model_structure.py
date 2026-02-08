#!/usr/bin/env python3
"""
Examine saved model structure
"""

import torch
import torch.nn as nn
import numpy as np

def examine_model_structure(model_path, model_name):
    """Examine the structure of a saved model"""
    print(f"\nExamining {model_name}")
    print("=" * 50)
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"Model loaded successfully")
        
        # Check what's in the checkpoint
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Look at state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        print(f"State dict keys: {list(state_dict.keys())}")
        
        # Analyze the first layer to understand input size
        first_key = list(state_dict.keys())[0]
        first_weight = state_dict[first_key]
        print(f"First layer ({first_key}): {first_weight.shape}")
        
        # Look for architecture hints
        if 'metadata' in checkpoint:
            print(f"Metadata: {checkpoint['metadata']}")
            
        # Try to determine the architecture by looking at layer names
        linear_layers = [key for key in state_dict.keys() if 'weight' in key and 'Linear' in key or key.startswith('network.')]
        
        print(f"Linear layer weights found: {len(linear_layers)}")
        for key in linear_layers:
            weight = state_dict[key]
            print(f"  {key}: {weight.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error examining {model_name}: {e}")
        return False

def main():
    print("Model Structure Examination")
    print("=" * 50)
    
    # Examine both models
    enhanced_ok = examine_model_structure("enhanced_neural_model.pth", "Enhanced Model")
    neural_ok = examine_model_structure("neural_model.pth", "Neural Model")
    
    print(f"\nSummary:")
    print(f"Enhanced model: {'OK' if enhanced_ok else 'FAILED'}")
    print(f"Neural model: {'OK' if neural_ok else 'FAILED'}")

if __name__ == "__main__":
    main()