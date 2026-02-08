#!/usr/bin/env python3
"""
Debug Model Loading Script
==========================
Tests model loading and identifies issues with the neural model.
"""

import sys
import os
import torch
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

def test_model_files():
    """Test if model files exist and are readable"""
    print("=== MODEL FILE CHECK ===")
    
    model_files = [
        "enhanced_neural_model.pth",
        "neural_model.pth",
        "models/best_model.pth",
        "models/current_model.pth"
    ]
    
    for model_file in model_files:
        path = Path(model_file)
        if path.exists():
            size = path.stat().st_size
            print(f"[OK] {model_file} exists ({size:,} bytes)")
        else:
            print(f"[MISSING] {model_file} does not exist")
    
    print()

def test_torch_load():
    """Test loading with torch directly"""
    print("=== TORCH LOAD TEST ===")
    
    model_files = [
        "enhanced_neural_model.pth",
        "neural_model.pth"
    ]
    
    for model_file in model_files:
        if not Path(model_file).exists():
            continue
            
        print(f"Testing {model_file}...")
        try:
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            print(f"[OK] {model_file} loaded successfully")
            
            # Check checkpoint contents
            print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
            
            if 'model_config' in checkpoint:
                print(f"   Model config: {checkpoint['model_config']}")
            if 'model_state_dict' in checkpoint:
                print(f"   State dict keys: {list(checkpoint['model_state_dict'].keys())[:5]}...")
            if 'scaler' in checkpoint:
                print(f"   Has scaler: True")
                
        except Exception as e:
            print(f"[ERROR] {model_file} failed to load: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
        
        print()

def test_model_manager():
    """Test using NeuralModelManager"""
    print("=== MODEL MANAGER TEST ===")
    
    try:
        from model_manager import NeuralModelManager
        
        model_manager = NeuralModelManager()
        print(f"Model manager created. Feature dim: {model_manager.feature_dim}")
        
        # Try to load model
        print("Attempting to load model...")
        success = model_manager.load_model()
        
        if success:
            print("[OK] Model loaded successfully!")
            info = model_manager.get_model_info()
            print(f"Model info: {info}")
        else:
            print("[ERROR] Model loading failed!")
            
        return success
        
    except Exception as e:
        print(f"[ERROR] Model manager test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_enhanced_model():
    """Test creating and loading enhanced model"""
    print("=== ENHANCED MODEL TEST ===")
    
    try:
        from model_manager import EnhancedNeuralNetwork
        
        # Create enhanced model
        print("Creating EnhancedNeuralNetwork...")
        model = EnhancedNeuralNetwork(input_dim=10)
        print(f"[OK] Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Try loading weights from enhanced_neural_model.pth
        if Path("enhanced_neural_model.pth").exists():
            print("Loading weights from enhanced_neural_model.pth...")
            checkpoint = torch.load("enhanced_neural_model.pth", map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("[OK] Weights loaded successfully")
            else:
                print("[ERROR] No model_state_dict in checkpoint")
        else:
            print("enhanced_neural_model.pth not found")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Enhanced model test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    print("NEURAL MODEL LOADING DEBUG")
    print("=" * 50)
    
    # Change to neural-forex-trading-app directory
    if not Path("model_manager.py").exists():
        print("Error: Not in neural-forex-trading-app directory")
        return False
    
    # Run tests
    test_model_files()
    test_torch_load()
    test_model_manager()
    test_enhanced_model()
    
    print("=== SUMMARY ===")
    print("Debug completed. Check results above for specific issues.")

if __name__ == "__main__":
    main()