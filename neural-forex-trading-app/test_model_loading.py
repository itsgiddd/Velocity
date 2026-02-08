#!/usr/bin/env python3
"""
Test script to verify enhanced neural model loading
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_manager import NeuralModelManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading both enhanced and regular models"""
    
    print("TESTING NEURAL MODEL LOADING")
    print("=" * 50)
    
    # Create model manager
    model_manager = NeuralModelManager()
    
    # Test 1: Load enhanced model (should auto-detect)
    print("\nTest 1: Loading Enhanced Model")
    print("-" * 30)
    
    try:
        success = model_manager.load_model("enhanced_neural_model.pth")
        if success:
            print("SUCCESS: Enhanced model loaded successfully!")
            
            # Get model info
            info = model_manager.get_model_info()
            print(f"Model Size: {info.get('model_size_mb', 0):.2f} MB")
            print(f"Parameters: {info.get('total_parameters', 0):,}")
            print(f"Metadata: {info.get('metadata', {})}")
            
            # Test prediction
            import numpy as np
            test_features = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])  # 8 features for enhanced model
            prediction = model_manager.predict(test_features)
            if prediction:
                print(f"Test Prediction: {prediction}")
            else:
                print("ERROR: Prediction failed")
                
        else:
            print("ERROR: Enhanced model loading failed!")
            
    except Exception as e:
        print(f"ERROR: Enhanced model error: {e}")
    
    # Test 2: Load regular model
    print("\nTest 2: Loading Regular Model")
    print("-" * 30)
    
    # Create new manager for clean test
    model_manager2 = NeuralModelManager()
    
    try:
        success = model_manager2.load_model("neural_model.pth")
        if success:
            print("SUCCESS: Regular model loaded successfully!")
            
            # Get model info
            info = model_manager2.get_model_info()
            print(f"Model Size: {info.get('model_size_mb', 0):.2f} MB")
            print(f"Parameters: {info.get('total_parameters', 0):,}")
            print(f"Metadata: {info.get('metadata', {})}")
            
            # Test prediction
            test_features = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])  # 6 features for regular model
            prediction = model_manager2.predict(test_features)
            if prediction:
                print(f"Test Prediction: {prediction}")
            else:
                print("ERROR: Prediction failed")
                
        else:
            print("ERROR: Regular model loading failed!")
            
    except Exception as e:
        print(f"ERROR: Regular model error: {e}")
    
    print("\nMODEL LOADING TESTS COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    test_model_loading()
