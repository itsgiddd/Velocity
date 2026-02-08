#!/usr/bin/env python3
"""
Test the model manager directly
"""

import sys
sys.path.append('.')

from model_manager import NeuralModelManager
import numpy as np

def test_model_manager():
    """Test the model manager"""
    print("Testing ModelManager")
    print("=" * 30)
    
    try:
        # Create manager
        manager = NeuralModelManager()
        print(f"Manager created, feature_dim: {manager.feature_dim}")
        
        # Load model
        success = manager.load_model()
        print(f"Model loading result: {success}")
        
        if success:
            print("Model loaded successfully!")
            print(f"Metadata: {manager.model_metadata}")
            
            # Test prediction with 6 features
            test_features = np.random.randn(1, 6)
            prediction = manager.predict(test_features)
            
            if prediction:
                print(f"Prediction successful!")
                print(f"Action: {prediction['action']}")
                print(f"Confidence: {prediction['confidence']:.1%}")
                print(f"Probabilities: {prediction['probabilities']}")
                return True
            else:
                print("Prediction failed")
                return False
        else:
            print("Model loading failed")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_manager()
    if success:
        print("\nSUCCESS: ModelManager works!")
    else:
        print("\nFAILED: ModelManager has issues")