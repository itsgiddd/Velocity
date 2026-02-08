#!/usr/bin/env python3
"""
Detailed model loading diagnostic script
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

def test_neural_model():
    """Test the old neural model"""
    print("Testing neural_model.pth (6-feature model)")
    print("=" * 50)
    
    try:
        # Load the model
        checkpoint = torch.load("neural_model.pth", map_location='cpu', weights_only=False)
        print(f"✓ Model loaded successfully")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Create the architecture
        model = nn.Sequential(
            nn.Linear(6, 128),  # 6 features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # SELL, HOLD, BUY
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✓ Model architecture loaded successfully")
        
        # Test prediction with 6 features
        test_features = np.random.randn(1, 6)
        with torch.no_grad():
            outputs = model(torch.FloatTensor(test_features))
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            
            classes = ['SELL', 'HOLD', 'BUY']
            print(f"✓ Test prediction successful")
            print(f"Predicted class: {classes[predicted_class]}")
            print(f"Probabilities: {probabilities.numpy()[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_enhanced_neural_model():
    """Test the enhanced neural model"""
    print("\nTesting enhanced_neural_model.pth (10-feature model)")
    print("=" * 50)
    
    try:
        # Load the model
        checkpoint = torch.load("enhanced_neural_model.pth", map_location='cpu', weights_only=False)
        print(f"✓ Model loaded successfully")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check metadata
        if 'metadata' in checkpoint:
            print(f"Metadata: {checkpoint['metadata']}")
        
        # Create the architecture  
        model = nn.Sequential(
            nn.Linear(10, 128),  # 10 features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # SELL, HOLD, BUY
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✓ Model architecture loaded successfully")
        
        # Test prediction with 10 features
        test_features = np.random.randn(1, 10)
        with torch.no_grad():
            outputs = model(torch.FloatTensor(test_features))
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            
            classes = ['SELL', 'HOLD', 'BUY']
            print(f"✓ Test prediction successful")
            print(f"Predicted class: {classes[predicted_class]}")
            print(f"Probabilities: {probabilities.numpy()[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_model_manager():
    """Test the current model manager"""
    print("\nTesting current ModelManager")
    print("=" * 50)
    
    try:
        # Import the model manager
        from model_manager import NeuralModelManager
        
        # Create manager
        manager = NeuralModelManager()
        print(f"✓ ModelManager created")
        print(f"Feature dimension: {manager.feature_dim}")
        
        # Try to load model
        success = manager.load_model()
        print(f"Model loading result: {success}")
        
        if success:
            print(f"✓ Model loaded successfully")
            print(f"Model metadata: {manager.model_metadata}")
            
            # Test prediction
            test_features = np.random.randn(1, 10)
            prediction = manager.predict(test_features)
            
            if prediction:
                print(f"✓ Prediction successful")
                print(f"Action: {prediction['action']}")
                print(f"Confidence: {prediction['confidence']}")
            else:
                print(f"✗ Prediction failed")
        else:
            print(f"✗ Model loading failed")
        
        return success
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Neural Model Loading Diagnostic")
    print("=" * 60)
    
    # Test both models
    neural_success = test_neural_model()
    enhanced_success = test_enhanced_neural_model()
    manager_success = test_model_manager()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Neural model (6-feature): {'✓' if neural_success else '✗'}")
    print(f"Enhanced model (10-feature): {'✓' if enhanced_success else '✗'}")
    print(f"ModelManager: {'✓' if manager_success else '✗'}")
    
    if neural_success and enhanced_success and manager_success:
        print("\n🎉 All tests passed! Models should work correctly.")
    elif enhanced_success:
        print("\n⚠️ Enhanced model works, but ModelManager has issues.")
    elif neural_success:
        print("\n⚠️ Neural model works, but Enhanced model or ModelManager has issues.")
    else:
        print("\n❌ Major issues detected. Check model files and architecture.")

if __name__ == "__main__":
    main()