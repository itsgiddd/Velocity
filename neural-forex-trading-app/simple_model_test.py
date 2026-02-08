#!/usr/bin/env python3
"""
Simple model loading test
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

class EnhancedNeuralNetwork(nn.Module):
    """Enhanced neural network for forex prediction with candlestick patterns"""
    
    def __init__(self, input_dim: int = 10, hidden_sizes: List[int] = [512, 256, 128, 64], output_size: int = 3):
        super(EnhancedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers (larger for 10 features + candlestick patterns)
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def test_enhanced_model():
    """Test the enhanced model"""
    print("Testing enhanced_neural_model.pth")
    
    try:
        # Load model
        checkpoint = torch.load("enhanced_neural_model.pth", map_location='cpu', weights_only=False)
        print("Model loaded successfully")
        
        # Create enhanced model (10 features)
        model = EnhancedNeuralNetwork(input_dim=10)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("Model architecture loaded")
        
        # Test prediction
        test_features = np.random.randn(1, 10)
        with torch.no_grad():
            outputs = model(torch.FloatTensor(test_features))
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            
            classes = ['SELL', 'HOLD', 'BUY']
            print(f"Prediction: {classes[predicted_class]}")
            print(f"Probabilities: {probabilities.numpy()[0]}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_neural_model():
    """Test the old neural model"""
    print("\nTesting neural_model.pth")
    
    try:
        # Load model
        checkpoint = torch.load("neural_model.pth", map_location='cpu', weights_only=False)
        print("Model loaded successfully")
        
        # Create enhanced model (6 features)
        model = EnhancedNeuralNetwork(input_dim=6, hidden_sizes=[256, 128, 64])
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("Model architecture loaded")
        
        # Test prediction
        test_features = np.random.randn(1, 6)
        with torch.no_grad():
            outputs = model(torch.FloatTensor(test_features))
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            
            classes = ['SELL', 'HOLD', 'BUY']
            print(f"Prediction: {classes[predicted_class]}")
            print(f"Probabilities: {probabilities.numpy()[0]}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("Model Loading Test")
    print("=" * 30)
    
    enhanced_ok = test_enhanced_model()
    neural_ok = test_neural_model()
    
    print("\nResults:")
    print(f"Enhanced model: {'OK' if enhanced_ok else 'FAILED'}")
    print(f"Neural model: {'OK' if neural_ok else 'FAILED'}")
    
    if enhanced_ok:
        print("\nUsing enhanced model (10 features)")
        return "enhanced"
    elif neural_ok:
        print("\nUsing neural model (6 features)")
        return "neural"
    else:
        print("\nBoth models failed")
        return None

if __name__ == "__main__":
    main()