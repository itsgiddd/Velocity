#!/usr/bin/env python3
"""
Test the actual working architecture
"""

import torch
import torch.nn as nn
import numpy as np

class WorkingNeuralNetwork(nn.Module):
    """Neural network matching the saved models"""
    
    def __init__(self, input_dim: int = 6, output_size: int = 3):
        super(WorkingNeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

def test_neural_model():
    """Test neural model with working architecture"""
    print("Testing neural_model.pth with working architecture")
    
    try:
        # Load model
        checkpoint = torch.load("neural_model.pth", map_location='cpu', weights_only=False)
        print("Model loaded successfully")
        
        # Create working model
        model = WorkingNeuralNetwork(input_dim=6)
        
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
    print("Working Model Test")
    print("=" * 30)
    
    success = test_neural_model()
    
    if success:
        print("\nSUCCESS: Neural model works with 6 features!")
        return True
    else:
        print("\nFAILED: Model still has issues")
        return False

if __name__ == "__main__":
    main()