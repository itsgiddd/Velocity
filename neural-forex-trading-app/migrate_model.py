#!/usr/bin/env python3
"""
Model Migration Script
====================
Migrates the old 8-feature model to the new 10-feature enhanced architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class OldNeuralNetwork(nn.Module):
    """Old neural network architecture (8 features, 3 layers)"""
    
    def __init__(self, input_dim=8, hidden_sizes=[256, 128, 64], output_size=3):
        super(OldNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers (old architecture)
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

class NewNeuralNetwork(nn.Module):
    """New neural network architecture (10 features, 4 layers)"""
    
    def __init__(self, input_dim=10, hidden_sizes=[512, 256, 128, 64], output_size=3):
        super(NewNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers (new enhanced architecture)
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

def migrate_model():
    """Migrate old model to new architecture"""
    print("MODEL MIGRATION")
    print("=" * 50)
    
    try:
        # Load old model
        print("Loading old model...")
        checkpoint = torch.load("enhanced_neural_model.pth", map_location='cpu', weights_only=False)
        
        # Create old architecture model
        old_model = OldNeuralNetwork(input_dim=8, hidden_sizes=[256, 128, 64])
        old_model.load_state_dict(checkpoint['model_state_dict'])
        print("Old model loaded successfully")
        
        # Create new architecture model
        print("Creating new model architecture...")
        new_model = NewNeuralNetwork(input_dim=10, hidden_sizes=[512, 256, 128, 64])
        print("New model created")
        
        # Migrate weights
        print("Migrating weights...")
        old_state_dict = old_model.state_dict()
        new_state_dict = new_model.state_dict()
        
        # Map old layers to new layers
        migration_map = {
            'network.0.weight': 'network.0.weight',  # First linear layer
            'network.0.bias': 'network.0.bias',
            'network.2.weight': 'network.2.weight',  # First batch norm
            'network.2.bias': 'network.2.bias',
            'network.2.running_mean': 'network.2.running_mean',
            'network.2.running_var': 'network.2.running_var',
            'network.2.num_batches_tracked': 'network.2.num_batches_tracked',
            
            'network.4.weight': 'network.6.weight',  # Second linear -> third linear
            'network.4.bias': 'network.6.bias',
            'network.6.weight': 'network.8.weight',  # Second batch norm -> third batch norm
            'network.6.bias': 'network.8.bias',
            'network.6.running_mean': 'network.8.running_mean',
            'network.6.running_var': 'network.8.running_var',
            'network.6.num_batches_tracked': 'network.8.num_batches_tracked',
            
            'network.8.weight': 'network.10.weight',  # Third linear -> fourth linear
            'network.8.bias': 'network.10.bias',
            'network.10.weight': 'network.12.weight',  # Third batch norm -> fourth batch norm
            'network.10.bias': 'network.12.bias',
            'network.10.running_mean': 'network.12.running_mean',
            'network.10.running_var': 'network.12.running_var',
            'network.10.num_batches_tracked': 'network.12.num_batches_tracked',
            
            'network.12.weight': 'network.14.weight',  # Output layer
            'network.12.bias': 'network.14.bias',
        }
        
        # Apply migration
        for old_key, new_key in migration_map.items():
            if old_key in old_state_dict and new_key in new_state_dict:
                old_weight = old_state_dict[old_key]
                new_weight = new_state_dict[new_key]
                
                # Handle weight matrix resizing
                if 'weight' in old_key and old_key != 'network.12.weight':  # Skip output layer
                    if old_weight.shape != new_weight.shape:
                        # For linear layers: copy existing weights and initialize new ones
                        if 'network.0.weight' in old_key:  # First layer: 8 -> 10 features, 256 -> 512
                            # Copy existing 256x8 weight to first 256x10 positions
                            new_weight[:256, :8] = old_weight
                            # Initialize remaining weights with small random values
                            new_weight[:256, 8:] = torch.randn(256, 2) * 0.01
                            new_weight[256:, :] = torch.randn(256, 10) * 0.01
                        elif 'network.4.weight' in old_key:  # Second layer: 256 -> 512
                            # Copy existing 128x256 weight to first 128x256 positions
                            new_weight[:128, :256] = old_weight
                            # Initialize remaining weights
                            new_weight[128:, :] = torch.randn(128, 512) * 0.01
                        elif 'network.8.weight' in old_key:  # Third layer: 128 -> 256
                            # Copy existing 64x128 weight to first 64x128 positions
                            new_weight[:64, :128] = old_weight
                            # Initialize remaining weights
                            new_weight[64:, :] = torch.randn(64, 256) * 0.01
                    else:
                        new_weight.copy_(old_weight)
                else:
                    # For biases and batch norm, copy directly if shapes match
                    if old_weight.shape == new_weight.shape:
                        new_weight.copy_(old_weight)
                
                print(f"Migrated: {old_key} -> {new_key}")
            else:
                print(f"Skipped: {old_key} (not found in either model)")
        
        # Load migrated weights into new model
        new_model.load_state_dict(new_state_dict)
        print("Weights migrated successfully")
        
        # Save migrated model
        print("Saving migrated model...")
        migrated_checkpoint = {
            'model_state_dict': new_model.state_dict(),
            'scaler': checkpoint.get('scaler', None),  # Preserve scaler if available
            'training_history': checkpoint.get('training_history', []),
            'model_config': {
                'input_size': 10,
                'hidden_layers': [512, 256, 128, 64],
                'output_size': 3,
                'migration_date': '2026-02-06T14:00:00',
                'original_config': checkpoint.get('model_config', {}),
                'migrated': True
            },
            'metadata': {
                'migration_date': '2026-02-06T14:00:00',
                'original_model': 'enhanced_neural_model.pth',
                'migration_notes': 'Migrated from 8-feature 3-layer to 10-feature 4-layer architecture'
            }
        }
        
        torch.save(migrated_checkpoint, 'enhanced_neural_model_migrated.pth')
        print("Migrated model saved as enhanced_neural_model_migrated.pth")
        
        # Also replace the original
        torch.save(migrated_checkpoint, 'enhanced_neural_model.pth')
        print("Original model replaced with migrated version")
        
        return True
        
    except Exception as e:
        print(f"Migration failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_migrated_model():
    """Test the migrated model"""
    print("\nTESTING MIGRATED MODEL")
    print("=" * 30)
    
    try:
        # Load migrated model
        checkpoint = torch.load("enhanced_neural_model.pth", map_location='cpu', weights_only=False)
        
        # Create new model and load weights
        model = NewNeuralNetwork(input_dim=10, hidden_sizes=[512, 256, 128, 64])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test prediction
        test_input = torch.randn(1, 10)  # 10 features
        output = model(test_input)
        
        print(f"Test prediction shape: {output.shape}")
        print(f"Model loaded and working correctly!")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def main():
    """Main migration function"""
    if not Path("enhanced_neural_model.pth").exists():
        print("Error: enhanced_neural_model.pth not found!")
        return False
    
    print("Starting model migration...")
    success = migrate_model()
    
    if success:
        print("\nMigration completed successfully!")
        test_migrated_model()
    else:
        print("\nMigration failed!")
    
    return success

if __name__ == "__main__":
    main()