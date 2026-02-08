#!/usr/bin/env python3
"""
Neural Model Manager
===================

Professional neural model manager for the trading app.
Handles model loading, validation, training, and management.

Features:
- Automatic model loading from saved files
- Model validation and testing
- Model training capabilities
- Version management
- Performance metrics tracking
- Model backup and restore
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import threading

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for forex prediction - matches saved models"""
    
    def __init__(self, input_dim: int = 6, output_size: int = 3):
        super(SimpleNeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)  # SELL, HOLD, BUY
        )
    
    def forward(self, x):
        return self.network(x)

class ProfitOptimizedNetwork(nn.Module):
    """Fallback class used to unpickle legacy profit-optimized PKL models."""

    def __init__(self, input_size: int = 80, hidden_size: int = 256, num_classes: int = 3):
        super(ProfitOptimizedNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_classes),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.direction_head(features), self.confidence_head(features)

class ScalpingNeuralNetwork(nn.Module):
    """Fallback class used to unpickle legacy scalping PKL models."""

    def __init__(self, input_size: int = 80, hidden_size: int = 128, num_classes: int = 3):
        super(ScalpingNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)

class LegacyFeatureEngine:
    """Placeholder class for unpickling serialized feature engines."""
    pass

class LegacyModelUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps legacy __main__ class names."""

    CLASS_MAP = {
        ("__main__", "ProfitOptimizedNetwork"): ProfitOptimizedNetwork,
        ("__main__", "ScalpingNeuralNetwork"): ScalpingNeuralNetwork,
        ("__main__", "ProfitFeatureEngine"): LegacyFeatureEngine,
        ("__main__", "ScalpingFeatureEngine"): LegacyFeatureEngine,
        ("profit_optimized_trainer", "ProfitOptimizedNetwork"): ProfitOptimizedNetwork,
        ("pkl_neural_model_trainer", "ScalpingNeuralNetwork"): ScalpingNeuralNetwork,
        ("profit_optimized_trainer", "ProfitFeatureEngine"): LegacyFeatureEngine,
        ("pkl_neural_model_trainer", "ScalpingFeatureEngine"): LegacyFeatureEngine,
    }

    def find_class(self, module, name):
        mapped_class = self.CLASS_MAP.get((module, name))
        if mapped_class is not None:
            return mapped_class
        return super().find_class(module, name)

class EnhancedNeuralNetwork(nn.Module):
    """Enhanced neural network for forex prediction"""
    
    def __init__(self, input_dim: int = 10, output_size: int = 3):
        super(EnhancedNeuralNetwork, self).__init__()
        
        # Simple but effective architecture for 10 features
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class NeuralModelManager:
    """Professional neural model manager"""
    
    def __init__(self, models_dir: str = "models"):
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model state
        self.current_model = None
        self.model_metadata = {}
        self.model_loaded = False
        
        # Training data cache
        self.training_data = None
        self.feature_dim = 6  # Neural model uses 6 features
        self.class_labels: List[str] = ["SELL", "HOLD", "BUY"]
        
        # Performance tracking
        self.performance_history = []
        
        # Thread safety
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_class_labels(raw_labels: Any, fallback: List[str]) -> List[str]:
        labels: List[str] = []
        if isinstance(raw_labels, dict):
            ordered_items = sorted(raw_labels.items(), key=lambda item: str(item[0]))
            labels = [str(value).strip().upper() for _, value in ordered_items]
        elif isinstance(raw_labels, (list, tuple)):
            labels = [str(value).strip().upper() for value in raw_labels]

        normalized: List[str] = []
        for label in labels[:3]:
            if label in ("BUY", "LONG"):
                normalized.append("BUY")
            elif label in ("SELL", "SHORT"):
                normalized.append("SELL")
            elif label in ("HOLD", "FLAT", "NONE"):
                normalized.append("HOLD")
            else:
                return fallback

        if len(normalized) == 3 and set(normalized) == {"BUY", "SELL", "HOLD"}:
            return normalized
        return fallback

    @staticmethod
    def _infer_feature_dim(model: nn.Module, fallback: int) -> int:
        try:
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    return int(layer.in_features)
        except Exception:
            pass
        return fallback

    @staticmethod
    def _load_pickle_payload(model_path: Path) -> Any:
        with model_path.open("rb") as model_file:
            return LegacyModelUnpickler(model_file).load()

    def _load_torch_checkpoint(self, checkpoint: Any) -> None:
        metadata = checkpoint.get('metadata', {}) if isinstance(checkpoint, dict) else {}
        loaded_feature_dim = checkpoint.get('feature_dim') if isinstance(checkpoint, dict) else None
        if loaded_feature_dim is None and isinstance(metadata, dict):
            loaded_feature_dim = metadata.get('feature_dim')
        if loaded_feature_dim is not None:
            self.feature_dim = int(loaded_feature_dim)

        self.current_model = SimpleNeuralNetwork(input_dim=self.feature_dim)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.current_model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            self.current_model.load_state_dict(checkpoint)
        else:
            raise ValueError("Unsupported checkpoint format")

        self.current_model.eval()

        if isinstance(checkpoint, dict) and 'metadata' in checkpoint:
            self.model_metadata = checkpoint['metadata']
        elif isinstance(checkpoint, dict) and 'training_date' in checkpoint:
            self.model_metadata = {
                'training_date': checkpoint['training_date'],
                'accuracy': checkpoint.get('accuracy', 'Unknown'),
                'win_rate': checkpoint.get('win_rate', 'Unknown')
            }
        else:
            self.model_metadata = {
                'training_date': 'Unknown',
                'accuracy': 'Unknown',
                'win_rate': 'Unknown'
            }
        if not isinstance(self.model_metadata, dict):
            self.model_metadata = {}

        class_labels = None
        if isinstance(checkpoint, dict):
            class_labels = checkpoint.get("class_labels") or checkpoint.get("label_names")
        if class_labels is None and isinstance(self.model_metadata, dict):
            class_labels = self.model_metadata.get("class_labels") or self.model_metadata.get("label_names")
        self.class_labels = self._normalize_class_labels(
            class_labels, fallback=["SELL", "HOLD", "BUY"]
        )

    def _load_pickle_checkpoint(self, payload: Any, model_path: Path) -> None:
        model_obj = None
        metadata: Dict[str, Any] = {}
        feature_dim = self.feature_dim
        class_labels = None

        if isinstance(payload, dict):
            model_obj = payload.get("neural_network") or payload.get("model") or payload.get("network")
            existing_metadata = payload.get("metadata")
            if isinstance(existing_metadata, dict):
                metadata.update(existing_metadata)

            for field in ("training_date", "model_version", "model_class", "num_classes", "training_symbols"):
                if field in payload and field not in metadata:
                    metadata[field] = payload.get(field)

            input_size = payload.get("input_size") or payload.get("feature_dim") or payload.get("input_dim")
            if input_size is not None:
                try:
                    feature_dim = int(input_size)
                except Exception:
                    pass

            feature_names = payload.get("feature_names")
            if isinstance(feature_names, list):
                metadata.setdefault("feature_count", len(feature_names))

            class_labels = payload.get("class_labels") or payload.get("label_names")
        else:
            model_obj = payload

        if not isinstance(model_obj, nn.Module):
            raise ValueError("PKL file does not contain a supported neural network object")

        self.current_model = model_obj
        self.current_model.eval()
        self.feature_dim = self._infer_feature_dim(self.current_model, fallback=feature_dim)

        metadata.setdefault("source_format", "pickle")
        metadata.setdefault("model_file", str(model_path))
        metadata.setdefault("feature_dim", self.feature_dim)
        self.model_metadata = metadata
        self.class_labels = self._normalize_class_labels(
            class_labels or self.model_metadata.get("class_labels"),
            fallback=["BUY", "SELL", "HOLD"],
        )
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load neural model from file
        
        Args:
            model_path: Path to model file (uses default if None)
            
        Returns:
            bool: True if model loaded successfully
        """
        with self._lock:
            try:
                if model_path is None:
                    # Look for regular model first, then enhanced
                    app_root = Path(__file__).resolve().parent.parent
                    default_paths = [
                        app_root / "ultimate_neural_model.pkl",
                        "ultimate_neural_model.pkl",
                        "neural_model.pth",
                        "enhanced_neural_model.pth",
                        "models/best_model.pth",
                        "models/current_model.pth"
                    ]
                    
                    model_path = None
                    for path in default_paths:
                        candidate = Path(path)
                        if candidate.exists():
                            model_path = str(candidate)
                            break
                    
                    if model_path is None:
                        self.logger.error("No model file found")
                        return False
                
                model_file = Path(model_path)
                if not model_file.exists():
                    raise FileNotFoundError(f"Model file not found: {model_file}")

                self.logger.info(f"Loading neural model from {model_file}")

                if model_file.suffix.lower() == ".pkl":
                    payload = self._load_pickle_payload(model_file)
                    self._load_pickle_checkpoint(payload, model_file)
                else:
                    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                    self._load_torch_checkpoint(checkpoint)
                
                self.model_loaded = True
                
                self.logger.info("Neural model loaded successfully")
                self.logger.info(f"Model metadata: {self.model_metadata}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                self.current_model = None
                self.model_loaded = False
                self.class_labels = ["SELL", "HOLD", "BUY"]
                return False
    
    def save_model(self, model_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Save current model to file
        
        Args:
            model_path: Path to save model
            metadata: Optional model metadata
            
        Returns:
            bool: True if saved successfully
        """
        with self._lock:
            try:
                if not self.model_loaded or self.current_model is None:
                    self.logger.error("No model loaded to save")
                    return False
                
                # Prepare save data
                save_data = {
                    'model_state_dict': self.current_model.state_dict(),
                    'feature_dim': self.feature_dim,
                    'metadata': metadata or self.model_metadata,
                    'save_date': datetime.now().isoformat()
                }
                
                # Ensure directory exists
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Save model
                torch.save(save_data, model_path)
                
                self.logger.info(f"Model saved to {model_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
                return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded and self.current_model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current model
        
        Returns:
            Dict containing model information
        """
        if not self.is_model_loaded():
            return {"error": "No model loaded"}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.current_model.parameters())
        trainable_params = sum(p.numel() for p in self.current_model.parameters() if p.requires_grad)
        
        info = {
            "model_loaded": True,
            "feature_dimension": self.feature_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": self._estimate_model_size(),
            "metadata": self.model_metadata,
            "performance_history": self.performance_history[-10:] if self.performance_history else []
        }
        
        return info
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB"""
        if not self.is_model_loaded():
            return 0.0
        
        param_size = 0
        for param in self.current_model.parameters():
            param_size += param.numel() * param.element_size()
        
        buffer_size = 0
        for buffer in self.current_model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def predict(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Make prediction using loaded model
        
        Args:
            features: Input features array
            
        Returns:
            Dict containing prediction results or None
        """
        if not self.is_model_loaded():
            self.logger.error("No model loaded for prediction")
            return None
        
        try:
            with torch.no_grad():
                # Ensure features are correct shape
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                # Convert to tensor
                X = torch.FloatTensor(features)
                
                # Make prediction
                model_outputs = self.current_model(X)
                output_logits = model_outputs
                confidence_head_value: Optional[float] = None
                if isinstance(model_outputs, (tuple, list)):
                    output_logits = model_outputs[0]
                    if len(model_outputs) > 1:
                        raw_confidence = model_outputs[1]
                        try:
                            conf_array = raw_confidence.detach().cpu().numpy().reshape(-1)
                            if len(conf_array) > 0:
                                confidence_head_value = float(np.clip(conf_array[0], 0.0, 1.0))
                        except Exception:
                            confidence_head_value = None

                if not torch.is_tensor(output_logits):
                    raise ValueError("Model output is not a tensor")
                if output_logits.ndim == 1:
                    output_logits = output_logits.reshape(1, -1)
                
                # Get probabilities
                probabilities = torch.softmax(output_logits, dim=1).detach().cpu().numpy()[0]
                predicted_class = int(torch.argmax(output_logits, dim=1).item())
                
                # Map classes
                classes = self.class_labels if len(self.class_labels) == 3 else ['SELL', 'HOLD', 'BUY']
                if predicted_class >= len(classes):
                    predicted_action = 'HOLD'
                    confidence = float(np.max(probabilities))
                else:
                    predicted_action = classes[predicted_class]
                    confidence = float(probabilities[predicted_class])
                if confidence_head_value is not None:
                    confidence = max(confidence, confidence_head_value)

                probability_map: Dict[str, float] = {'SELL': 0.0, 'HOLD': 0.0, 'BUY': 0.0}
                for idx, class_name in enumerate(classes):
                    if idx < len(probabilities):
                        normalized_name = str(class_name).strip().upper()
                        if normalized_name in probability_map:
                            probability_map[normalized_name] = float(probabilities[idx])
                
                return {
                    'action': predicted_action,
                    'confidence': confidence,
                    'probabilities': {
                        'SELL': probability_map['SELL'],
                        'HOLD': probability_map['HOLD'],
                        'BUY': probability_map['BUY']
                    },
                    'class_labels': classes,
                    'confidence_head': confidence_head_value,
                    'raw_outputs': output_logits.detach().cpu().numpy().tolist()
                }
                
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return None
    
    def validate_model(self, test_features: np.ndarray, test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Validate model performance on test data
        
        Args:
            test_features: Test feature array
            test_labels: True labels array
            
        Returns:
            Dict containing validation metrics
        """
        if not self.is_model_loaded():
            return {"error": "No model loaded for validation"}
        
        try:
            predictions = []
            confidences = []
            
            # Make predictions on test data
            for i in range(len(test_features)):
                pred = self.predict(test_features[i])
                if pred:
                    predictions.append(pred['action'])
                    confidences.append(pred['confidence'])
            
            if not predictions:
                return {"error": "No valid predictions made"}
            
            # Calculate metrics
            accuracy = self._calculate_accuracy(predictions, test_labels)
            avg_confidence = np.mean(confidences)
            
            # Store performance
            performance = {
                'validation_date': datetime.now().isoformat(),
                'accuracy': accuracy,
                'average_confidence': avg_confidence,
                'num_predictions': len(predictions)
            }
            
            self.performance_history.append(performance)
            
            self.logger.info(f"Model validation completed - Accuracy: {accuracy:.3f}, Confidence: {avg_confidence:.3f}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Model validation error: {e}")
            return {"error": str(e)}
    
    def _calculate_accuracy(self, predictions: List[str], true_labels: np.ndarray) -> float:
        """Calculate prediction accuracy"""
        if len(predictions) != len(true_labels):
            return 0.0
        
        # Convert string predictions to numeric
        pred_map = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
        numeric_predictions = [pred_map.get(pred, 1) for pred in predictions]
        
        correct = sum(1 for p, t in zip(numeric_predictions, true_labels) if p == t)
        accuracy = correct / len(predictions)
        
        return accuracy
    
    def train_new_model(self, training_data: Tuple[np.ndarray, np.ndarray], 
                       epochs: int = 100, learning_rate: float = 0.001) -> bool:
        """
        Train a new neural model
        
        Args:
            training_data: Tuple of (features, labels)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            
        Returns:
            bool: True if training successful
        """
        try:
            features, labels = training_data
            
            # Update feature dimension
            self.feature_dim = features.shape[1] if len(features.shape) > 1 else 1
            
            # Create new model
            self.current_model = SimpleNeuralNetwork(input_dim=self.feature_dim)
            optimizer = torch.optim.Adam(self.current_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Convert to tensors
            X = torch.FloatTensor(features)
            y = torch.LongTensor(labels)
            
            # Training loop
            self.current_model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.current_model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    self.logger.info(f"Training epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
            
            # Evaluate on training data
            self.current_model.eval()
            with torch.no_grad():
                outputs = self.current_model(X)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == y).float().mean().item()
            
            # Update metadata
            self.model_metadata = {
                'training_date': datetime.now().isoformat(),
                'training_epochs': epochs,
                'learning_rate': learning_rate,
                'training_accuracy': accuracy,
                'feature_dimension': self.feature_dim,
                'training_samples': len(features)
            }
            
            self.model_loaded = True
            
            self.logger.info(f"Model training completed - Final accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available model files
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        model_files = list(self.models_dir.glob("*.pth")) + list(self.models_dir.glob("*.pkl"))
        for model_file in model_files:
            try:
                # Get file stats
                stat = model_file.stat()
                
                # Try to load metadata
                metadata = {}
                try:
                    if model_file.suffix.lower() == ".pkl":
                        payload = self._load_pickle_payload(model_file)
                        if isinstance(payload, dict):
                            raw_meta = payload.get('metadata')
                            if isinstance(raw_meta, dict):
                                metadata.update(raw_meta)
                            for field in ("training_date", "model_version", "model_class"):
                                if field in payload and field not in metadata:
                                    metadata[field] = payload[field]
                        metadata.setdefault("source_format", "pickle")
                    else:
                        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                        if isinstance(checkpoint, dict) and 'metadata' in checkpoint:
                            metadata = checkpoint['metadata']
                        elif isinstance(checkpoint, dict) and 'training_date' in checkpoint:
                            metadata = {'training_date': checkpoint['training_date']}
                except:
                    pass
                
                models.append({
                    'file_path': str(model_file),
                    'file_name': model_file.name,
                    'file_size_mb': stat.st_size / (1024 * 1024),
                    'created_date': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'metadata': metadata
                })
                
            except Exception as e:
                self.logger.error(f"Error reading model {model_file}: {e}")
        
        # Sort by modification date (newest first)
        models.sort(key=lambda x: x['modified_date'], reverse=True)
        
        return models
    
    def delete_model(self, model_path: str) -> bool:
        """
        Delete a model file
        
        Args:
            model_path: Path to model file
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            model_file = Path(model_path)
            if model_file.exists():
                model_file.unlink()
                self.logger.info(f"Deleted model: {model_path}")
                return True
            else:
                self.logger.warning(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting model {model_path}: {e}")
            return False
    
    def backup_model(self, backup_name: str = None) -> str:
        """
        Create backup of current model
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            str: Path to backup file
        """
        if not self.is_model_loaded():
            self.logger.error("No model loaded to backup")
            return ""
        
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}.pth"
            
            backup_path = self.models_dir / backup_name
            
            # Save current model as backup
            backup_metadata = {
                **self.model_metadata,
                'backup_date': datetime.now().isoformat(),
                'backup_type': 'manual'
            }
            
            if self.save_model(str(backup_path), backup_metadata):
                self.logger.info(f"Model backed up to: {backup_path}")
                return str(backup_path)
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"Model backup error: {e}")
            return ""
