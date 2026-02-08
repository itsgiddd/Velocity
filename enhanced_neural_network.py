"""
Enhanced Neural Network for Multi-Timeframe Analysis and Dynamic Profit Prediction
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

CLASS_LABELS = ("BUY", "SELL", "HOLD")
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_LABELS)}
INDEX_TO_CLASS = {idx: label for idx, label in enumerate(CLASS_LABELS)}

class MultiTimeframeNeuralNetwork(nn.Module):
    """
    Enhanced neural network that analyzes multiple timeframes simultaneously
    and makes dynamic profit predictions
    """
    
    def __init__(self, input_features_per_timeframe=64):
        super().__init__()
        
        # Multi-timeframe input dimensions
        self.m15_features = input_features_per_timeframe
        self.h1_features = input_features_per_timeframe
        self.h4_features = input_features_per_timeframe  
        self.d1_features = input_features_per_timeframe
        
        # Individual timeframe processors
        self.m15_processor = nn.Sequential(
            nn.Linear(self.m15_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.h1_processor = nn.Sequential(
            nn.Linear(self.h1_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.h4_processor = nn.Sequential(
            nn.Linear(self.h4_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.d1_processor = nn.Sequential(
            nn.Linear(self.d1_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Attention mechanism for timeframe fusion
        self.timeframe_attention = nn.Sequential(
            nn.Linear(32 * 4, 64),
            nn.Tanh(),
            nn.Linear(64, 4),  # 4 timeframes
            nn.Softmax(dim=1)
        )
        
        # Fusion layer
        fusion_input_size = 32 * 4 + 4  # 4 timeframes + attention weights
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Dynamic profit prediction head
        self.profit_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # [entry_signal, stop_loss_offset, take_profit_offset, confidence]
            nn.Tanh()
        )
        
        # Trading decision head (logits; softmax is applied outside for training stability)
        self.decision_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [BUY, SELL, HOLD]
        )
        
        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),  # Risk factors
            nn.Sigmoid()
        )
        
    def forward(self, m15_data, h1_data, h4_data, d1_data):
        """
        Forward pass with multi-timeframe analysis
        """
        # Process each timeframe
        m15_features = self.m15_processor(m15_data)
        h1_features = self.h1_processor(h1_data)
        h4_features = self.h4_processor(h4_data)
        d1_features = self.d1_processor(d1_data)
        
        # Combine timeframe features
        combined_features = torch.cat([m15_features, h1_features, h4_features, d1_features], dim=1)
        
        # Calculate attention weights
        attention_weights = self.timeframe_attention(combined_features)
        
        # Apply attention and add attention weights
        attended_features = combined_features * attention_weights.repeat(1, 32)
        
        # Concatenate attended features with attention weights
        fusion_input = torch.cat([attended_features, attention_weights], dim=1)
        
        # Final fusion
        fused_features = self.fusion_layer(fusion_input)
        
        # Generate predictions
        profit_prediction = self.profit_predictor(fused_features)
        trading_decision = self.decision_head(fused_features)
        risk_assessment = self.risk_head(fused_features)
        
        return {
            "profit_targets": profit_prediction,
            "trading_decision": trading_decision,
            "risk_assessment": risk_assessment,
            "timeframe_attention": attention_weights,
            "fused_features": fused_features
        }

class MultiTimeframeFeatureExtractor:
    """
    Extract neural features from multiple timeframes
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _safe_last(self, arr: np.ndarray, default: float) -> float:
        if arr is None or len(arr) == 0:
            return default
        value = float(arr[-1])
        if np.isnan(value) or np.isinf(value):
            return default
        return value
        
    def extract_timeframe_features(self, data: pd.DataFrame, timeframe: str) -> np.ndarray:
        """
        Extract neural features from a single timeframe
        """
        if data is None or len(data) < 50:
            return np.zeros(64)  # Return zeros if insufficient data
            
        features = []
        
        # Price action features (16 features)
        closes = data['close'].values[-20:]  # Last 20 closes
        highs = data['high'].values[-20:]    # Last 20 highs
        lows = data['low'].values[-20:]      # Last 20 lows
        
        # Normalize prices
        closes_norm = (closes - closes.mean()) / (closes.std() + 1e-8)
        highs_norm = (highs - closes.mean()) / (closes.std() + 1e-8)
        lows_norm = (lows - closes.mean()) / (closes.std() + 1e-8)
        
        features.extend(closes_norm[-16:])  # Last 16 normalized closes
        
        # Technical indicators (16 features)
        rsi = self.calculate_rsi(data['close'], 14)
        macd_line, macd_signal = self.calculate_macd(data['close'])
        bb_upper, bb_lower = self.calculate_bollinger_bands(data['close'], 20)
        
        close_last = float(closes[-1]) if len(closes) > 0 else 1.0
        bb_upper_last = self._safe_last(bb_upper, close_last)
        bb_lower_last = self._safe_last(bb_lower, close_last)
        bb_width = (bb_upper_last - bb_lower_last) / (close_last + 1e-8)
        bb_position = (close_last - bb_lower_last) / ((bb_upper_last - bb_lower_last) + 1e-8)

        indicators = [
            self._safe_last(rsi, 50.0) / 100.0,  # RSI normalized to 0..1
            self._safe_last(macd_line, 0.0) / (close_last + 1e-8),  # Scale-invariant MACD
            self._safe_last(macd_signal, 0.0) / (close_last + 1e-8),
            bb_width,
            float(np.clip(bb_position, -2.0, 2.0)),
            (highs[-1] - lows[-1]) / close_last if close_last != 0 else 0.0,  # Candle range
            (closes[-1] - closes[-2]) / (closes[-2] + 1e-8) if len(closes) > 1 else 0.0,  # Last return
        ]
        
        # Add more indicators to reach 16
        while len(indicators) < 16:
            indicators.append(0.0)
        
        features.extend(indicators[:16])
        
        # Volume features (8 features)
        if 'tick_volume' in data.columns:
            volume = data['tick_volume'].values[-10:]
            volume_norm = (volume - volume.mean()) / (volume.std() + 1e-8)
            features.extend(volume_norm[-8:])
        else:
            features.extend([0.0] * 8)
        
        # Pattern recognition features (8 features)
        pattern_features = self.extract_pattern_features(data)
        features.extend(pattern_features)
        
        # Volatility features (8 features)
        volatility_features = self.extract_volatility_features(data)
        features.extend(volatility_features)
        
        # Trend features (8 features)
        trend_features = self.extract_trend_features(data)
        features.extend(trend_features)
        
        return np.array(features[:64])  # Ensure exactly 64 features
        
    def extract_pattern_features(self, data: pd.DataFrame) -> List[float]:
        """Extract pattern-based features"""
        if len(data) < 10:
            return [0.0] * 8
            
        features = []
        
        # Simple trend patterns
        closes = data['close'].values[-10:]
        
        # Trend direction
        if len(closes) >= 3:
            trend_up = sum(1 for i in range(1, min(4, len(closes))) if closes[-i] > closes[-i-1])
            trend_down = sum(1 for i in range(1, min(4, len(closes))) if closes[-i] < closes[-i-1])
            
            features.append(trend_up / 3.0)  # Up trend score
            features.append(trend_down / 3.0)  # Down trend score
        else:
            features.extend([0.0, 0.0])
            
        # Support/resistance levels
        recent_high = np.max(data['high'].values[-20:])
        recent_low = np.min(data['low'].values[-20:])
        current_price = data['close'].iloc[-1]
        
        support_distance = (current_price - recent_low) / current_price if current_price != 0 else 0
        resistance_distance = (recent_high - current_price) / current_price if current_price != 0 else 0
        
        features.append(support_distance)
        features.append(resistance_distance)
        
        # Price momentum
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] != 0 else 0
            features.append(momentum)
        else:
            features.append(0.0)
            
        # Fill remaining features
        while len(features) < 8:
            features.append(0.0)
            
        return features[:8]
        
    def extract_volatility_features(self, data: pd.DataFrame) -> List[float]:
        """Extract volatility-based features"""
        if len(data) < 20:
            return [0.0] * 8
            
        features = []
        
        # Price volatility
        closes = data['close'].values
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        features.append(volatility)
        
        # Average true range (simplified)
        if len(data) >= 14:
            tr_values = []
            for i in range(max(0, len(data) - 14), len(data)):
                high_low = data['high'].iloc[i] - data['low'].iloc[i]
                high_close = abs(data['high'].iloc[i] - data['close'].iloc[i-1] if i > 0 else 0)
                low_close = abs(data['low'].iloc[i] - data['close'].iloc[i-1] if i > 0 else 0)
                tr = max(high_low, high_close, low_close)
                tr_values.append(tr)
            
            atr = np.mean(tr_values) if tr_values else 0
            features.append(atr / data['close'].iloc[-1])  # Normalized ATR
        else:
            features.append(0.0)
            
        # Fill remaining features
        while len(features) < 8:
            features.append(0.0)
            
        return features[:8]
        
    def extract_trend_features(self, data: pd.DataFrame) -> List[float]:
        """Extract trend-based features"""
        if len(data) < 20:
            return [0.0] * 8
            
        features = []
        
        # Moving averages
        closes = data['close'].values
        current_price = closes[-1]
        
        if len(closes) >= 20:
            ma_20 = np.mean(closes[-20:])
            ma_10 = np.mean(closes[-10:])
            ma_5 = np.mean(closes[-5:])
            
            features.append((current_price - ma_20) / ma_20)  # Price vs MA20
            features.append((ma_10 - ma_20) / ma_20)  # MA10 vs MA20
            features.append((ma_5 - ma_10) / ma_10)  # MA5 vs MA10
        else:
            features.extend([0.0, 0.0, 0.0])
            
        # Trend strength
        if len(closes) >= 10:
            trend_slope = np.polyfit(range(10), closes[-10:], 1)[0]
            features.append(trend_slope / current_price if current_price != 0 else 0)
        else:
            features.append(0.0)
            
        # Fill remaining features
        while len(features) < 8:
            features.append(0.0)
            
        return features[:8]
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return np.array([])
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
        
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return np.array([]), np.array([])
            
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        
        return macd_line.values, macd_signal.values
        
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return np.array([]), np.array([])
            
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        return upper_band.values, lower_band.values

class EnhancedNeuralTrainer:
    """
    Train the enhanced neural network with multi-timeframe analysis
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiTimeframeNeuralNetwork().to(self.device)
        self.feature_extractor = MultiTimeframeFeatureExtractor()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_function = nn.MSELoss()
        self.decision_loss = nn.CrossEntropyLoss()
        
        if model_path:
            self.load_model(model_path)

    def _build_class_weight_tensor(self, training_data: List[Dict]) -> torch.Tensor:
        counts = np.zeros(len(CLASS_LABELS), dtype=np.float32)
        for item in training_data:
            label = int(item.get("decision", CLASS_TO_INDEX["HOLD"]))
            label = int(np.clip(label, 0, len(CLASS_LABELS) - 1))
            counts[label] += 1.0
        counts = np.where(counts > 0, counts, 1.0)
        weights = counts.sum() / counts
        weights = weights / (weights.mean() + 1e-8)
        return torch.FloatTensor(weights).to(self.device)
            
    def train_model(self, training_data: List[Dict], epochs: int = 100):
        """
        Train the enhanced neural network
        """
        self.logger = logging.getLogger(__name__)
        if not training_data:
            self.logger.warning("No training data provided; skipping training")
            return
        self.logger.info(f"Starting training with {len(training_data)} samples for {epochs} epochs")
        self.decision_loss = nn.CrossEntropyLoss(
            weight=self._build_class_weight_tensor(training_data)
        )
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            indices = np.random.permutation(len(training_data))
            for idx in indices:
                batch = training_data[int(idx)]
                # Extract multi-timeframe features
                m15_features = self.feature_extractor.extract_timeframe_features(
                    batch.get('m15_data'), 'M15'
                )
                h1_features = self.feature_extractor.extract_timeframe_features(
                    batch.get('h1_data'), 'H1'
                )
                h4_features = self.feature_extractor.extract_timeframe_features(
                    batch.get('h4_data'), 'H4'
                )
                d1_features = self.feature_extractor.extract_timeframe_features(
                    batch.get('d1_data'), 'D1'
                )
                
                # Convert to tensors
                m15_tensor = torch.FloatTensor(m15_features).unsqueeze(0).to(self.device)
                h1_tensor = torch.FloatTensor(h1_features).unsqueeze(0).to(self.device)
                h4_tensor = torch.FloatTensor(h4_features).unsqueeze(0).to(self.device)
                d1_tensor = torch.FloatTensor(d1_features).unsqueeze(0).to(self.device)
                
                # Forward pass
                predictions = self.model(m15_tensor, h1_tensor, h4_tensor, d1_tensor)
                
                # Calculate loss (simplified - in practice you'd have real targets)
                raw_targets = list(batch.get("profit_targets", [0.0, -0.01, 0.02, 0.5]))
                if len(raw_targets) < 4:
                    raw_targets.extend([0.0] * (4 - len(raw_targets)))
                raw_targets = raw_targets[:4]
                target_tensor = torch.FloatTensor(raw_targets).unsqueeze(0).to(self.device)

                profit_loss = self.loss_function(predictions["profit_targets"], target_tensor)
                
                decision_target = int(batch.get("decision", 2))
                decision_target = min(max(decision_target, 0), 2)
                decision_loss = self.decision_loss(
                    predictions["trading_decision"],
                    torch.LongTensor([decision_target]).to(self.device)  # 2 = HOLD
                )
                
                # Combined loss
                total_batch_loss = profit_loss + decision_loss
                
                # Backpropagation
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {total_loss/len(training_data):.4f}")
                
    def predict(self, m15_data: pd.DataFrame, h1_data: pd.DataFrame, 
                h4_data: pd.DataFrame, d1_data: pd.DataFrame) -> Dict:
        """
        Make prediction using the enhanced neural network
        """
        self.model.eval()
        
        with torch.no_grad():
            # Extract features
            m15_features = self.feature_extractor.extract_timeframe_features(m15_data, 'M15')
            h1_features = self.feature_extractor.extract_timeframe_features(h1_data, 'H1')
            h4_features = self.feature_extractor.extract_timeframe_features(h4_data, 'H4')
            d1_features = self.feature_extractor.extract_timeframe_features(d1_data, 'D1')
            
            # Convert to tensors
            m15_tensor = torch.FloatTensor(m15_features).unsqueeze(0).to(self.device)
            h1_tensor = torch.FloatTensor(h1_features).unsqueeze(0).to(self.device)
            h4_tensor = torch.FloatTensor(h4_features).unsqueeze(0).to(self.device)
            d1_tensor = torch.FloatTensor(d1_features).unsqueeze(0).to(self.device)
            
            # Get predictions
            predictions = self.model(m15_tensor, h1_tensor, h4_tensor, d1_tensor)
            decision_probs = torch.softmax(predictions["trading_decision"], dim=1)
            probs_np = decision_probs.squeeze().cpu().numpy()
            decision_index = int(np.argmax(probs_np))
            sorted_probs = np.sort(probs_np)[::-1]
            decision_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
            entropy = float(-np.sum(probs_np * np.log(probs_np + 1e-12)))
            max_entropy = float(np.log(len(probs_np))) if len(probs_np) > 1 else 1.0
            normalized_uncertainty = float(entropy / (max_entropy + 1e-12))
            
            return {
                "profit_targets": predictions["profit_targets"].squeeze().cpu().numpy(),
                "trading_decision": probs_np,
                "risk_assessment": predictions["risk_assessment"].squeeze().cpu().numpy(),
                "timeframe_attention": predictions["timeframe_attention"].squeeze().cpu().numpy(),
                "confidence": float(np.max(probs_np)),
                "decision_index": decision_index,
                "decision_label": INDEX_TO_CLASS.get(decision_index, "HOLD"),
                "decision_margin": decision_margin,
                "uncertainty": normalized_uncertainty
            }
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_labels': CLASS_LABELS,
            'input_features_per_timeframe': 64,
        }, path)
        
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def main():
    """Demo the enhanced neural network"""
    # Create sample data
    sample_data = {
        'open': np.random.random(100) + 150,
        'high': np.random.random(100) + 151,
        'low': np.random.random(100) + 149,
        'close': np.random.random(100) + 150,
        'tick_volume': np.random.randint(100, 1000, 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize trainer
    trainer = EnhancedNeuralTrainer()
    
    # Extract features
    features = trainer.feature_extractor.extract_timeframe_features(df, 'M15')
    print(f"Extracted {len(features)} features from timeframe")
    print(f"Feature sample: {features[:10]}")
    
    # Make prediction
    predictions = trainer.predict(df, df, df, df)
    print(f"\nPrediction keys: {predictions.keys()}")
    print(f"Trading decision: {predictions['trading_decision']}")
    print(f"Confidence: {predictions['confidence']:.3f}")
    print(f"Timeframe attention: {predictions['timeframe_attention']}")

if __name__ == "__main__":
    main()
