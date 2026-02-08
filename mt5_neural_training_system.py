#!/usr/bin/env python3
"""
Advanced MT5 Neural Training System
==================================

Comprehensive neural network training using MT5 historical data
with continuous learning and self-improvement capabilities.

Features:
- Real MT5 historical data collection
- Advanced feature engineering
- Multi-timeframe neural training
- Continuous learning pipeline
- Performance-based model updates
- Frequent trading optimization
- Loss prevention mechanisms
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import builtins as _builtins

CLASS_INDEX = {"SELL": 0, "HOLD": 1, "BUY": 2}
INDEX_CLASS = {idx: name for name, idx in CLASS_INDEX.items()}


def print(*args, **kwargs):  # noqa: A001
    """
    Console-safe print for Windows code pages that cannot encode emoji.
    """
    try:
        _builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = [str(a).encode("ascii", "ignore").decode() for a in args]
        _builtins.print(*safe_args, **kwargs)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    # Data parameters
    lookback_periods: int = 100
    prediction_horizon: int = 5
    min_data_points: int = 1000
    
    # Neural network parameters
    hidden_dim: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    
    # Training parameters
    batch_size: int = 32
    epochs_per_cycle: int = 50
    validation_split: float = 0.2
    
    # Performance thresholds
    min_accuracy_threshold: float = 0.65
    target_win_rate: float = 0.78
    max_drawdown: float = 0.05
    
    # Trading frequency
    min_trades_per_day: int = 5
    max_trades_per_day: int = 50
    
    # Risk management
    position_size_factor: float = 1.0
    risk_per_trade: float = 0.02

class MT5DataCollector:
    """Advanced MT5 historical data collector"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY', 'CADCHF',
            'BTCUSD'
        ]
        self.timeframes = [
            mt5.TIMEFRAME_M5,   # 5-minute
            mt5.TIMEFRAME_M15,  # 15-minute  
            mt5.TIMEFRAME_M30,  # 30-minute
            mt5.TIMEFRAME_H1,   # 1-hour
            mt5.TIMEFRAME_H4,   # 4-hour
        ]
        self.data_cache = {}

    def _resolve_symbol(self, symbol: str) -> str:
        """Resolve requested symbol to broker-available symbol."""
        requested = str(symbol or "").strip()
        if not requested:
            return requested

        info = mt5.symbol_info(requested)
        if info is not None:
            return info.name if getattr(info, 'name', None) else requested

        symbols = mt5.symbols_get() or []
        names = [s.name for s in symbols if getattr(s, 'name', None)]
        upper_names = {name.upper(): name for name in names}
        upper_requested = requested.upper()

        if upper_requested in upper_names:
            return upper_names[upper_requested]

        # Prefer full requested-symbol matches first (e.g., BTCUSD, BTCUSDm, BTCUSD.i).
        full_matches = [name for name in names if upper_requested in name.upper()]
        if full_matches:
            full_matches = sorted(
                full_matches,
                key=lambda n: (0 if n.upper().startswith(upper_requested) else 1, len(n))
            )
            return full_matches[0]

        if upper_requested.endswith("USD"):
            base = upper_requested[:-3]

            # Keep USD variants ahead of base-only alias.
            usd_variants = [
                name for name in names
                if name.upper().startswith(base) and "USD" in name.upper()
            ]
            if usd_variants:
                return sorted(usd_variants, key=len)[0]

            if base in upper_names:
                return upper_names[base]

            starts = [name for name in names if name.upper().startswith(base)]
            if starts:
                return sorted(starts, key=len)[0]

        return requested
        
    def collect_historical_data(self, days_back: int = 365) -> Dict[str, Dict[int, pd.DataFrame]]:
        """Collect comprehensive historical data from MT5"""
        print(f"📊 Collecting MT5 historical data ({days_back} days)...")
        if not mt5.initialize():
            print(f"MT5 initialize failed: {mt5.last_error()}")
            return {}

        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_data = {}
        
        for symbol in self.symbols:
            resolved_symbol = self._resolve_symbol(symbol)
            symbol_label = f"{symbol}->{resolved_symbol}" if resolved_symbol != symbol else symbol
            print(f"  Processing {symbol_label}...")
            symbol_data = {}

            try:
                mt5.symbol_select(resolved_symbol, True)
            except Exception:
                pass
            for timeframe in self.timeframes:
                try:
                    # Get historical data
                    rates = mt5.copy_rates_range(resolved_symbol, timeframe, start_date, end_date)
                    
                    if rates is None or len(rates) < 100:
                        print(f"    Insufficient data for {symbol_label} {timeframe}")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # Add technical indicators
                    df = self._add_technical_indicators(df)
                    
                    symbol_data[timeframe] = df
                    print(f"    {symbol_label} {timeframe}: {len(df)} candles")
                    
                except Exception as e:
                    print(f"    Error collecting {symbol_label} {timeframe}: {e}")
                    continue
            
            if symbol_data:
                all_data[symbol] = symbol_data
        
        print(f"✅ Collected data for {len(all_data)} symbols")
        return all_data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # Price-based indicators
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Stochastic
            low_min = df['low'].rolling(14).min()
            high_max = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Price action
            df['body'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['total_range'] = df['high'] - df['low']
            
            # Volatility
            df['atr'] = df['total_range'].rolling(14).mean()
            df['volatility'] = df['close'].rolling(20).std()
            
            # Support/Resistance levels
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['resistance1'] = 2 * df['pivot'] - df['low']
            df['support1'] = 2 * df['pivot'] - df['high']
            
            return df
            
        except Exception as e:
            print(f"    ⚠️  Error adding indicators: {e}")
            return df

class AdvancedFeatureEngineer:
    """Advanced feature engineering for neural training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def create_features(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> pd.DataFrame:
        """Create comprehensive feature matrix"""
        print("🔧 Creating advanced features...")
        
        all_features = []
        all_labels = []
        
        for symbol, timeframes in data.items():
            print(f"  🧠 Processing {symbol}...")
            
            # Get primary timeframe (M15 for training)
            if mt5.TIMEFRAME_M15 not in timeframes:
                continue
                
            df = timeframes[mt5.TIMEFRAME_M15].copy()
            
            # Create features
            features, labels = self._create_symbol_features(df, symbol)
            
            all_features.extend(features)
            all_labels.extend(labels)
        
        # Combine all features
        feature_df = pd.DataFrame(all_features)
        labels = np.array(all_labels)
        
        print(f"✅ Created {len(feature_df)} training samples with {feature_df.shape[1]} features")
        return feature_df, labels
    
    def _create_symbol_features(self, df: pd.DataFrame, symbol: str) -> Tuple[List, List]:
        """Create features for a specific symbol"""
        features = []
        labels = []
        
        # Remove NaN values
        df = df.dropna()
        
        if len(df) < self.config.lookback_periods + self.config.prediction_horizon:
            return features, labels
        
        for i in range(self.config.lookback_periods, len(df) - self.config.prediction_horizon):
            # Get historical window
            window = df.iloc[i-self.config.lookback_periods:i]
            
            # Calculate features
            feature_vector = []
            
            # Price features
            current_price = df.iloc[i]['close']
            prev_price = df.iloc[i-1]['close']
            
            feature_vector.extend([
                current_price / prev_price - 1,  # Return
                (current_price - window['close'].mean()) / (window['close'].std() + 1e-8),  # Z-score
                (current_price - window['close'].min()) / ((window['close'].max() - window['close'].min()) + 1e-8),  # Percentile
            ])
            
            # Technical indicator features
            indicators = ['sma_5', 'sma_10', 'sma_20', 'ema_12', 'macd', 'rsi', 'stoch_k', 'bb_position']
            for indicator in indicators:
                if indicator in window.columns:
                    value = float(window[indicator].iloc[-1])
                    if indicator in {'sma_5', 'sma_10', 'sma_20', 'ema_12'}:
                        value = (value / (current_price + 1e-8)) - 1.0
                    elif indicator == 'macd':
                        value = value / (current_price + 1e-8)
                    elif indicator in {'rsi', 'stoch_k'}:
                        value = value / 100.0
                    elif indicator == 'bb_position':
                        value = float(np.clip(value, -2.0, 2.0))
                    feature_vector.append(value)
                else:
                    feature_vector.append(0.0)
            
            # Momentum features
            returns = window['close'].pct_change().dropna()
            feature_vector.extend([
                float(returns.mean()) if len(returns) else 0.0,  # Average return
                float(returns.std()) if len(returns) else 0.0,   # Volatility
                float(returns.skew()) if len(returns) else 0.0,  # Skewness
                float(returns.kurtosis()) if len(returns) else 0.0,  # Kurtosis
            ])
            
            # Volume features (if available)
            if 'tick_volume' in window.columns:
                volume_ratio = window['tick_volume'].iloc[-1] / (window['tick_volume'].mean() + 1e-8)
                feature_vector.append(float(np.clip(volume_ratio, 0.0, 10.0)))
            else:
                feature_vector.append(1.0)
            
            # Market regime features
            trend_strength = self._calculate_trend_strength(window)
            feature_vector.extend(trend_strength)
            
            # Pattern recognition features
            pattern_features = self._extract_pattern_features(window)
            feature_vector.extend(pattern_features)
            
            # Risk features
            risk_features = self._calculate_risk_features(window)
            feature_vector.extend(risk_features)
            
            # Create label (next 5 candles direction)
            future_prices = df.iloc[i:i+self.config.prediction_horizon]['close']
            future_return = (future_prices.iloc[-1] / current_price) - 1
            
            # Create label: 0=SELL, 1=HOLD, 2=BUY
            if future_return > 0.001:  # 0.1% threshold
                label = 2  # BUY
            elif future_return < -0.001:  # -0.1% threshold
                label = 0  # SELL
            else:
                label = 1  # HOLD
            
            cleaned_vector = [
                float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
                for value in feature_vector
            ]
            features.append(cleaned_vector)
            labels.append(label)
        
        return features, labels
    
    def _calculate_trend_strength(self, window: pd.DataFrame) -> List[float]:
        """Calculate trend strength indicators"""
        try:
            # Linear trend
            prices = window['close'].values
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            price_scale = np.mean(np.abs(prices)) + 1e-8
            normalized_slope = (slope / price_scale) * len(prices)
            
            # Trend consistency
            up_moves = np.sum(np.diff(prices) > 0)
            trend_consistency = up_moves / max(len(prices) - 1, 1)
            
            # ADX-like measure
            high_low = window['high'] - window['low']
            close_open = np.abs(window['close'] - window['open'])
            range_sum = float((high_low + close_open).sum())
            normalized_range = range_sum / (price_scale * max(len(prices), 1))
            
            return [float(normalized_slope), float(trend_consistency), float(normalized_range)]
            
        except:
            return [0.0, 0.5, 0.0]
    
    def _extract_pattern_features(self, window: pd.DataFrame) -> List[float]:
        """Extract candlestick pattern features"""
        try:
            # Recent candlestick patterns
            recent_candles = window.tail(5)
            
            # Doji detection
            doji_count = sum(abs(recent_candles['close'] - recent_candles['open']) < 
                           recent_candles['total_range'] * 0.1)
            
            # Hammer/shooting star patterns
            hammer_count = 0
            for _, candle in recent_candles.iterrows():
                body = abs(candle['close'] - candle['open'])
                lower_shadow = min(candle['open'], candle['close']) - candle['low']
                upper_shadow = candle['high'] - max(candle['open'], candle['close'])
                
                if lower_shadow > body * 2 and upper_shadow < body:
                    hammer_count += 1
            
            return [doji_count / 5, hammer_count / 5]
            
        except:
            return [0.0, 0.0]
    
    def _calculate_risk_features(self, window: pd.DataFrame) -> List[float]:
        """Calculate risk-related features"""
        try:
            returns = window['close'].pct_change().dropna()
            
            # Value at Risk (5%)
            var_5 = np.percentile(returns, 5)
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return [var_5, max_drawdown]
            
        except:
            return [0.0, 0.0]

class AdvancedNeuralNetwork(nn.Module):
    """Advanced neural network for forex prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super(AdvancedNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.input_dropout = nn.Dropout(0.2)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        self.hidden_dropouts = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_dropouts.append(nn.Dropout(0.3))
        
        # Output layers for different predictions
        self.direction_head = nn.Linear(hidden_dim, 3)  # Buy, Sell, Hold
        self.confidence_head = nn.Linear(hidden_dim, 1)  # Confidence score
        self.risk_head = nn.Linear(hidden_dim, 1)  # Risk assessment
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Input processing
        x = self.relu(self.input_bn(self.input_layer(x)))
        x = self.input_dropout(x)
        
        # Hidden layers
        for layer, bn, dropout in zip(self.hidden_layers, self.hidden_bns, self.hidden_dropouts):
            residual = x
            x = self.relu(bn(layer(x)))
            x = dropout(x)
            
            # Add residual connection for deeper layers
            if x.size(1) == residual.size(1):
                x = x + residual
        
        # Output heads
        direction_logits = self.direction_head(x)
        confidence = torch.sigmoid(self.confidence_head(x))
        risk_score = torch.sigmoid(self.risk_head(x))
        
        return {
            'direction': direction_logits,
            'confidence': confidence,
            'risk': risk_score
        }

class ContinuousLearningSystem:
    """System for continuous learning and improvement"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []
        self.performance_metrics = {}
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        
    def initialize_model(self, input_dim: int):
        """Initialize the neural network model"""
        print(f"🧠 Initializing neural network (input_dim: {input_dim}, device: {self.device})...")
        
        self.model = AdvancedNeuralNetwork(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        print(f"✅ Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

    def _fit_feature_scaler(self, X_train: np.ndarray) -> None:
        """Fit standardization statistics on training data only."""
        mean = np.nanmean(X_train, axis=0)
        std = np.nanstd(X_train, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self.feature_mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.feature_std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)

    def _apply_feature_scaler(self, X: np.ndarray) -> np.ndarray:
        """Apply feature standardization with previously fitted statistics."""
        if self.feature_mean is None or self.feature_std is None:
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        scaled = (X - self.feature_mean) / self.feature_std
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
    def train_model(self, features: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """Train the neural network"""
        print("🚀 Starting neural network training...")
        
        # Prepare data
        X = features.values.astype(np.float32)
        y = labels.astype(np.int64)
        
        if len(X) < 50:
            raise ValueError(f"Insufficient samples for training: {len(X)}")
        
        # Split data
        split_idx = int(len(X) * (1 - self.config.validation_split))
        split_idx = max(1, min(split_idx, len(X) - 1))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features using train-only statistics to reduce pair-dependent bias.
        self._fit_feature_scaler(X_train)
        X_train = self._apply_feature_scaler(X_train)
        X_val = self._apply_feature_scaler(X_val)

        # Handle class imbalance (HOLD is usually dominant).
        class_counts = np.bincount(y_train, minlength=3).astype(np.float32)
        class_weights = np.where(class_counts > 0, class_counts.sum() / class_counts, 0.0)
        class_weights = class_weights / (class_weights.mean() + 1e-8)
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(self.device)
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        val_predictions = None
        
        for epoch in range(self.config.epochs_per_cycle):
            # Training
            self.model.train()
            epoch_loss = 0.0
            batches = 0
            
            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_train_tensor[i:i+self.config.batch_size]
                batch_y = y_train_tensor[i:i+self.config.batch_size]

                # BatchNorm layers require at least two samples in training mode.
                if batch_X.shape[0] < 2:
                    continue
                
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                direction_loss = self.criterion(outputs['direction'], batch_y)
                
                # Calibrate confidence: high for correct predictions, low for incorrect.
                with torch.no_grad():
                    predicted = torch.argmax(outputs['direction'], dim=1)
                    correct_mask = (predicted == batch_y).float()
                
                target_confidence = 0.2 + 0.8 * correct_mask
                confidence_loss = torch.mean(
                    (outputs['confidence'].squeeze() - target_confidence) ** 2
                )
                
                # Total loss
                total_loss = direction_loss + 0.1 * confidence_loss
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                batches += 1
            
            avg_epoch_loss = epoch_loss / max(batches, 1)
            train_losses.append(avg_epoch_loss)
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_predictions = torch.argmax(val_outputs['direction'], dim=1)
                    val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
                    val_accuracies.append(val_accuracy)
                    
                    print(f"  Epoch {epoch}: Loss={avg_epoch_loss:.4f}, Val Acc={val_accuracy:.3f}")
        
        # Ensure final validation metrics are always computed.
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val_tensor)
            val_predictions = torch.argmax(val_outputs['direction'], dim=1)
            final_val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
            val_accuracies.append(final_val_accuracy)

        # Calculate final metrics
        val_predictions_np = val_predictions.detach().cpu().numpy()
        final_metrics = {
            'final_train_loss': train_losses[-1],
            'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0.0,
            'win_rate': self._calculate_win_rate(val_predictions_np, y_val),
            'total_samples': len(X_train) + len(X_val)
        }
        
        self.training_history.append(final_metrics)
        
        print(f"✅ Training completed!")
        print(f"   Final validation accuracy: {final_metrics['final_val_accuracy']:.3f}")
        print(f"   Win rate: {final_metrics['win_rate']:.1%}")
        
        return final_metrics
    
    def _calculate_win_rate(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate win rate from predictions"""
        # Consider only buy/sell predictions (not HOLD class=1)
        trading_mask = predictions != 1
        
        if np.sum(trading_mask) == 0:
            return 0.0
        
        correct_trades = np.sum((predictions[trading_mask] == true_labels[trading_mask]))
        return correct_trades / np.sum(trading_mask)
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        with torch.no_grad():
            raw_features = np.asarray(features, dtype=np.float32).reshape(1, -1)
            scaled_features = self._apply_feature_scaler(raw_features)
            X_tensor = torch.FloatTensor(scaled_features).to(self.device)
            outputs = self.model(X_tensor)
            
            direction_probs = torch.softmax(outputs['direction'], dim=1).cpu().numpy()[0]
            confidence = float(outputs['confidence'].cpu().numpy()[0][0])
            risk_score = float(outputs['risk'].cpu().numpy()[0][0])
            
            # Determine action
            action_idx = int(np.argmax(direction_probs))
            action = INDEX_CLASS.get(action_idx, "HOLD")
            action_prob = float(direction_probs[action_idx])
            trade_score = float(confidence * action_prob * (1.0 - 0.5 * risk_score))
            
            return {
                'action': action,
                'confidence': confidence,
                'action_probability': action_prob,
                'trade_score': trade_score,
                'risk_score': risk_score,
                'probabilities': {
                    'SELL': float(direction_probs[0]),
                    'HOLD': float(direction_probs[1]),
                    'BUY': float(direction_probs[2])
                }
            }

class TradingOptimizer:
    """System for optimizing trading frequency and profitability"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trading_history = []
        self.performance_tracker = {}
        
    def optimize_trading_parameters(self, model, features: pd.DataFrame, 
                                  labels: np.ndarray) -> Dict[str, Any]:
        """Optimize trading parameters for maximum profitability"""
        print("⚡ Optimizing trading parameters...")
        
        # Test different trade-score thresholds.
        thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
        best_params = {}
        best_performance = -1.0
        
        for threshold in thresholds:
            performance = self._evaluate_threshold(model, features, labels, threshold)
            objective = float(performance['win_rate'] * min(
                max(performance['trades_per_day'], 1.0) / max(self.config.min_trades_per_day, 1),
                1.0
            ))
            
            if objective > best_performance:
                best_performance = objective
                best_params = {
                    'trade_score_threshold': threshold,
                    'objective_score': float(objective),
                    **performance
                }
        
        if not best_params:
            best_params = {
                'trade_score_threshold': thresholds[0],
                'objective_score': 0.0,
                'win_rate': 0.0,
                'trades_per_day': 0.0,
                'avg_trade_score': 0.0,
            }
        
        print(f"✅ Optimized parameters: {best_params}")
        return best_params
    
    def _evaluate_threshold(self, model_predictor, features: pd.DataFrame, 
                          labels: np.ndarray, threshold: float) -> Dict[str, float]:
        """Evaluate trading performance with specific threshold"""
        X = features.values
        y = labels
        
        predictions = []
        trade_scores = []
        for i in range(len(X)):
            pred = model_predictor.predict(X[i])
            score = float(pred.get('trade_score', pred.get('confidence', 0.0)))
            trade_scores.append(score)
            if score >= threshold:
                if pred['action'] == 'BUY':
                    predictions.append(2)
                elif pred['action'] == 'SELL':
                    predictions.append(0)
                else:
                    predictions.append(1)
            else:
                predictions.append(1)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        trading_mask = predictions != 1
        if np.sum(trading_mask) == 0:
            return {
                'win_rate': 0.0,
                'trades_per_day': 0.0,
                'avg_trade_score': float(np.mean(trade_scores) if trade_scores else 0.0),
            }
        
        correct_trades = np.sum((predictions[trading_mask] == y[trading_mask]))
        win_rate = float(correct_trades / np.sum(trading_mask))
        
        # Estimate trades per day (simplified)
        total_samples = len(predictions)
        trading_frequency = np.sum(trading_mask) / total_samples
        estimated_trades_per_day = trading_frequency * 96  # 96 candles/day on M15
        
        return {
            'win_rate': float(win_rate),
            'trades_per_day': float(min(estimated_trades_per_day, self.config.max_trades_per_day)),
            'avg_trade_score': float(np.mean(trade_scores) if trade_scores else 0.0)
        }

def main():
    """Main training pipeline"""
    print("🚀 Advanced MT5 Neural Training System")
    print("=" * 60)
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Initialize systems
    data_collector = MT5DataCollector()
    feature_engineer = AdvancedFeatureEngineer(config)
    learning_system = ContinuousLearningSystem(config)
    optimizer = TradingOptimizer(config)
    
    # Collect historical data
    historical_data = data_collector.collect_historical_data(days_back=365)
    
    if not historical_data:
        print("❌ No historical data collected!")
        return False
    
    # Create features
    features, labels = feature_engineer.create_features(historical_data)
    
    if len(features) < config.min_data_points:
        print(f"❌ Insufficient training data: {len(features)} < {config.min_data_points}")
        return False
    
    # Initialize and train model
    input_dim = features.shape[1]
    learning_system.initialize_model(input_dim)
    
    # Train the model
    training_metrics = learning_system.train_model(features, labels)
    
    # Optimize thresholds on out-of-sample validation only.
    split_idx = int(len(features) * (1 - config.validation_split))
    split_idx = max(1, min(split_idx, len(features) - 1))
    validation_features = features.iloc[split_idx:]
    validation_labels = labels[split_idx:]

    # Optimize trading parameters
    optimized_params = optimizer.optimize_trading_parameters(
        learning_system, validation_features, validation_labels
    )
    
    # Save the trained model
    model_path = "trained_neural_model.pth"
    torch.save({
        'model_state_dict': learning_system.model.state_dict(),
        'config': config,
        'input_dim': input_dim,
        'feature_mean': learning_system.feature_mean,
        'feature_std': learning_system.feature_std,
        'class_mapping': CLASS_INDEX,
        'last_update': datetime.now().isoformat(),
        'training_metrics': training_metrics,
        'optimized_params': optimized_params
    }, model_path)
    
    print(f"✅ Model saved to {model_path}")
    
    # Generate performance report
    report = {
        'training_date': datetime.now().isoformat(),
        'data_samples': len(features),
        'model_parameters': sum(p.numel() for p in learning_system.model.parameters()),
        'training_metrics': training_metrics,
        'optimized_parameters': optimized_params,
        'validation_samples': len(validation_features),
        'performance_summary': {
            'win_rate_target': config.target_win_rate,
            'win_rate_achieved': optimized_params.get('win_rate', 0),
            'trades_frequency': optimized_params.get('trades_per_day', 0),
            'model_improvement': 'Ready for continuous learning'
        }
    }
    
    # Save report
    with open('training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📊 Training completed successfully!")
    print(f"   Model accuracy: {training_metrics['final_val_accuracy']:.1%}")
    print(f"   Win rate: {optimized_params.get('win_rate', 0):.1%}")
    print(f"   Estimated trades/day: {optimized_params.get('trades_per_day', 0):.1f}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Neural training system completed successfully!")
    else:
        print("❌ Training failed!")
        sys.exit(1)
