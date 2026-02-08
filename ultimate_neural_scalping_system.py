#!/usr/bin/env python3
"""
Ultimate Neural Scalping System - Maximum Profit Focus
================================================

A specialized neural trading system designed for:
- ALL 9 currency pairs + crypto
- Continuous scalping (no position holding)
- Maximum profitability focus
- Comprehensive market knowledge
- Real-time decision making

Key Features:
- Ultra-fast scalping decisions (5-15 minute trades)
- All market information integration
- No restrictive filters (maximum trading frequency)
- Profit-optimized neural architecture
- Real-time MT5 data integration
- Advanced feature engineering for scalping
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
import pickle
import talib
from pathlib import Path
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalpingFeatureEngine:
    """
    Advanced feature engineering specifically for scalping
    Focus: Extract maximum market information for ultra-fast decisions
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.last_update = {}
    
    def create_scalping_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Create comprehensive features optimized for scalping
        Focus: Speed + Information density
        """
        try:
            features = pd.DataFrame(index=data.index)
            
            # === PRICE ACTION (High Frequency) ===
            # Immediate price movements
            features['returns_1m'] = data['close'].pct_change()
            features['returns_5m'] = data['close'].pct_change(periods=5)
            features['returns_15m'] = data['close'].pct_change(periods=15)
            
            # Price position relative to recent highs/lows
            features['position_5m'] = (data['close'] - data['low'].rolling(5).min()) / (data['high'].rolling(5).max() - data['low'].rolling(5).min())
            features['position_15m'] = (data['close'] - data['low'].rolling(15).min()) / (data['high'].rolling(15).max() - data['low'].rolling(15).min())
            
            # === MOMENTUM (Ultra-fast) ===
            # Very short-term momentum
            features['momentum_3m'] = data['close'] - data['close'].shift(3)
            features['momentum_7m'] = data['close'] - data['close'].shift(7)
            features['momentum_13m'] = data['close'] - data['close'].shift(13)
            
            # RSI for scalping (fast periods)
            for period in [7, 14, 21]:
                features[f'rsi_{period}'] = talib.RSI(data['close'].values, timeperiod=period)
                features[f'rsi_oversold_{period}'] = (features[f'rsi_{period}'] < 30).astype(int)
                features[f'rsi_overbought_{period}'] = (features[f'rsi_{period}'] > 70).astype(int)
            
            # === VOLATILITY (Dynamic) ===
            features['atr_7m'] = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=7)
            features['atr_ratio'] = features['atr_7m'] / data['close']
            features['volatility_3m'] = features['returns_1m'].rolling(3).std()
            features['volatility_15m'] = features['returns_1m'].rolling(15).std()
            
            # === VOLUME ANALYSIS ===
            if 'tick_volume' in data.columns:
                features['volume_sma_5'] = data['tick_volume'].rolling(5).mean()
                features['volume_ratio'] = data['tick_volume'] / features['volume_sma_5']
                features['volume_spike'] = (features['volume_ratio'] > 2.0).astype(int)
            
            # === SUPPORT/RESISTANCE (Dynamic) ===
            # Dynamic support and resistance levels
            for period in [10, 20, 50]:
                features[f'support_{period}'] = data['low'].rolling(period).min()
                features[f'resistance_{period}'] = data['high'].rolling(period).max()
                features[f'distance_to_support_{period}'] = (data['close'] - features[f'support_{period}']) / features[f'atr_7m']
                features[f'distance_to_resistance_{period}'] = (features[f'resistance_{period}'] - data['close']) / features[f'atr_7m']
            
            # === MOVING AVERAGES (Multiple Timeframes) ===
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = talib.SMA(data['close'].values, timeperiod=period)
                features[f'ema_{period}'] = talib.EMA(data['close'].values, timeperiod=period)
                features[f'price_vs_sma_{period}'] = data['close'] / features[f'sma_{period}'] - 1
                features[f'price_vs_ema_{period}'] = data['close'] / features[f'ema_{period}'] - 1
            
            # MA convergence/divergence
            features['sma_5_20_cross'] = np.where(features['sma_5'] > features['sma_20'], 1, -1)
            features['ema_10_50_cross'] = np.where(features['ema_10'] > features['ema_50'], 1, -1)
            
            # === BOLLINGER BANDS (Dynamic) ===
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values)
            features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            features['bb_squeeze'] = ((bb_upper - bb_lower) / bb_middle < 0.1).astype(int)
            features['bb_expansion'] = ((bb_upper - bb_lower) / bb_middle > 0.2).astype(int)
            
            # === MACD (Fast) ===
            macd, macd_signal, macd_hist = talib.MACD(data['close'].values)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_crossover'] = np.where(macd > macd_signal, 1, -1)
            features['macd_strength'] = np.abs(macd_hist)
            
            # === STOCHASTIC (Quick) ===
            slowk, slowd = talib.STOCH(data['high'].values, data['low'].values, data['close'].values)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            features['stoch_oversold'] = (slowk < 20).astype(int)
            features['stoch_overbought'] = (slowk > 80).astype(int)
            
            # === TIME-BASED FEATURES ===
            # Market sessions and timing
            timestamps = pd.to_datetime(data.index)
            features['hour'] = timestamps.hour
            features['minute'] = timestamps.minute
            features['day_of_week'] = timestamps.dayofweek
            
            # Market session indicators
            features['london_session'] = ((features['hour'] >= 8) & (features['hour'] <= 16)).astype(int)
            features['ny_session'] = ((features['hour'] >= 13) & (features['hour'] <= 22)).astype(int)
            features['overlap'] = ((features['london_session'] == 1) & (features['ny_session'] == 1)).astype(int)
            
            # === CORRELATION FEATURES (Multi-symbol) ===
            # Placeholder for cross-pair analysis - will be populated in multi-symbol mode
            
            # === PATTERN RECOGNITION ===
            # Candlestick patterns for scalping
            features['doji'] = talib.CDLDOJI(data['open'].values, data['high'].values, 
                                           data['low'].values, data['close'].values)
            features['hammer'] = talib.CDLHAMMER(data['open'].values, data['high'].values,
                                             data['low'].values, data['close'].values)
            features['engulfing'] = talib.CDLENGULFING(data['open'].values, data['high'].values,
                                                    data['low'].values, data['close'].values)
            
            # === MARKET MICROSTRUCTURE ===
            # Spread and liquidity indicators
            if 'spread' in data.columns:
                features['spread_normalized'] = data['spread'] / data['close']
            else:
                # Estimate spread from high-low
                features['estimated_spread'] = (data['high'] - data['low']) / data['close']
            
            # Price efficiency
            price_change = abs(data['close'] - data['close'].shift(5))
            total_movement = data['close'].diff().abs().rolling(5).sum()
            features['price_efficiency'] = price_change / (total_movement + 1e-8)
            
            # === ADVANCED SCALPING INDICATORS ===
            # Force Index
            force_index = (data['close'] - data['close'].shift(1)) * (data['tick_volume'] if 'tick_volume' in data.columns else 1000)
            features['force_index'] = force_index.rolling(13).sum()
            
            # Awesome Oscillator
            try:
                # Calculate Awesome Oscillator manually since talib.AO might not exist
                sma5 = talib.SMA(data['high'].values + data['low'].values, timeperiod=5) / 2
                sma34 = talib.SMA(data['high'].values + data['low'].values, timeperiod=34) / 2
                features['ao'] = sma5 - sma34
            except:
                features['ao'] = 0
            
            # === COMPOSITE SIGNALS ===
            # Multiple timeframe confirmation
            features['bullish_convergence'] = (
                (features['sma_5'] > features['sma_20']) &
                (features['rsi_7'] < 70) &
                (features['macd'] > features['macd_signal'])
            ).astype(int)
            
            features['bearish_convergence'] = (
                (features['sma_5'] < features['sma_20']) &
                (features['rsi_7'] > 30) &
                (features['macd'] < features['macd_signal'])
            ).astype(int)
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            # Remove infinite values
            features = features.replace([np.inf, -np.inf], 0)
            
            logger.info(f"Created {features.shape[1]} scalping features for {symbol}")
            return features
            
        except Exception as e:
            logger.error(f"Error creating scalping features for {symbol}: {e}")
            return pd.DataFrame()

class UltraFastNeuralNetwork(nn.Module):
    """
    Ultra-fast neural network optimized for scalping decisions
    Architecture: Fast inference + High accuracy
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        
        # Fast inference layers
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 4, 3)  # BUY, SELL, HOLD
        )
        
        # Confidence layer
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features for different heads
        for layer in self.network[:-1]:  # All layers except last
            x = layer(x)
        
        # Main prediction
        prediction = self.network[-1](x)  # Last layer
        
        # Additional heads
        confidence = self.confidence_head(x)
        risk_score = self.risk_head(x)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'risk_score': risk_score,
            'features': x  # For feature importance analysis
        }

class UltimateNeuralScalpingSystem:
    """
    Ultimate Neural Scalping System - Maximum Profit Focus
    """
    
    def __init__(self):
        self.feature_engine = ScalpingFeatureEngine()
        self.model = None
        self.is_trained = False
        self.training_history = []
        
        # Trading pairs (all 9 pairs)
        self.trading_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", 
            "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD"
        ]
        
        # Scalping parameters
        self.scalping_config = {
            'trade_duration_minutes': [5, 10, 15],  # Scalp duration options
            'max_hold_time': 20,  # Maximum hold time in minutes
            'min_confidence': 0.6,  # Minimum confidence for scalping
            'target_profit_pips': [5, 10, 15],  # Scalping targets
            'stop_loss_pips': [3, 5, 7],  # Stop loss levels
            'position_size_multiplier': 1.5  # Larger positions for scalping
        }
        
        # MT5 connection
        self.mt5_connected = False
        self._connect_mt5()
        
        logger.info("Ultimate Neural Scalping System initialized")
    
    def _connect_mt5(self):
        """Connect to MT5 for real-time data"""
        try:
            if mt5.initialize():
                self.mt5_connected = True
                logger.info("MT5 connected successfully")
            else:
                logger.warning("Failed to connect to MT5")
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
    
    def get_real_time_data(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M1, 
                          bars: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get real-time data from MT5 for scalping, with fallback to synthetic data
        """
        try:
            if self.mt5_connected:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    return df
            
            # Fallback to synthetic data for training
            logger.warning(f"Using synthetic data for {symbol} - MT5 data unavailable")
            return self._generate_synthetic_data(symbol, bars)
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            # Return synthetic data as fallback
            return self._generate_synthetic_data(symbol, bars)
    
    def _generate_synthetic_data(self, symbol: str, bars: int) -> pd.DataFrame:
        """
        Generate synthetic forex data for training when MT5 is unavailable
        """
        try:
            # Create realistic price data
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            
            # Base prices for different symbols
            base_prices = {
                'EURUSD': 1.1000, 'GBPUSD': 1.2500, 'USDJPY': 110.00,
                'AUDUSD': 0.7500, 'USDCAD': 1.2500, 'NZDUSD': 0.7000,
                'EURJPY': 120.00, 'GBPJPY': 135.00, 'BTCUSD': 45000.0
            }
            
            base_price = base_prices.get(symbol, 1.1000)
            
            # Generate realistic price movements
            dates = pd.date_range(start='2024-01-01', periods=bars, freq='1min')
            
            # Volatility based on symbol type
            volatility = {
                'EURUSD': 0.001, 'GBPUSD': 0.0015, 'USDJPY': 0.001,
                'AUDUSD': 0.0012, 'USDCAD': 0.001, 'NZDUSD': 0.0015,
                'EURJPY': 0.002, 'GBPJPY': 0.0025, 'BTCUSD': 0.01
            }.get(symbol, 0.001)
            
            returns = np.random.normal(0, volatility, bars)
            prices = base_price * np.cumprod(1 + returns)
            
            # Create OHLCV data
            df = pd.DataFrame(index=dates)
            df['open'] = prices
            df['close'] = prices
            
            # Generate realistic high/low
            high_noise = np.abs(np.random.normal(0, volatility/2, bars))
            low_noise = np.abs(np.random.normal(0, volatility/2, bars))
            
            df['high'] = df['close'] * (1 + high_noise)
            df['low'] = df['close'] * (1 - low_noise)
            
            # Ensure high >= low
            df['high'] = np.maximum(df['high'], df['low'])
            
            # Generate volume
            df['tick_volume'] = np.random.randint(100, 1000, bars)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating synthetic data for {symbol}: {e}")
            return None
    
    def prepare_scalping_data(self, symbol: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Prepare data for scalping neural network training
        """
        try:
            # Get real-time data from MT5
            data = self.get_real_time_data(symbol, mt5.TIMEFRAME_M1, 5000)
            if data is None:
                return np.array([]), np.array([]), {}
            
            # Create features
            features = self.feature_engine.create_scalping_features(data, symbol)
            
            if features.empty:
                return np.array([]), np.array([]), {}
            
            # Create scalping labels (BUY/SELL/HOLD)
            labels = self._create_scalping_labels(data, features)
            
            # Align features and labels
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels[:min_len]
            
            # Remove rows with NaN values
            valid_mask = ~(features.isna().any(axis=1) | np.isnan(labels))
            features = features[valid_mask].values
            labels = labels[valid_mask]
            
            if len(features) == 0:
                return np.array([]), np.array([]), {}
            
            logger.info(f"Prepared scalping data for {symbol}: {len(features)} samples, {features.shape[1]} features")
            
            metadata = {
                'symbol': symbol,
                'feature_count': features.shape[1],
                'sample_count': len(features),
                'data_period': f"{data.index[0]} to {data.index[-1]}",
                'scalping_labels': np.bincount(labels.astype(int))
            }
            
            return features, labels, metadata
            
        except Exception as e:
            logger.error(f"Error preparing scalping data for {symbol}: {e}")
            return np.array([]), np.array([]), {}
    
    def _create_scalping_labels(self, data: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
        """
        Create scalping labels based on short-term price movements
        Focus: Quick profit opportunities (5-15 minute windows)
        """
        try:
            # Future price movements for different timeframes
            future_5m = data['close'].shift(-5)  # 5 minutes ahead
            future_10m = data['close'].shift(-10)  # 10 minutes ahead
            future_15m = data['close'].shift(-15)  # 15 minutes ahead
            
            # Current price
            current_price = data['close']
            
            # Calculate potential profits/losses
            profit_5m = (future_5m - current_price) / current_price
            profit_10m = (future_10m - current_price) / current_price
            profit_15m = (future_15m - current_price) / current_price
            
            # Scalping thresholds (adjust based on symbol)
            scalping_threshold = 0.0008  # 8 pips for major pairs, adjust for others
            
            # Create labels
            labels = np.zeros(len(data))
            
            # BUY signals
            buy_condition = (
                (profit_5m > scalping_threshold) |
                (profit_10m > scalping_threshold * 1.5) |
                (profit_15m > scalping_threshold * 2.0)
            )
            labels[buy_condition] = 1  # BUY
            
            # SELL signals
            sell_condition = (
                (profit_5m < -scalping_threshold) |
                (profit_10m < -scalping_threshold * 1.5) |
                (profit_15m < -scalping_threshold * 2.0)
            )
            labels[sell_condition] = 2  # SELL
            
            # HOLD for everything else (0)
            
            return labels.astype(int)
            
        except Exception as e:
            logger.error(f"Error creating scalping labels: {e}")
            return np.zeros(len(data)).astype(int)
    
    def train_scalping_model(self, epochs: int = 100, batch_size: int = 64):
        """
        Train the ultimate scalping neural network on all pairs
        """
        logger.info("Starting ultimate scalping model training...")
        
        all_features = []
        all_labels = []
        metadata_summary = []
        
        # Collect training data from all pairs
        for symbol in self.trading_pairs:
            logger.info(f"Preparing data for {symbol}...")
            features, labels, metadata = self.prepare_scalping_data(symbol)
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                metadata_summary.append(metadata)
                logger.info(f"{symbol}: {len(features)} samples")
        
        if not all_features:
            logger.error("No training data available")
            return False
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        logger.info(f"Combined training data: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Label distribution: BUY={np.sum(y==1)}, SELL={np.sum(y==2)}, HOLD={np.sum(y==0)}")
        
        # Create model
        self.model = UltraFastNeuralNetwork(input_size=X.shape[1])
        
        # Prepare data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs['prediction'], batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.training_history.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        # Save model
        self.save_model("ultimate_scalping_model.pkl")
        
        logger.info("Ultimate scalping model training completed!")
        return True
    
    def predict_scalping_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Generate scalping signal for a symbol
        """
        try:
            if not self.is_trained:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Model not trained'}
            
            # Get latest data
            data = self.get_real_time_data(symbol, mt5.TIMEFRAME_M1, 100)
            if data is None:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No data available'}
            
            # Create features
            features = self.feature_engine.create_scalping_features(data, symbol)
            
            if features.empty:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Feature creation failed'}
            
            # Get latest features
            latest_features = features.iloc[-1:].values
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(torch.FloatTensor(latest_features))
                
                # Get prediction
                prediction = outputs['prediction'].numpy()[0]
                confidence = outputs['confidence'].numpy()[0][0]
                risk_score = outputs['risk_score'].numpy()[0][0]
                
                # Convert to action
                action_idx = np.argmax(prediction)
                actions = ['HOLD', 'BUY', 'SELL']
                action = actions[action_idx]
                
                # Calculate probability
                probs = torch.softmax(torch.FloatTensor(prediction), dim=-1).numpy()
                probability = probs[action_idx]
                
                return {
                    'action': action,
                    'confidence': confidence,
                    'probability': probability,
                    'risk_score': risk_score,
                    'all_probabilities': {
                        'HOLD': probs[0],
                        'BUY': probs[1],
                        'SELL': probs[2]
                    },
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'scalping_config': self.scalping_config
                }
                
        except Exception as e:
            logger.error(f"Error predicting scalping signal for {symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def save_model(self, filepath: str):
        """Save the trained model as PKL"""
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False
            
            model_data = {
                'model': self.model.state_dict(),
                'training_history': self.training_history,
                'is_trained': self.is_trained,
                'trading_pairs': self.trading_pairs,
                'scalping_config': self.scalping_config,
                'feature_engine': self.feature_engine,
                'training_metadata': {
                    'pairs': self.trading_pairs,
                    'total_features': len(self.trading_pairs) * 50,  # Estimated
                    'training_time': datetime.now()
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Ultimate scalping model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str):
        """Load the trained model from PKL"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = UltraFastNeuralNetwork(input_size=model_data['model']['network.0.weight'].shape[1])
            self.model.load_state_dict(model_data['model'])
            self.training_history = model_data['training_history']
            self.is_trained = model_data['is_trained']
            self.trading_pairs = model_data['trading_pairs']
            self.scalping_config = model_data['scalping_config']
            self.feature_engine = model_data['feature_engine']
            
            logger.info(f"Ultimate scalping model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_all_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Get scalping signals for all trading pairs
        """
        signals = {}
        for symbol in self.trading_pairs:
            signal = self.predict_scalping_signal(symbol)
            signals[symbol] = signal
        
        return signals

def main():
    """Main training and testing function"""
    # Initialize the ultimate scalping system
    system = UltimateNeuralScalpingSystem()
    
    # Train the model
    logger.info("Starting ultimate neural scalping system training...")
    success = system.train_scalping_model(epochs=50, batch_size=32)
    
    if success:
        logger.info("Training completed successfully!")
        
        # Test signals for all pairs
        signals = system.get_all_signals()
        
        print("\n" + "="*60)
        print("ULTIMATE NEURAL SCALPING SIGNALS")
        print("="*60)
        
        for symbol, signal in signals.items():
            print(f"\n{symbol}:")
            print(f"  Action: {signal['action']}")
            print(f"  Confidence: {signal['confidence']:.3f}")
            print(f"  Probability: {signal['probability']:.3f}")
            print(f"  Risk Score: {signal['risk_score']:.3f}")
        
        # Save model
        system.save_model("ultimate_scalping_neural_network.pkl")
        print(f"\nModel saved as 'ultimate_scalping_neural_network.pkl'")
        
    else:
        logger.error("Training failed!")

if __name__ == "__main__":
    main()
