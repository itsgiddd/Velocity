#!/usr/bin/env python3
"""
Enhanced Live Trading Bot with Continuous Learning Integration
============================================================

Advanced trading bot that:
1. Uses continuously trained neural networks
2. Implements frequent trading capabilities
3. Minimizes losses through advanced risk management
4. Adapts to market changes in real-time
5. Monitors performance and self-improves
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import json
import time
import threading
import warnings
from pathlib import Path
from enum import Enum
import builtins as _builtins
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def print(*args, **kwargs):  # noqa: A001
    """Console-safe print for Windows code pages that cannot encode emoji."""
    try:
        _builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = [str(a).encode("ascii", "ignore").decode() for a in args]
        _builtins.print(*safe_args, **kwargs)

@dataclass
class EnhancedTradingConfig:
    """Enhanced trading configuration"""
    # Basic trading parameters
    confidence_threshold: float = 0.75  # Higher threshold for better accuracy
    max_risk_per_trade: float = 0.015    # Reduced risk for frequent trading
    max_concurrent_positions: int = 10   # Allow more frequent trading
    
    # Neural network parameters
    model_path: str = "trained_neural_model.pth"
    feature_dim: int = 23
    
    # Frequent trading settings
    min_time_between_trades: int = 30     # 30 seconds minimum
    max_trades_per_hour: int = 20         # High frequency capability
    daily_trade_limit: int = 100          # Allow many trades per day
    
    # Risk management
    max_daily_loss: float = 0.05         # 5% max daily loss
    position_size_factor: float = 1.5     # Larger positions for better returns
    correlation_limit: float = 0.7         # Limit correlated trades
    
    # Performance targets
    target_win_rate: float = 0.82        # Higher target win rate
    min_profit_factor: float = 1.5         # Minimum profit factor
    max_drawdown: float = 0.03            # 3% max drawdown

class TradingMode(Enum):
    """Trading mode enumeration"""
    DEMO = "demo"
    LIVE = "live"

class TradeResult(Enum):
    """Trade result enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    """Enhanced trading signal with confidence and risk assessment"""
    symbol: str
    action: TradeResult
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_score: float
    expected_profit_factor: float
    timeframe_consensus: float
    market_condition: str
    reason: str

class EnhancedNeuralPredictor:
    """Enhanced neural predictor using trained models"""
    
    def __init__(self, config: EnhancedTradingConfig):
        self.config = config
        self.model = None
        self.model_input_dim = int(config.feature_dim)
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self._feature_engineer = None
        self.is_trained = False
        self.last_training_time = None
        
    def load_trained_model(self, model_path: str = None):
        """Load the best trained neural network model"""
        try:
            candidates = [
                model_path or self.config.model_path,
                "trained_neural_model.pth",
                "current_neural_model.pth",
            ]
            unique_candidates = list(dict.fromkeys(candidates))

            resolved_path = None
            for candidate in unique_candidates:
                if candidate and Path(candidate).exists():
                    resolved_path = candidate
                    break

            if resolved_path is None:
                print(f"Model file not found in: {unique_candidates}")
                return False

            checkpoint = torch.load(resolved_path, map_location='cpu')

            from mt5_neural_training_system import AdvancedNeuralNetwork

            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint does not contain a valid model state_dict")

            inferred_input_dim = checkpoint.get('input_dim')
            if inferred_input_dim is None and 'input_layer.weight' in state_dict:
                inferred_input_dim = int(state_dict['input_layer.weight'].shape[1])
            if inferred_input_dim is None:
                inferred_input_dim = self.config.feature_dim

            self.model_input_dim = int(inferred_input_dim)
            self.config.feature_dim = self.model_input_dim

            self.model = AdvancedNeuralNetwork(
                input_dim=self.model_input_dim,
                hidden_dim=256,
                num_layers=3
            )

            self.model.load_state_dict(state_dict)

            self.model.eval()
            self.is_trained = True

            if 'last_update' in checkpoint:
                self.last_training_time = checkpoint['last_update']

            raw_mean = checkpoint.get('feature_mean')
            raw_std = checkpoint.get('feature_std')
            if raw_mean is not None and raw_std is not None:
                self.feature_mean = np.asarray(raw_mean, dtype=np.float32).reshape(-1)
                self.feature_std = np.asarray(raw_std, dtype=np.float32).reshape(-1)
                if len(self.feature_mean) != self.model_input_dim or len(self.feature_std) != self.model_input_dim:
                    print("Feature scaler shape mismatch in checkpoint; ignoring stored scaler")
                    self.feature_mean = None
                    self.feature_std = None
                else:
                    self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)

            print(f"Loaded trained neural model from {resolved_path} (input_dim={self.model_input_dim})")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def _get_feature_engineer(self):
        """Lazy-load the exact feature engineer used during training."""
        if self._feature_engineer is None:
            from mt5_neural_training_system import AdvancedFeatureEngineer, TrainingConfig
            self._feature_engineer = AdvancedFeatureEngineer(TrainingConfig())
        return self._feature_engineer

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            numeric = float(value)
            if np.isnan(numeric) or np.isinf(numeric):
                return default
            return numeric
        except Exception:
            return default

    def _apply_feature_scaler(self, features: np.ndarray) -> np.ndarray:
        """Apply training-time feature scaling when available."""
        arr = np.asarray(features, dtype=np.float32)
        if self.feature_mean is None or self.feature_std is None:
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if len(arr) != len(self.feature_mean):
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = (arr - self.feature_mean) / self.feature_std
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal using the trained neural network"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            # Extract features from market data
            features = self._extract_enhanced_features(market_data)
            
            if features is None:
                return None
            
            # Make prediction
            with torch.no_grad():
                scaled_features = self._apply_feature_scaler(features)
                X_tensor = torch.FloatTensor(scaled_features.reshape(1, -1))
                outputs = self.model(X_tensor)
                
                # Extract predictions
                direction_probs = torch.softmax(outputs['direction'], dim=1).numpy()[0]
                confidence = float(outputs['confidence'].numpy()[0][0])
                risk_score = float(outputs['risk'].numpy()[0][0])
                
                # Determine action
                action_idx = int(np.argmax(direction_probs))
                actions = [TradeResult.SELL, TradeResult.HOLD, TradeResult.BUY]
                action = actions[action_idx]
                if action == TradeResult.HOLD:
                    return None
                action_probability = float(direction_probs[action_idx])
                trade_score = float(confidence * action_probability * (1.0 - 0.5 * risk_score))
                
                # Only proceed if confidence is high enough
                if confidence < self.config.confidence_threshold:
                    return None
                if action_probability < 0.45:
                    return None
                if trade_score < (self.config.confidence_threshold * 0.55):
                    return None
                
                # Create trading signal
                signal = self._create_trading_signal(
                    market_data, action, confidence, risk_score, direction_probs, action_probability, trade_score
                )
                
                return signal
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def _extract_enhanced_features(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features using the same schema as mt5_neural_training_system."""
        try:
            timeframes = market_data['timeframes']
            if 'M15' not in timeframes:
                return None

            m15_data = timeframes['M15'].dropna().copy()
            feature_engineer = self._get_feature_engineer()
            lookback = int(feature_engineer.config.lookback_periods)

            if len(m15_data) < lookback + 1:
                return None

            i = len(m15_data) - 1
            window = m15_data.iloc[i - lookback:i]
            if len(window) < lookback:
                return None

            current_price = self._safe_float(m15_data.iloc[i]['close'], 0.0)
            prev_price = self._safe_float(m15_data.iloc[i - 1]['close'], current_price)
            if current_price <= 0 or prev_price <= 0:
                return None

            feature_vector = []

            # Price features
            feature_vector.extend([
                self._safe_float(current_price / (prev_price + 1e-8) - 1.0),
                self._safe_float((current_price - window['close'].mean()) / (window['close'].std() + 1e-8)),
                self._safe_float(
                    (current_price - window['close'].min()) /
                    ((window['close'].max() - window['close'].min()) + 1e-8)
                ),
            ])

            # Technical indicator features (same order as trainer)
            indicators = ['sma_5', 'sma_10', 'sma_20', 'ema_12', 'macd', 'rsi', 'stoch_k', 'bb_position']
            for indicator in indicators:
                if indicator in window.columns:
                    value = self._safe_float(window[indicator].iloc[-1], 0.0)
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
                self._safe_float(returns.mean(), 0.0),
                self._safe_float(returns.std(), 0.0),
                self._safe_float(returns.skew(), 0.0),
                self._safe_float(returns.kurtosis(), 0.0),
            ])

            # Volume feature
            if 'tick_volume' in window.columns:
                volume_ratio = window['tick_volume'].iloc[-1] / (window['tick_volume'].mean() + 1e-8)
                feature_vector.append(float(np.clip(self._safe_float(volume_ratio, 1.0), 0.0, 10.0)))
            else:
                feature_vector.append(1.0)

            # Shared helper features from the training pipeline
            feature_vector.extend(feature_engineer._calculate_trend_strength(window))
            feature_vector.extend(feature_engineer._extract_pattern_features(window))
            feature_vector.extend(feature_engineer._calculate_risk_features(window))

            cleaned_features = [
                float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
                for value in feature_vector
            ]

            target_dim = int(self.model_input_dim or self.config.feature_dim)
            if len(cleaned_features) < target_dim:
                cleaned_features.extend([0.0] * (target_dim - len(cleaned_features)))

            return np.asarray(cleaned_features[:target_dim], dtype=np.float32)

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def _create_trading_signal(self, market_data: Dict[str, Any], action: TradeResult,
                             confidence: float, risk_score: float,
                             direction_probs: np.ndarray, action_probability: float,
                             trade_score: float) -> TradingSignal:
        """Create enhanced trading signal with full risk assessment"""
        symbol = market_data['symbol']
        m15_data = market_data['timeframes']['M15']
        
        current_price = self._safe_float(m15_data['close'].iloc[-1], 0.0)

        if 'atr' in m15_data.columns and not pd.isna(m15_data['atr'].iloc[-1]):
            atr = self._safe_float(m15_data['atr'].iloc[-1], 0.0)
        else:
            atr = self._safe_float((m15_data['high'] - m15_data['low']).rolling(14).mean().iloc[-1], 0.0)

        # ATR-based distances scale much better across different FX pairs.
        min_distance = max(current_price * 0.0002, 1e-6)
        base_distance = max(atr, min_distance)
        sl_distance = base_distance * (1.0 + 0.5 * risk_score)
        tp_distance = sl_distance * (1.6 + confidence * 0.8)

        if action == TradeResult.BUY:
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        else:  # SELL
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance
        
        # Calculate position size based on risk score
        base_position_size = self.config.max_risk_per_trade
        adjusted_position_size = (
            base_position_size *
            float(np.clip(0.4 + trade_score, 0.2, 1.5)) *
            float(np.clip(1.0 - 0.5 * risk_score, 0.2, 1.0))
        )
        
        # Expected profit factor
        sl_distance = abs(current_price - stop_loss)
        tp_distance = abs(take_profit - current_price)
        expected_profit_factor = tp_distance / sl_distance if sl_distance > 0 else 1.0
        
        # Market condition assessment
        market_condition = self._assess_market_condition(m15_data)
        
        # Timeframe consensus
        timeframe_consensus = float(action_probability)
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=adjusted_position_size,
            risk_score=risk_score,
            expected_profit_factor=expected_profit_factor,
            timeframe_consensus=timeframe_consensus,
            market_condition=market_condition,
            reason=(
                f"Neural {action.value} | conf {confidence:.1%} | "
                f"prob {action_probability:.1%} | score {trade_score:.3f}"
            )
        )
    
    def _assess_market_condition(self, m15_data: pd.DataFrame) -> str:
        """Assess current market condition"""
        try:
            # Simple market condition assessment
            volatility = m15_data['close'].pct_change().std()
            trend_strength = abs(m15_data['close'].iloc[-1] - m15_data['close'].rolling(20).mean().iloc[-1])
            
            if volatility > 0.02:  # High volatility
                return "VOLATILE"
            elif trend_strength > 0.001:  # Strong trend
                return "TRENDING"
            else:
                return "RANGING"
                
        except:
            return "UNKNOWN"

class AdvancedRiskManager:
    """Advanced risk management for frequent trading"""
    
    def __init__(self, config: EnhancedTradingConfig):
        self.config = config
        self.daily_stats = {
            'trades_count': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'start_balance': 0.0,
            'peak_balance': 0.0
        }
        self.last_trade_time = {}
        self.open_positions = []
        
    def can_trade(self, symbol: str, signal: TradingSignal, account_info: Any) -> Tuple[bool, str]:
        """Check if we can trade based on risk management rules"""
        # Check daily limits
        if self.daily_stats['trades_count'] >= self.config.daily_trade_limit:
            return False, "Daily trade limit reached"
        
        # Check time between trades
        current_time = time.time()
        if symbol in self.last_trade_time:
            time_diff = current_time - self.last_trade_time[symbol]
            if time_diff < self.config.min_time_between_trades:
                return False, f"Too soon since last trade on {symbol}"
        
        # Check concurrent positions
        if len(self.open_positions) >= self.config.max_concurrent_positions:
            return False, "Maximum concurrent positions reached"
        
        # Check daily loss limit
        if self.daily_stats['total_pnl'] <= -self.config.max_daily_loss * account_info.balance:
            return False, "Daily loss limit reached"
        
        # Check correlation with existing positions
        if self._has_correlated_position(signal.symbol):
            return False, "Correlated position already open"
        
        # Check profit factor
        if signal.expected_profit_factor < self.config.min_profit_factor:
            return False, "Insufficient profit factor"
        
        # Check confidence
        if signal.confidence < self.config.confidence_threshold:
            return False, "Insufficient confidence"
        
        return True, "Trade approved"
    
    def update_daily_stats(self, trade_result: float):
        """Update daily trading statistics"""
        self.daily_stats['trades_count'] += 1
        
        if trade_result > 0:
            self.daily_stats['winning_trades'] += 1
        else:
            self.daily_stats['losing_trades'] += 1
        
        self.daily_stats['total_pnl'] += trade_result
        
        # Update drawdown
        current_balance = self.daily_stats['start_balance'] + self.daily_stats['total_pnl']
        if current_balance > self.daily_stats['peak_balance']:
            self.daily_stats['peak_balance'] = current_balance
        
        self.daily_stats['current_drawdown'] = (
            (self.daily_stats['peak_balance'] - current_balance) / 
            self.daily_stats['peak_balance']
        )
        
        if self.daily_stats['current_drawdown'] > self.daily_stats['max_drawdown']:
            self.daily_stats['max_drawdown'] = self.daily_stats['current_drawdown']
    
    def _has_correlated_position(self, symbol: str) -> bool:
        """Check if there's a correlated position already open"""
        # Simplified correlation check
        # In practice, you'd check actual currency correlations
        for position in self.open_positions:
            if position['symbol'] == symbol:
                return True
        return False
    
    def get_win_rate(self) -> float:
        """Calculate current win rate"""
        if self.daily_stats['trades_count'] == 0:
            return 0.0
        return self.daily_stats['winning_trades'] / self.daily_stats['trades_count']

class EnhancedLiveTradingBot:
    """Enhanced live trading bot with continuous learning integration"""
    
    def __init__(self, trading_mode: TradingMode = TradingMode.DEMO):
        self.trading_mode = trading_mode
        self.config = EnhancedTradingConfig()
        self.is_running = False
        self.bot_thread = None
        
        # Initialize components
        self.neural_predictor = EnhancedNeuralPredictor(self.config)
        self.risk_manager = AdvancedRiskManager(self.config)
        
        # MT5 connection
        self.mt5_initialized = False
        
        # Performance tracking
        self.start_time = None
        self.symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY', 'CADCHF',
            'BTCUSD'
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _resolve_symbol(self, symbol: str) -> str:
        """Resolve requested symbol to broker-available symbol (e.g., BTCUSD -> BTC)."""
        requested = str(symbol or "").strip()
        if not requested:
            return requested

        info = mt5.symbol_info(requested)
        if info is not None:
            return requested

        symbols = mt5.symbols_get() or []
        names = [s.name for s in symbols]
        upper_names = {name.upper(): name for name in names}
        upper_requested = requested.upper()

        if upper_requested in upper_names:
            return upper_names[upper_requested]

        if upper_requested.endswith("USD"):
            base = upper_requested[:-3]
            if base in upper_names:
                return upper_names[base]
            starts = [name for name in names if name.upper().startswith(base)]
            if starts:
                return sorted(starts, key=len)[0]

        partial = [name for name in names if upper_requested in name.upper()]
        if partial:
            return sorted(partial, key=len)[0]

        return requested
        
    def initialize(self):
        """Initialize the trading bot"""
        print("🚀 Initializing Enhanced Live Trading Bot...")
        
        # Initialize MT5
        if not self._initialize_mt5():
            return False
        
        # Load trained neural model
        if not self.neural_predictor.load_trained_model():
            print("No trained neural model available; aborting startup")
            return False
        
        # Initialize risk manager
        account_info = mt5.account_info()
        if account_info:
            self.risk_manager.daily_stats['start_balance'] = account_info.balance
            self.risk_manager.daily_stats['peak_balance'] = account_info.balance
        
        self.mt5_initialized = True
        print("✅ Enhanced trading bot initialized successfully")
        return True
    
    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            account_info = mt5.account_info()
            if not account_info:
                print("❌ Failed to get account info")
                return False
            
            print(f"✅ MT5 connected - Account: {account_info.login}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Mode: {self.trading_mode.value.upper()}")
            
            return True
            
        except Exception as e:
            print(f"❌ MT5 initialization error: {e}")
            return False
    
    def start_trading(self):
        """Start the enhanced trading system"""
        if not self.mt5_initialized:
            print("❌ Bot not initialized. Call initialize() first.")
            return False
        
        if self.is_running:
            print("⚠️  Trading bot is already running")
            return False
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start trading thread
        self.bot_thread = threading.Thread(target=self._trading_loop)
        self.bot_thread.daemon = True
        self.bot_thread.start()
        
        print("🎯 Enhanced trading bot started!")
        print("   Features: Neural predictions, Frequent trading, Advanced risk management")
        print("   Target: 82%+ win rate with minimal losses")
        print("   Press Ctrl+C to stop")
        
        return True
    
    def stop_trading(self):
        """Stop the trading bot"""
        self.is_running = False
        
        if self.bot_thread:
            self.bot_thread.join(timeout=5)
        
        print("🛑 Enhanced trading bot stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        print("🔄 Starting enhanced trading loop...")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Get account info
                account_info = mt5.account_info()
                if not account_info:
                    print("⚠️  Lost MT5 connection, attempting to reconnect...")
                    if not self._initialize_mt5():
                        time.sleep(60)
                        continue
                    account_info = mt5.account_info()
                
                # Get open positions
                open_positions = mt5.positions_get()
                self.risk_manager.open_positions = [
                    {'symbol': pos.symbol, 'type': pos.type} 
                    for pos in open_positions
                ] if open_positions else []
                
                # Analyze each symbol
                for symbol in self.symbols:
                    if not self.is_running:
                        break
                    
                    try:
                        # Get market data
                        market_data = self._get_market_data(symbol)
                        if market_data is None:
                            continue
                        
                        # Generate neural signal
                        signal = self.neural_predictor.predict_signal(market_data)
                        
                        if signal:
                            # Check if we can trade
                            can_trade, reason = self.risk_manager.can_trade(
                                symbol, signal, account_info
                            )
                            
                            if can_trade:
                                # Execute trade
                                success = self._execute_enhanced_trade(signal, account_info)
                                if success:
                                    self.risk_manager.last_trade_time[symbol] = time.time()
                                    print(f"✅ {signal.symbol}: {signal.action.value} @ {signal.entry_price:.5f}")
                                    print(f"   Confidence: {signal.confidence:.1%}, Risk: {signal.risk_score:.3f}")
                            else:
                                print(f"⏭️  {symbol}: {reason}")
                        
                    except Exception as e:
                        print(f"❌ Error processing {symbol}: {e}")
                        continue
                
                # Update performance stats
                self._update_performance_stats()
                
                # Sleep before next cycle
                time.sleep(10)  # 10-second cycles for high frequency
                
        except Exception as e:
            print(f"❌ Critical error in trading loop: {e}")
    
    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data for analysis"""
        try:
            timeframes = {}
            resolved_symbol = self._resolve_symbol(symbol)
            try:
                mt5.symbol_select(resolved_symbol, True)
            except Exception:
                pass

            symbol_info = mt5.symbol_info(resolved_symbol)
            spread_points = float(getattr(symbol_info, 'spread', 0.0)) if symbol_info else 0.0
            point = float(getattr(symbol_info, 'point', 0.0)) if symbol_info else 0.0
            
            # Get data for different timeframes
            for tf_name, tf in [('M5', mt5.TIMEFRAME_M5), ('M15', mt5.TIMEFRAME_M15), 
                               ('H1', mt5.TIMEFRAME_H1), ('H4', mt5.TIMEFRAME_H4)]:
                rates = mt5.copy_rates_from_pos(resolved_symbol, tf, 0, 250)
                if rates is None or len(rates) < 120:
                    continue
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Add indicators aligned with mt5_neural_training_system.
                df['sma_5'] = df['close'].rolling(5).mean()
                df['sma_10'] = df['close'].rolling(10).mean()
                df['sma_20'] = df['close'].rolling(20).mean()
                df['ema_12'] = df['close'].ewm(span=12).mean()
                df['ema_26'] = df['close'].ewm(span=26).mean()
                df['macd'] = df['ema_12'] - df['ema_26']
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']

                df['bb_middle'] = df['close'].rolling(20).mean()
                bb_std = df['close'].rolling(20).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

                df['rsi'] = self._calculate_rsi(df['close'])
                low_min = df['low'].rolling(14).min()
                high_max = df['high'].rolling(14).max()
                df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
                df['stoch_d'] = df['stoch_k'].rolling(3).mean()

                df['body'] = abs(df['close'] - df['open'])
                df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
                df['total_range'] = df['high'] - df['low']
                df['atr'] = df['total_range'].rolling(14).mean()
                df['volatility'] = df['close'].rolling(20).std()

                df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
                df['resistance1'] = 2 * df['pivot'] - df['low']
                df['support1'] = 2 * df['pivot'] - df['high']

                if point > 0 and spread_points > 0:
                    df['spread'] = (spread_points * point) / (df['close'] + 1e-8)
                else:
                    df['spread'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
                
                timeframes[tf_name] = df
            
            if not timeframes:
                return None
            if 'M15' not in timeframes:
                return None
             
            return {
                'symbol': resolved_symbol,
                'requested_symbol': symbol,
                'timeframes': timeframes,
                'current_price': timeframes['M15']['close'].iloc[-1]
            }
            
        except Exception as e:
            print(f"❌ Error getting market data for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bb_position(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Bollinger Bands position"""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (df['close'] - lower) / (upper - lower)
    
    def _execute_enhanced_trade(self, signal: TradingSignal, account_info) -> bool:
        """Execute trade with enhanced risk management"""
        try:
            # Calculate lot size
            lot_size = self._calculate_position_size(signal, account_info)
            
            if lot_size <= 0:
                return False
            
            # Prepare MT5 order
            symbol_info = mt5.symbol_info(signal.symbol)
            if not symbol_info:
                return False
            
            if signal.action == TradeResult.BUY:
                order_type = mt5.ORDER_TYPE_BUY
                price = symbol_info.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = symbol_info.bid
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Neural-{signal.confidence:.1%}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"🎯 Enhanced trade executed: {signal.symbol} {signal.action.value}")
                print(f"   Entry: {price:.5f}, SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}")
                print(f"   Lot size: {lot_size:.2f}, Confidence: {signal.confidence:.1%}")
                return True
            else:
                print(f"❌ Trade failed: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            return False
    
    def _calculate_position_size(self, signal: TradingSignal, account_info) -> float:
        """Calculate position size based on risk and neural confidence"""
        symbol_info = mt5.symbol_info(signal.symbol)
        if not symbol_info:
            return 0

        balance = float(account_info.balance)
        risk_fraction = float(np.clip(signal.position_size, 0.0, self.config.max_risk_per_trade * 2.0))
        risk_amount = balance * risk_fraction

        tick_size = symbol_info.trade_tick_size if symbol_info.trade_tick_size > 0 else symbol_info.point
        tick_value = symbol_info.trade_tick_value if symbol_info.trade_tick_value > 0 else 1.0
        sl_ticks = abs(signal.entry_price - signal.stop_loss) / (tick_size + 1e-12)
        if sl_ticks <= 0:
            return 0

        loss_per_lot = sl_ticks * tick_value
        if loss_per_lot <= 0:
            return 0

        lot_size = risk_amount / loss_per_lot
        lot_size *= (0.5 + signal.confidence * 0.5)

        step = symbol_info.volume_step if symbol_info.volume_step > 0 else 0.01
        min_vol = symbol_info.volume_min if symbol_info.volume_min > 0 else 0.01
        max_vol = symbol_info.volume_max if symbol_info.volume_max > 0 else 100.0

        lot_size = round(lot_size / step) * step
        lot_size = max(min_vol, min(lot_size, max_vol))

        return round(float(lot_size), 2)
    
    def _update_performance_stats(self):
        """Update and display performance statistics"""
        if not hasattr(self, '_last_stats_update'):
            self._last_stats_update = time.time()
        
        # Update every 60 seconds
        if time.time() - self._last_stats_update < 60:
            return
        
        self._last_stats_update = time.time()
        
        win_rate = self.risk_manager.get_win_rate()
        daily_pnl = self.risk_manager.daily_stats['total_pnl']
        trades_count = self.risk_manager.daily_stats['trades_count']
        
        print(f"\n📊 Performance Update:")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Daily P&L: ${daily_pnl:.2f}")
        print(f"   Trades Today: {trades_count}")
        print(f"   Current Drawdown: {self.risk_manager.daily_stats['current_drawdown']:.1%}")

def main():
    """Main function to run enhanced trading bot"""
    print("🚀 Enhanced Live Trading Bot with Continuous Learning")
    print("=" * 60)
    
    # Initialize bot
    bot = EnhancedLiveTradingBot(trading_mode=TradingMode.DEMO)
    
    if not bot.initialize():
        print("❌ Failed to initialize trading bot")
        return False
    
    # Start trading
    if bot.start_trading():
        try:
            # Keep main thread alive
            while bot.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping enhanced trading bot...")
            bot.stop_trading()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)


