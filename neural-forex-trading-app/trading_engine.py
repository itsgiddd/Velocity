#!/usr/bin/env python3
"""
Neural Trading Engine
====================

Professional trading engine that integrates neural network predictions
with MT5 trading operations for automated forex trading.

Features:
- Neural network signal generation
- Automated trade execution
- Risk management
- Position monitoring
- Performance tracking
- Real-time trading loop
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Import app modules
from mt5_connector import MT5Connector
from model_manager import NeuralModelManager

class TradingSignal:
    """Trading signal data structure"""
    
    def __init__(self, symbol: str, action: str, confidence: float, 
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float, reason: str):
        self.symbol = symbol
        self.action = action  # 'BUY', 'SELL', 'HOLD'
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.reason = reason
        self.timestamp = datetime.now()
        self.executed = False
        self.order_ticket = None

class Position:
    """Position tracking data structure"""
    
    def __init__(self, ticket: int, symbol: str, action: str, 
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float):
        self.ticket = ticket
        self.symbol = symbol
        self.action = action
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.open_time = datetime.now()
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.status = 'OPEN'  # OPEN, CLOSED, PARTIAL

class TradingEngine:
    """Professional neural trading engine"""
    
    def __init__(self, mt5_connector: MT5Connector, model_manager: NeuralModelManager,
                 risk_per_trade: float = 0.015, confidence_threshold: float = 0.65,
                 trading_pairs: List[str] = None, max_concurrent_positions: int = 5):
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.mt5_connector = mt5_connector
        self.model_manager = model_manager
        
        # Trading parameters
        self.risk_per_trade = risk_per_trade  # 1.5% default
        self.confidence_threshold = confidence_threshold  # 65% default
        self.trading_pairs = trading_pairs or ['USDJPY']  # Only USDJPY
        self.max_concurrent_positions = max_concurrent_positions
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.positions: Dict[int, Position] = {}
        self.signals_history: List[TradingSignal] = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }
        
        # Feature engineering cache
        self.feature_cache = {}
        self.last_update = {}
        
        # Timeframes for analysis
        self.timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1
        }
        
        # Performance tracking
        self.start_time = None
        self.last_performance_update = None
        
        self.logger.info("Neural Trading Engine initialized")
    
    def start(self):
        """Start the trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine already running")
            return
        
        # Validate prerequisites
        if not self.mt5_connector.is_connected():
            raise Exception("MT5 not connected")
        
        if not self.model_manager.is_model_loaded():
            raise Exception("Neural model not loaded")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        self.logger.info("Neural Trading Engine started")
    
    def stop(self):
        """Stop the trading engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for trading thread to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        
        self.logger.info("Neural Trading Engine stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting trading loop")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Update positions
                self._update_positions()
                
                # Generate signals for trading pairs
                for symbol in self.trading_pairs:
                    if not self.is_running:
                        break
                    
                    try:
                        signal = self._generate_signal(symbol)
                        if signal:
                            self._process_signal(signal)
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Update performance metrics
                if self.last_performance_update is None or \
                   (current_time - self.last_performance_update).seconds >= 60:
                    self._update_performance_metrics()
                    self.last_performance_update = current_time
                
                # Sleep before next iteration
                time.sleep(5)  # 5-second cycles
                
        except Exception as e:
            self.logger.error(f"Critical error in trading loop: {e}")
        finally:
            self.is_running = False
    
    def _generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate neural trading signal for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TradingSignal object or None
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            if not market_data:
                return None
            
            # Extract features
            features = self._extract_features(market_data)
            if features is None:
                return None
            
            # Get neural prediction
            prediction = self.model_manager.predict(features)
            if not prediction:
                return None
            
            # Check confidence threshold
            if prediction['confidence'] < self.confidence_threshold:
                return None
            
            # Get symbol info for trading
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Calculate trading parameters
            entry_price = symbol_info['ask'] if prediction['action'] == 'BUY' else symbol_info['bid']
            stop_loss, take_profit = self._calculate_sl_tp(
                symbol, prediction['action'], entry_price, symbol_info
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, entry_price, stop_loss, symbol_info)
            
            if position_size <= 0:
                return None
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                action=prediction['action'],
                confidence=prediction['confidence'],
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"Neural prediction: {prediction['action']} ({prediction['confidence']:.1%} confidence)"
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data for a symbol"""
        try:
            market_data = {}
            
            # Get data for multiple timeframes
            for tf_name, tf_constant in self.timeframes.items():
                rates = self.mt5_connector.get_rates(symbol, tf_constant, 0, 100)
                if rates:
                    market_data[tf_name] = rates
                else:
                    self.logger.warning(f"No {tf_name} data for {symbol}")
                    return None
            
            # Get symbol info
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            if symbol_info:
                market_data['symbol_info'] = symbol_info
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _extract_features(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract 10 features for enhanced neural network"""
        try:
            # Use M15 as primary timeframe
            if 'M15' not in market_data:
                return None
            
            m15_data = market_data['M15']
            if len(m15_data) < 20:  # Reduced requirement for real-time trading
                self.logger.warning(f"Insufficient data: {len(m15_data)} points (need 20+)")
                return None
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(m15_data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Calculate comprehensive technical indicators (same as enhanced training)
            
            # 1. Price momentum (10-period)
            df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
            
            # 2. Z-score (price deviation from 20-period mean)
            df['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            
            # 3. SMA ratios
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_15'] = df['close'].rolling(15).mean()
            df['sma_5_ratio'] = df['sma_5'] / df['close'] - 1
            df['sma_20_ratio'] = df['sma_15'] / df['close'] - 1
            
            # 4. RSI
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # 5. Volatility (annualized)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(10).std() * np.sqrt(252)
            
            # 6. Trend strength (10-period)
            df['trend_strength'] = df['close'] / df['close'].shift(10) - 1
            
            # 7. Bollinger Bands position
            df['bb_middle'] = df['close'].rolling(15).mean()
            df['bb_std'] = df['close'].rolling(15).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 8-10. Candlestick Pattern Recognition
            df['trend_continuation_score'] = 0.0
            df['trend_reversal_score'] = 0.0
            
            # Calculate consecutive candles for pattern recognition
            for i in range(4, len(df)):
                consecutive_bullish = 0
                consecutive_bearish = 0
                
                # Count consecutive bullish candles
                for j in range(i, max(0, i-5), -1):
                    if df.iloc[j]['close'] > df.iloc[j]['open']:
                        consecutive_bullish += 1
                    else:
                        break
                
                # Count consecutive bearish candles  
                for j in range(i, max(0, i-5), -1):
                    if df.iloc[j]['close'] < df.iloc[j]['open']:
                        consecutive_bearish += 1
                    else:
                        break
                
                # Pattern scores (normalized)
                df.loc[i, 'trend_continuation_score'] = min(consecutive_bullish, consecutive_bearish, 4) / 4.0
                df.loc[i, 'trend_reversal_score'] = min(consecutive_bullish, consecutive_bearish, 3) / 3.0
            
            # Get latest values (remove NaN)
            latest = df.iloc[-1]
            
            # Handle NaN values with fallbacks
            required_features = ['price_momentum', 'price_zscore', 'sma_5_ratio', 'sma_20_ratio', 
                                'rsi', 'volatility', 'trend_strength', 'bb_position', 
                                'trend_continuation_score', 'trend_reversal_score']
            
            # Fill NaN values with reasonable defaults
            latest_filled = latest.copy()
            
            # Set defaults for missing indicators
            if pd.isna(latest['price_momentum']):
                latest_filled['price_momentum'] = 0.0
            if pd.isna(latest['price_zscore']):
                latest_filled['price_zscore'] = 0.0
            if pd.isna(latest['sma_5_ratio']):
                latest_filled['sma_5_ratio'] = 0.0
            if pd.isna(latest['sma_20_ratio']):
                latest_filled['sma_20_ratio'] = 0.0
            if pd.isna(latest['rsi']):
                latest_filled['rsi'] = 50.0  # Neutral RSI
            if pd.isna(latest['volatility']):
                latest_filled['volatility'] = 0.01  # Default volatility
            if pd.isna(latest['trend_strength']):
                latest_filled['trend_strength'] = 0.0
            if pd.isna(latest['bb_position']):
                latest_filled['bb_position'] = 0.5  # Middle of bands
            if pd.isna(latest['trend_continuation_score']):
                latest_filled['trend_continuation_score'] = 0.0
            if pd.isna(latest['trend_reversal_score']):
                latest_filled['trend_reversal_score'] = 0.0
            
            # Create 6-feature vector for neural model
            features = [
                latest_filled['price_momentum'],           # 1. Price momentum
                latest_filled['price_zscore'],              # 2. Z-score
                latest_filled['sma_5_ratio'],              # 3. SMA 5 ratio
                latest_filled['sma_20_ratio'],              # 4. SMA 20 ratio
                latest_filled['rsi'],                       # 5. RSI
                latest_filled['volatility']                 # 6. Volatility
            ]
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting 10 features: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_sl_tp(self, symbol: str, action: str, entry_price: float, 
                         symbol_info: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels - USDJPY 20-30 pip strategy"""
        try:
            # USDJPY pip-based calculation for 20-30 pip profit targets
            if 'JPY' in symbol:
                # JPY pairs: pip = 0.01
                pip_value = 0.01
            else:
                # Non-JPY pairs: pip = 0.0001  
                pip_value = 0.0001
            
            if action == 'BUY':
                # Quick scalping: 20 pip profit target, 15 pip stop loss
                take_profit = entry_price + (pip_value * 20)  # 20 pips profit
                stop_loss = entry_price - (pip_value * 15)   # 15 pips risk
            else:  # SELL
                take_profit = entry_price - (pip_value * 20)  # 20 pips profit
                stop_loss = entry_price + (pip_value * 15)   # 15 pips risk
            
            # Adjust rounding based on pair type
            if 'JPY' in symbol:
                stop_loss = round(stop_loss, 3)
                take_profit = round(take_profit, 3)
            else:
                stop_loss = round(stop_loss, 5)
                take_profit = round(take_profit, 5)
            
            self.logger.info(f"{symbol} {action}: Entry={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}")
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating SL/TP for {symbol}: {e}")
            return 0.0, 0.0
    
    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float,
                               symbol_info: Dict[str, Any]) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account info
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                return 0.0
            
            balance = account_info['balance']
            risk_amount = balance * self.risk_per_trade
            
            # Calculate pip value
            pip_value = 10 if 'JPY' not in symbol else 1
            
            # Calculate stop loss in pips
            sl_distance = abs(entry_price - stop_loss)
            sl_pips = sl_distance * (10000 if 'JPY' not in symbol else 100)
            
            if sl_pips == 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / (sl_pips * pip_value)
            
            # Apply symbol constraints
            volume_min = symbol_info.get('volume_min', 0.01)
            volume_max = symbol_info.get('volume_max', 100.0)
            volume_step = symbol_info.get('volume_step', 0.01)
            
            # Round to step size
            position_size = round(position_size / volume_step) * volume_step
            
            # Ensure within limits
            position_size = max(volume_min, min(volume_max, position_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def _process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        try:
            # Check if we can take this trade
            if not self._can_trade(signal):
                return
            
            # Execute trade
            order_result = self._execute_trade(signal)
            if order_result:
                signal.executed = True
                signal.order_ticket = order_result.get('order')
                
                # Create position tracking
                position = Position(
                    ticket=order_result['order'],
                    symbol=signal.symbol,
                    action=signal.action,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=signal.position_size
                )
                
                self.positions[position.ticket] = position
                self.signals_history.append(signal)
                
                self.logger.info(f"Trade executed: {signal.symbol} {signal.action} @ {signal.entry_price}")
                self.logger.info(f"SL: {signal.stop_loss}, TP: {signal.take_profit}")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    def _can_trade(self, signal: TradingSignal) -> bool:
        """Check if we can execute the trade"""
        # Check maximum concurrent positions
        if len(self.positions) >= self.max_concurrent_positions:
            return False
        
        # Check if we already have a position in this symbol
        for position in self.positions.values():
            if position.symbol == signal.symbol and position.status == 'OPEN':
                return False
        
        # Check risk parameters
        if signal.confidence < self.confidence_threshold:
            return False
        
        # Additional risk checks can be added here
        
        return True
    
    def _execute_trade(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """Execute the actual trade"""
        try:
            # Prepare order request
            symbol_info = self.mt5_connector.get_symbol_info(signal.symbol)
            if not symbol_info:
                return None
            
            # Determine order type and price
            if signal.action == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = symbol_info['ask']
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = symbol_info['bid']
            
            # Create order request
            order_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": signal.position_size,
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
            result = self.mt5_connector.send_order(order_request)
            
            if result and result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                return result
            else:
                self.logger.error(f"Trade execution failed: {result}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def _update_positions(self):
        """Update position information and check for exits"""
        try:
            # Get current positions from MT5
            mt5_positions = self.mt5_connector.get_positions()
            
            # Update our position tracking
            for ticket, position in list(self.positions.items()):
                # Find position in MT5
                mt5_pos = None
                for pos in mt5_positions:
                    if pos['ticket'] == ticket:
                        mt5_pos = pos
                        break
                
                if mt5_pos:
                    # Update position data
                    position.current_price = mt5_pos['price_current']
                    position.unrealized_pnl = mt5_pos['profit']
                    
                    # Check exit conditions
                    if self._should_close_position(position, mt5_pos):
                        self._close_position(position)
                else:
                    # Position no longer exists in MT5, mark as closed
                    position.status = 'CLOSED'
                    self.logger.info(f"Position {ticket} closed")
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _should_close_position(self, position: Position, mt5_pos: Dict[str, Any]) -> bool:
        """Simple pip-based exit - Close at TP or SL for USDJPY 20-30 pip strategy"""
        try:
            current_price = mt5_pos['price_current']
            
            # Check if take profit is reached (20 pips profit)
            if self._is_take_profit_hit(position, current_price):
                self.logger.info(f"Take profit reached for {position.symbol} - 20 pips profit!")
                return True
            
            # Check if stop loss is hit (15 pips loss)
            if self._is_stop_loss_hit(position, current_price):
                self.logger.info(f"Stop loss hit for {position.symbol} - 15 pips loss")
                return True
            
            # No other exit conditions - let the trade run to TP or SL
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def _is_stop_loss_hit(self, position: Position, current_price: float) -> bool:
        """Check if hard stop loss is hit"""
        if position.action == 'BUY':
            return current_price <= position.stop_loss
        else:  # SELL
            return current_price >= position.stop_loss
    
    def _is_take_profit_hit(self, position: Position, current_price: float) -> bool:
        """Check if take profit is hit"""
        if position.action == 'BUY':
            return current_price >= position.take_profit
        else:  # SELL
            return current_price <= position.take_profit
    
    def _calculate_profit_percentage(self, position: Position, current_price: float) -> float:
        """Calculate current profit percentage"""
        if position.action == 'BUY':
            return ((current_price - position.entry_price) / position.entry_price) * 100
        else:  # SELL
            return ((position.entry_price - current_price) / position.entry_price) * 100
    
    def _should_activate_trailing_stop(self, position: Position, current_price: float) -> bool:
        """Determine if trailing stop should be activated"""
        profit_pct = self._calculate_profit_percentage(position, current_price)
        
        # For trades in 0.5-1% profit, activate trailing stop
        if 0.5 <= profit_pct <= 1.0:
            # Move stop loss to break-even + small buffer
            buffer = 0.2 if position.action == 'BUY' else -0.2
            new_stop = position.entry_price + buffer
            
            if position.action == 'BUY':
                return current_price <= new_stop
            else:  # SELL
                return current_price >= new_stop
        
        # For trades with 1%+ profit, more aggressive trailing
        elif profit_pct > 1.0:
            # Trail behind current price by 0.3%
            trail_distance = 0.003
            if position.action == 'BUY':
                trail_stop = current_price - trail_distance
                return current_price <= trail_stop
            else:  # SELL
                trail_stop = current_price + trail_distance
                return current_price >= trail_stop
        
        return False
    
    def _should_exit_by_time(self, profit_pct: float, time_open: float) -> bool:
        """Determine if position should exit due to time stagnation"""
        # Exit logic based on time and profit level
        if time_open > 24:  # 24+ hours
            if profit_pct > 0.2:  # Small profit after 24h
                return True
        elif time_open > 12:  # 12+ hours
            if profit_pct > 0.5:  # Medium profit after 12h
                return True
        elif time_open > 6:  # 6+ hours
            if profit_pct < -0.1:  # Losing after 6h
                return True
        
        return False
    
    def _should_exit_profit_stagnation(self, position: Position, current_price: float) -> bool:
        """Exit if profit stagnates after significant movement"""
        try:
            # Get recent price history for this symbol
            market_data = self._get_recent_prices(position.symbol, 100)
            if not market_data:
                return False
            
            # Find the peak profit achieved
            peak_profit = 0.0
            for price_data in market_data:
                price = price_data['close']
                if position.action == 'BUY':
                    profit = (price - position.entry_price) / position.entry_price * 100
                else:  # SELL
                    profit = (position.entry_price - price) / position.entry_price * 100
                
                if profit > peak_profit:
                    peak_profit = profit
            
            # Current profit
            current_profit = self._calculate_profit_percentage(position, current_price)
            
            # If we've given back more than 60% of peak profit, exit
            profit_reduction = peak_profit - current_profit
            if peak_profit > 0.8 and profit_reduction > peak_profit * 0.6:
                self.logger.info(f"Profit stagnation exit: Peak {peak_profit:.2f}%, Current {current_profit:.2f}%")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in profit stagnation check: {e}")
            return False
    
    def _is_trade_dropping_after_peak(self, position: Position, current_price: float) -> bool:
        """Detect if trade has peaked and is dropping back down"""
        try:
            # Get price history to analyze peak detection
            market_data = self._get_recent_prices(position.symbol, 100)
            if not market_data or len(market_data) < 20:
                return False
            
            current_profit = self._calculate_profit_percentage(position, current_price)
            
            # Find the peak profit achieved during this trade
            peak_profit = 0.0
            peak_price = 0.0
            
            for i, price_data in enumerate(market_data[-50:]):  # Last 50 candles
                price = price_data['close']
                if position.action == 'BUY':
                    profit = (price - position.entry_price) / position.entry_price * 100
                else:  # SELL
                    profit = (position.entry_price - price) / position.entry_price * 100
                
                if profit > peak_profit:
                    peak_profit = profit
                    peak_price = price
            
            # Only consider if we're still in decent profit
            if current_profit < 0.5:  # Less than 0.5% profit
                return False
            
            # Check if we've dropped significantly from peak
            if peak_profit < 0.8:  # Peak was less than 0.8%
                return False
            
            # Calculate how much we've dropped from peak
            profit_drop = peak_profit - current_profit
            
            # Only trigger if we've given back at least 50% of peak profit
            # AND the current profit is still reasonable (not back to break-even)
            if profit_drop >= peak_profit * 0.5 and current_profit >= 0.3:
                self.logger.info(f"Peak detection: Peak {peak_profit:.2f}%, Current {current_profit:.2f}%, Drop {profit_drop:.2f}%")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in peak detection: {e}")
            return False
    
    def _should_exit_only_if_most_profit_given_back(self, position: Position, current_price: float) -> bool:
        """Very conservative - only exit if we've given back MOST of peak profit"""
        try:
            # Get price history
            market_data = self._get_recent_prices(position.symbol, 100)
            if not market_data or len(market_data) < 20:
                return False
            
            current_profit = self._calculate_profit_percentage(position, current_price)
            
            # Find the peak profit
            peak_profit = 0.0
            for price_data in market_data[-50:]:  # Last 50 candles
                price = price_data['close']
                if position.action == 'BUY':
                    profit = (price - position.entry_price) / position.entry_price * 100
                else:  # SELL
                    profit = (position.entry_price - price) / position.entry_price * 100
                
                if profit > peak_profit:
                    peak_profit = profit
            
            # Very conservative thresholds
            if peak_profit < 1.0:  # Only consider if peak was >1%
                return False
            
            # Only exit if we've given back 70% of peak profit AND current profit < 0.4%
            profit_remaining = peak_profit - current_profit
            if profit_remaining >= peak_profit * 0.7 and current_profit < 0.4:
                self.logger.info(f"Conservative exit: Peak {peak_profit:.2f}%, Current {current_profit:.2f}%, Given back {profit_remaining:.2f}%")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in conservative profit protection: {e}")
            return False
        """AGGRESSIVE trailing stop - protect profits much earlier"""
        profit_pct = self._calculate_profit_percentage(position, current_price)
        
        # AGGRESSIVE thresholds for profit protection
        if profit_pct >= 0.3:  # Any profit above 0.3%
            # Move to break-even IMMEDIATELY for any profit
            if profit_pct <= 0.8:  # For smaller profits
                buffer = 0.1 if position.action == 'BUY' else -0.1  # Smaller buffer
                new_stop = position.entry_price + buffer
                
                if position.action == 'BUY':
                    return current_price <= new_stop
                else:  # SELL
                    return current_price >= new_stop
            
            # For larger profits, trail very closely
            elif profit_pct > 0.8:
                # Trail behind by only 0.2% (very tight)
                trail_distance = 0.002
                if position.action == 'BUY':
                    trail_stop = current_price - trail_distance
                    return current_price <= trail_stop
                else:  # SELL
                    trail_stop = current_price + trail_distance
                    return current_price >= trail_stop
        
        return False
    
    def _should_exit_any_profit_if_trend_weak(self, position: Position, current_price: float) -> bool:
        """Exit with ANY profit if market trend looks weak/dangerous"""
        try:
            # Get recent market data to assess trend strength
            market_data = self._get_recent_prices(position.symbol, 50)
            if not market_data:
                return False
            
            current_profit = self._calculate_profit_percentage(position, current_price)
            
            # Assess trend strength
            trend_strength = self._analyze_trend_strength(market_data)
            
            # EXIT CONDITIONS for weak trends with ANY profit:
            
            # 1. Very weak trend (below 0.2) with any profit above 0.1%
            if trend_strength < 0.2 and current_profit > 0.1:
                self.logger.info(f"Very weak trend ({trend_strength:.2f}) with profit ({current_profit:.2f}%) - EXITING")
                return True
            
            # 2. Declining trend momentum with any profit
            momentum = self._calculate_trend_momentum(market_data)
            if momentum < -0.3 and current_profit > 0.15:
                self.logger.info(f"Negative momentum ({momentum:.2f}) with profit ({current_profit:.2f}%) - EXITING")
                return True
            
            # 3. Volatility spike indicating potential reversal
            volatility = self._calculate_volatility(market_data)
            if volatility > 0.8 and current_profit > 0.2:
                self.logger.info(f"High volatility ({volatility:.2f}) with profit ({current_profit:.2f}%) - EXITING")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {e}")
            return False
    
    def _should_take_profit_by_time(self, profit_pct: float, time_open: float) -> bool:
        """Take profit based on time - MORE AGGRESSIVE thresholds"""
        # AGGRESSIVE time-based profit taking
        
        if time_open > 8:  # 8+ hours
            if profit_pct > 0.1:  # ANY profit after 8 hours
                return True
        elif time_open > 4:  # 4+ hours
            if profit_pct > 0.3:  # Small profit after 4 hours
                return True
        elif time_open > 2:  # 2+ hours
            if profit_pct > 0.5:  # Decent profit after 2 hours
                return True
        elif time_open > 1:  # 1+ hour
            if profit_pct > 0.8:  # Good profit after 1 hour
                return True
        
        return False
    
    def _should_exit_profit_stagnation_aggressive(self, position: Position, current_price: float) -> bool:
        """AGGRESSIVE profit stagnation - exit much earlier"""
        try:
            # Get recent price history for this symbol
            market_data = self._get_recent_prices(position.symbol, 100)
            if not market_data:
                return False
            
            # Find the peak profit achieved
            peak_profit = 0.0
            for price_data in market_data:
                price = price_data['close']
                if position.action == 'BUY':
                    profit = (price - position.entry_price) / position.entry_price * 100
                else:  # SELL
                    profit = (position.entry_price - price) / position.entry_price * 100
                
                if profit > peak_profit:
                    peak_profit = profit
            
            # Current profit
            current_profit = self._calculate_profit_percentage(position, current_price)
            
            # AGGRESSIVE: Exit if we've given back more than 40% of peak profit (was 60%)
            profit_reduction = peak_profit - current_profit
            if peak_profit > 0.5 and profit_reduction > peak_profit * 0.4:
                self.logger.info(f"AGGRESSIVE profit stagnation: Peak {peak_profit:.2f}%, Current {current_profit:.2f}% - EXITING")
                return True
            
            # EVEN MORE AGGRESSIVE: If we've hit a good profit (>1%) and given back 25%
            if peak_profit > 1.0 and profit_reduction > peak_profit * 0.25:
                self.logger.info(f"VERY AGGRESSIVE profit protection: Peak {peak_profit:.2f}%, Current {current_profit:.2f}% - EXITING")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in aggressive profit stagnation check: {e}")
            return False
    
    def _calculate_trend_momentum(self, market_data: List[Dict]) -> float:
        """Calculate trend momentum (positive = gaining strength, negative = losing strength)"""
        if len(market_data) < 10:
            return 0.0
        
        # Compare recent price movement vs earlier movement
        recent_prices = [d['close'] for d in market_data[-10:]]
        earlier_prices = [d['close'] for d in market_data[-20:-10]]
        
        recent_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        earlier_change = (earlier_prices[-1] - earlier_prices[0]) / earlier_prices[0]
        
        return recent_change - earlier_change
    
    def _calculate_volatility(self, market_data: List[Dict]) -> float:
        """Calculate volatility (higher = more unstable)"""
        if len(market_data) < 10:
            return 0.0
        
        prices = [d['close'] for d in market_data[-20:]]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return variance ** 0.5  # Standard deviation

    def _should_exit_market_reversal(self, position: Position, current_price: float) -> bool:
        """Exit based on market reversal signals"""
        try:
            # Get recent price data to analyze trend
            market_data = self._get_recent_prices(position.symbol, 50)
            if not market_data or len(market_data) < 10:
                return False
            
            # Analyze trend strength
            trend_strength = self._analyze_trend_strength(market_data)
            current_profit = self._calculate_profit_percentage(position, current_price)
            
            # Exit conditions based on trend weakening
            if trend_strength < 0.3 and current_profit > 0.3:  # Weak trend but profitable
                return True
            
            # Check for trend reversal
            if self._detect_trend_reversal(market_data, position.action):
                if current_profit > 0.1:  # Exit if slightly profitable
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in market reversal check: {e}")
            return False
    
    def _should_take_partial_profit(self, position: Position, current_price: float) -> bool:
        """CONSERVATIVE partial profit taking - only for account growth"""
        current_profit = self._calculate_profit_percentage(position, current_price)
        time_open = (datetime.now() - position.open_time).total_seconds() / 3600
        
        # CONSERVATIVE: Only take partial profits at much higher levels for account growth
        
        # Level 1: Take 25% at 1.5% profit (let it run more)
        if current_profit >= 1.5:
            self._partial_close_position(position, 0.25)
            self.logger.info(f"Conservative partial profit: 25% taken at {current_profit:.2f}% profit")
            return True
        
        # Level 2: Take 30% at 2.0% profit 
        if current_profit >= 2.0:
            self._partial_close_position(position, 0.3)
            self.logger.info(f"Partial profit: 30% taken at {current_profit:.2f}% profit")
            return True
        
        # Level 3: Take 40% at 2.5% profit 
        if current_profit >= 2.5:
            self._partial_close_position(position, 0.4)
            self.logger.info(f"Partial profit: 40% taken at {current_profit:.2f}% profit")
            return True
        
        # Level 4: Take remaining at 3.0% profit (let it run even more)
        if current_profit >= 3.0:
            self._partial_close_position(position, 1.0)  # Close remaining position
            self.logger.info(f"Final profit: Position closed at {current_profit:.2f}% profit")
            return True
        
        return False
    
    def _get_recent_prices(self, symbol: str, count: int) -> List[Dict[str, Any]]:
        """Get recent price data for analysis"""
        try:
            rates = self.mt5_connector.get_rates(symbol, self.timeframes['M15'], 0, count)
            return rates if rates else []
        except Exception as e:
            self.logger.error(f"Error getting recent prices: {e}")
            return []
    
    def _analyze_trend_strength(self, market_data: List[Dict[str, Any]]) -> float:
        """Analyze trend strength (0-1 scale)"""
        if len(market_data) < 10:
            return 0.0
        
        try:
            # Simple trend strength calculation using price movements
            price_changes = []
            for i in range(1, len(market_data)):
                prev_price = market_data[i-1]['close']
                curr_price = market_data[i]['close']
                change = (curr_price - prev_price) / prev_price
                price_changes.append(abs(change))
            
            # Calculate average directional movement
            positive_moves = sum(1 for change in price_changes[-10:] if change > 0)
            return positive_moves / 10.0
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend strength: {e}")
            return 0.0
    
    def _detect_trend_reversal(self, market_data: List[Dict[str, Any]], action: str) -> bool:
        """Detect potential trend reversal"""
        if len(market_data) < 20:
            return False
        
        try:
            # Compare recent momentum to older momentum
            recent_prices = [d['close'] for d in market_data[-10:]]
            older_prices = [d['close'] for d in market_data[-20:-10]]
            
            # Calculate momentum
            recent_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            older_momentum = (older_prices[-1] - older_prices[0]) / older_prices[0]
            
            # Detect reversal
            if action == 'BUY':
                return recent_momentum < 0 and older_momentum > 0
            else:  # SELL
                return recent_momentum > 0 and older_momentum < 0
            
        except Exception as e:
            self.logger.error(f"Error detecting trend reversal: {e}")
            return False
    
    def _close_position(self, position: Position):
        """Close a position with proper MT5 order execution"""
        try:
            # Get symbol info for closing
            symbol_info = self.mt5_connector.get_symbol_info(position.symbol)
            if not symbol_info:
                self.logger.error(f"Cannot get symbol info for {position.symbol}")
                return
            
            # Determine closing order type and price
            if position.action == 'BUY':
                # To close BUY position, we SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = symbol_info['bid']
            else:  # SELL
                # To close SELL position, we BUY
                order_type = mt5.ORDER_TYPE_BUY
                price = symbol_info['ask']
            
            # Create closing order request
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.position_size,
                "type": order_type,
                "position": position.ticket,  # Close specific position
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Neural-Close-{position.unrealized_pnl:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Send closing order
            result = self.mt5_connector.send_order(close_request)
            
            if result and result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                position.status = 'CLOSED'
                self.logger.info(f"Position {position.ticket} closed successfully: "
                               f"{position.symbol} {position.action}, P&L: ${position.unrealized_pnl:.2f}")
                
                # Update performance metrics
                self._update_performance_metrics()
                
            else:
                self.logger.error(f"Failed to close position {position.ticket}: {result}")
                
        except Exception as e:
            self.logger.error(f"Error closing position {position.ticket}: {e}")
    
    def _partial_close_position(self, position: Position, close_percentage: float = 0.5):
        """Close a portion of the position for partial profit taking"""
        try:
            # Calculate partial volume
            partial_volume = position.position_size * close_percentage
            
            if partial_volume < self._get_min_volume(position.symbol):
                self.logger.warning(f"Partial volume too small for {position.symbol}")
                return
            
            # Get symbol info
            symbol_info = self.mt5_connector.get_symbol_info(position.symbol)
            if not symbol_info:
                return
            
            # Determine closing order type and price
            if position.action == 'BUY':
                order_type = mt5.ORDER_TYPE_SELL
                price = symbol_info['bid']
            else:  # SELL
                order_type = mt5.ORDER_TYPE_BUY
                price = symbol_info['ask']
            
            # Create partial closing order
            partial_close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": partial_volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Neural-Partial-Close-{close_percentage:.1%}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Send partial closing order
            result = self.mt5_connector.send_order(partial_close_request)
            
            if result and result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                # Update remaining position size
                position.position_size -= partial_volume
                
                partial_pnl = position.unrealized_pnl * close_percentage
                self.logger.info(f"Partial close successful: {close_percentage:.1%} of position "
                               f"closed, P&L: ${partial_pnl:.2f}")
                
                # If this was the last partial close, mark as closed
                if position.position_size <= self._get_min_volume(position.symbol):
                    position.status = 'CLOSED'
                    self.logger.info(f"Position {position.ticket} fully closed via partial exits")
                    
            else:
                self.logger.error(f"Failed to partially close position {position.ticket}: {result}")
                
        except Exception as e:
            self.logger.error(f"Error in partial close for position {position.ticket}: {e}")
    
    def _get_min_volume(self, symbol: str) -> float:
        """Get minimum volume for symbol"""
        try:
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            return symbol_info['volume_min'] if symbol_info else 0.01
        except:
            return 0.01
    
    def _update_performance_metrics(self):
        """Update trading performance metrics"""
        try:
            # Calculate metrics from positions and signals
            closed_positions = [p for p in self.positions.values() if p.status == 'CLOSED']
            
            if closed_positions:
                winning_trades = sum(1 for p in closed_positions if p.unrealized_pnl > 0)
                losing_trades = sum(1 for p in closed_positions if p.unrealized_pnl < 0)
                total_trades = len(closed_positions)
                
                self.performance_metrics.update({
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': sum(p.unrealized_pnl for p in closed_positions),
                    'current_drawdown': self._calculate_drawdown(closed_positions)
                })
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_drawdown(self, positions: List[Position]) -> float:
        """Calculate current drawdown"""
        # Simplified drawdown calculation
        if not positions:
            return 0.0
        
        profits = [p.unrealized_pnl for p in positions]
        if not profits:
            return 0.0
        
        peak = max(profits)
        current = sum(profits)
        
        if peak <= 0:
            return 0.0
        
        return max(0, (peak - current) / peak)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get list of recent signals"""
        recent_signals = [s for s in self.signals_history[-20:]]  # Last 20 signals
        
        return [
            {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'action': signal.action,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size,
                'reason': signal.reason,
                'executed': signal.executed
            }
            for signal in recent_signals
        ]
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get list of active positions"""
        active_positions = [p for p in self.positions.values() if p.status == 'OPEN']
        
        return [
            {
                'ticket': position.ticket,
                'symbol': position.symbol,
                'action': position.action,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'position_size': position.position_size,
                'unrealized_pnl': position.unrealized_pnl,
                'open_time': position.open_time.isoformat(),
                'status': position.status
            }
            for position in active_positions
        ]
