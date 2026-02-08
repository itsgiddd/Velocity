#!/usr/bin/env python3
"""
Maximum Profit Trading Engine
==========================

Enhanced trading engine with:
- Multi-timeframe analysis (M15, H1, H4, D1)
- 4-candle continuation pattern recognition
- Maximum profit taking strategy
- USDJPY bidirection trading capability
- Comprehensive market analysis
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
    """Enhanced trading signal data structure"""
    
    def __init__(self, symbol: str, action: str, confidence: float, 
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float, reason: str, timeframe_analysis: Dict[str, Any]):
        self.symbol = symbol
        self.action = action  # 'BUY', 'SELL', 'HOLD'
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.reason = reason
        self.timeframe_analysis = timeframe_analysis
        self.timestamp = datetime.now()
        self.executed = False
        self.order_ticket = None

class Position:
    """Enhanced position tracking with maximum profit tracking"""
    
    def __init__(self, ticket: int, symbol: str, action: str, 
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float):
        self.ticket = ticket
        self.symbol = symbol
        self.action = action
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit  # Will be updated for maximum profit
        self.position_size = position_size
        self.open_time = datetime.now()
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.status = 'OPEN'  # OPEN, CLOSED, PARTIAL
        self.peak_profit = 0.0
        self.max_profit_target = None

class MaximumProfitTradingEngine:
    """Enhanced trading engine with maximum profit capability"""
    
    def __init__(self, mt5_connector: MT5Connector, model_manager: NeuralModelManager,
                 risk_per_trade: float = 0.015, confidence_threshold: float = 0.65,
                 trading_pairs: List[str] = None, max_concurrent_positions: int = 3):
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.mt5_connector = mt5_connector
        self.model_manager = model_manager
        
        # Trading parameters
        self.risk_per_trade = risk_per_trade  # 1.5% default
        self.confidence_threshold = confidence_threshold  # 65% default
        self.trading_pairs = trading_pairs or ['USDJPY']  # Only USDJPY
        self.max_concurrent_positions = max_concurrent_positions
        
        # Timeframes for analysis
        self.timeframes = {
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
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
            'max_profit_trades': 0,
            'continuation_patterns_detected': 0
        }

    def start_trading(self):
        """Start the enhanced trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine is already running")
            return
        
        self.logger.info("Starting Maximum Profit Trading Engine")
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
    def stop_trading(self):
        """Stop the trading engine"""
        self.logger.info("Stopping Maximum Profit Trading Engine")
        self.is_running = False
        
    def _trading_loop(self):
        """Enhanced trading loop with multi-timeframe analysis"""
        self.logger.info("Enhanced trading loop started")
        
        try:
            while self.is_running:
                try:
                    # Get market data for all timeframes
                    for symbol in self.trading_pairs:
                        # Multi-timeframe analysis
                        market_data = self._get_comprehensive_market_data(symbol)
                        if not market_data:
                            continue
                        
                        # Check if we can open new positions
                        if len(self.positions) < self.max_concurrent_positions:
                            # Generate enhanced signal
                            signal = self._generate_enhanced_signal(symbol, market_data)
                            if signal:
                                self._execute_signal(signal)
                        
                        # Monitor existing positions with maximum profit tracking
                        self._monitor_positions_maximum_profit()
                    
                    # Wait before next iteration
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")
                    time.sleep(60)  # Wait longer on error
                    
        except Exception as e:
            self.logger.error(f"Critical error in trading loop: {e}")
        finally:
            self.is_running = False

    def _get_comprehensive_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data for multiple timeframes"""
        try:
            market_data = {}
            
            # Get data for all timeframes
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
            self.logger.error(f"Error getting comprehensive market data for {symbol}: {e}")
            return None

    def _generate_enhanced_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate enhanced signal with multi-timeframe analysis"""
        try:
            # Multi-timeframe analysis
            timeframe_analysis = self._analyze_all_timeframes(symbol, market_data)
            
            # 4-candle continuation pattern analysis
            continuation_analysis = self._analyze_continuation_patterns(market_data)
            
            # Combine all analyses
            combined_confidence = self._calculate_combined_confidence(timeframe_analysis, continuation_analysis)
            
            # Check confidence threshold
            if combined_confidence < self.confidence_threshold:
                return None
            
            # Get symbol info
            symbol_info = market_data['symbol_info']
            
            # Determine action based on comprehensive analysis
            action = self._determine_trade_action(timeframe_analysis, continuation_analysis)
            
            if action == 'HOLD':
                return None
            
            # Calculate trading parameters
            entry_price = symbol_info['ask'] if action == 'BUY' else symbol_info['bid']
            
            # Dynamic stop loss based on multiple timeframes
            stop_loss = self._calculate_dynamic_stop_loss(symbol, action, entry_price, timeframe_analysis)
            
            # Maximum profit target (no fixed 20-pip limit)
            take_profit = self._calculate_maximum_profit_target(symbol, action, entry_price, timeframe_analysis)
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, entry_price, stop_loss, symbol_info)
            
            if position_size <= 0:
                return None
            
            # Create enhanced signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=combined_confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"Multi-timeframe analysis: {action} based on comprehensive market analysis",
                timeframe_analysis={
                    'timeframe_analysis': timeframe_analysis,
                    'continuation_analysis': continuation_analysis,
                    'combined_confidence': combined_confidence
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced signal for {symbol}: {e}")
            return None

    def _analyze_all_timeframes(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all timeframes for comprehensive market view"""
        analysis = {}
        
        try:
            for tf_name, data in market_data.items():
                if tf_name == 'symbol_info':
                    continue
                
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                # Technical analysis for each timeframe
                tf_analysis = {
                    'trend': self._analyze_trend(df),
                    'momentum': self._analyze_momentum(df),
                    'volatility': self._analyze_volatility(df),
                    'support_resistance': self._analyze_support_resistance(df),
                    'price_action': self._analyze_price_action(df)
                }
                
                analysis[tf_name] = tf_analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframes: {e}")
            return {}

    def _analyze_continuation_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze 4-candle continuation patterns"""
        try:
            # Use M15 data for pattern recognition
            if 'M15' not in market_data:
                return {'pattern_detected': False, 'confidence': 0.0}
            
            df = pd.DataFrame(market_data['M15'])
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # 4-candle continuation analysis
            consecutive_analysis = self._detect_4_candle_continuation(df)
            
            # Multi-timeframe pattern confirmation
            pattern_confirmation = self._confirm_pattern_across_timeframes(market_data)
            
            return {
                'pattern_detected': consecutive_analysis['pattern_detected'],
                'pattern_type': consecutive_analysis['pattern_type'],
                'pattern_strength': consecutive_analysis['pattern_strength'],
                'confirmation_strength': pattern_confirmation['confirmation_strength'],
                'overall_confidence': consecutive_analysis['pattern_strength'] * pattern_confirmation['confirmation_strength']
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing continuation patterns: {e}")
            return {'pattern_detected': False, 'confidence': 0.0}

    def _detect_4_candle_continuation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect 4-candle continuation patterns"""
        try:
            if len(df) < 4:
                return {'pattern_detected': False, 'pattern_type': 'none', 'pattern_strength': 0.0}
            
            # Get last 4 candles
            last_4_candles = df.iloc[-4:]
            
            # Analyze candle directions
            directions = []
            for i, (_, candle) in enumerate(last_4_candles.iterrows()):
                if candle['close'] > candle['open']:
                    directions.append('bullish')
                elif candle['close'] < candle['open']:
                    directions.append('bearish')
                else:
                    directions.append('doji')
            
            # Check for continuation patterns
            if all(d == 'bullish' for d in directions):
                return {
                    'pattern_detected': True,
                    'pattern_type': 'bullish_continuation',
                    'pattern_strength': min(1.0, len([d for d in directions if d == 'bullish']) / 4.0)
                }
            elif all(d == 'bearish' for d in directions):
                return {
                    'pattern_detected': True,
                    'pattern_type': 'bearish_continuation',
                    'pattern_strength': min(1.0, len([d for d in directions if d == 'bearish']) / 4.0)
                }
            else:
                return {'pattern_detected': False, 'pattern_type': 'none', 'pattern_strength': 0.0}
                
        except Exception as e:
            self.logger.error(f"Error detecting continuation patterns: {e}")
            return {'pattern_detected': False, 'pattern_type': 'none', 'pattern_strength': 0.0}

    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze trend across timeframes"""
        try:
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # Trend strength
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            trend_up = current_price > sma_20 > sma_50
            trend_down = current_price < sma_20 < sma_50
            
            if trend_up:
                trend_strength = (current_price - sma_20) / sma_20
            elif trend_down:
                trend_strength = -(sma_20 - current_price) / sma_20
            else:
                trend_strength = 0.0
            
            return {
                'trend_direction': 'up' if trend_up else 'down' if trend_down else 'sideways',
                'trend_strength': abs(trend_strength),
                'sma_alignment': 1.0 if trend_up or trend_down else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return {'trend_direction': 'sideways', 'trend_strength': 0.0, 'sma_alignment': 0.0}

    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze momentum indicators"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()
            
            current_rsi = rsi.iloc[-1]
            current_macd = macd.iloc[-1]
            current_signal = macd_signal.iloc[-1]
            
            return {
                'rsi': current_rsi,
                'rsi_momentum': 1.0 if current_rsi > 50 else 0.0,
                'macd': current_macd,
                'macd_signal_line': current_signal,
                'macd_momentum': 1.0 if current_macd > current_signal else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return {'rsi': 50.0, 'rsi_momentum': 0.5, 'macd': 0.0, 'macd_signal_line': 0.0, 'macd_momentum': 0.5}

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility"""
        try:
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            current_atr = atr.iloc[-1]
            current_price = df['close'].iloc[-1]
            bb_position = (current_price - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
            
            return {
                'atr': current_atr,
                'volatility': current_atr / current_price,
                'bb_position': bb_position,
                'bb_squeeze': bb_std.iloc[-1] / df['bb_middle'].iloc[-1] < 0.02
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")
            return {'atr': 0.0, 'volatility': 0.0, 'bb_position': 0.5, 'bb_squeeze': False}

    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze support and resistance levels"""
        try:
            # Find pivot highs and lows
            df['pivot_high'] = (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])
            df['pivot_low'] = (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])
            
            # Get recent support and resistance
            recent_highs = df[df['pivot_high']]['high'].tail(5)
            recent_lows = df[df['pivot_low']]['low'].tail(5)
            
            current_price = df['close'].iloc[-1]
            
            if len(recent_highs) > 0 and len(recent_lows) > 0:
                resistance = recent_highs.mean()
                support = recent_lows.mean()
                
                distance_to_resistance = (resistance - current_price) / current_price
                distance_to_support = (current_price - support) / current_price
                
                return {
                    'resistance_level': resistance,
                    'support_level': support,
                    'distance_to_resistance': distance_to_resistance,
                    'distance_to_support': distance_to_support,
                    'near_resistance': distance_to_resistance < 0.01,
                    'near_support': distance_to_support < 0.01
                }
            
            return {
                'resistance_level': current_price * 1.02,
                'support_level': current_price * 0.98,
                'distance_to_resistance': 0.02,
                'distance_to_support': 0.02,
                'near_resistance': False,
                'near_support': False
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing support/resistance: {e}")
            return {'resistance_level': df['close'].iloc[-1] * 1.02, 'support_level': df['close'].iloc[-1] * 0.98, 'distance_to_resistance': 0.02, 'distance_to_support': 0.02, 'near_resistance': False, 'near_support': False}

    def _analyze_price_action(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze price action patterns"""
        try:
            # Recent candles analysis
            last_3_candles = df.iloc[-3:]
            
            # Candle patterns
            bullish_candles = sum(1 for _, candle in last_3_candles.iterrows() if candle['close'] > candle['open'])
            bearish_candles = sum(1 for _, candle in last_3_candles.iterrows() if candle['close'] < candle['open'])
            
            # Body sizes
            avg_body_size = last_3_candles.apply(lambda x: abs(x['close'] - x['open']), axis=1).mean()
            current_body_size = abs(last_3_candles['close'].iloc[-1] - last_3_candles['open'].iloc[-1])
            
            return {
                'bullish_bias': bullish_candles / 3.0,
                'bearish_bias': bearish_candles / 3.0,
                'body_expansion': current_body_size / avg_body_size if avg_body_size > 0 else 1.0,
                'price_acceleration': (df['close'].iloc[-1] - df['close'].iloc[-3]) / df['close'].iloc[-3]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing price action: {e}")
            return {'bullish_bias': 0.5, 'bearish_bias': 0.5, 'body_expansion': 1.0, 'price_acceleration': 0.0}

    def _calculate_combined_confidence(self, timeframe_analysis: Dict[str, Any], continuation_analysis: Dict[str, Any]) -> float:
        """Calculate combined confidence from all analyses"""
        try:
            # Weight different timeframes
            timeframe_weights = {'M15': 0.4, 'H1': 0.3, 'H4': 0.2, 'D1': 0.1}
            
            combined_score = 0.0
            total_weight = 0.0
            
            for tf_name, analysis in timeframe_analysis.items():
                if tf_name in timeframe_weights:
                    weight = timeframe_weights[tf_name]
                    
                    # Calculate timeframe score
                    trend_score = analysis.get('trend', {}).get('trend_strength', 0.0)
                    momentum_score = analysis.get('momentum', {}).get('rsi_momentum', 0.5)
                    price_action_score = analysis.get('price_action', {}).get('bullish_bias', 0.5)
                    
                    tf_score = (trend_score + momentum_score + price_action_score) / 3.0
                    
                    combined_score += tf_score * weight
                    total_weight += weight
            
            # Add continuation pattern bonus
            pattern_bonus = continuation_analysis.get('overall_confidence', 0.0) * 0.3
            
            final_confidence = (combined_score + pattern_bonus) if total_weight > 0 else 0.0
            
            return min(1.0, max(0.0, final_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating combined confidence: {e}")
            return 0.5

    def _determine_trade_action(self, timeframe_analysis: Dict[str, Any], continuation_analysis: Dict[str, Any]) -> str:
        """Determine trade action based on comprehensive analysis"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # Analyze each timeframe
            for tf_name, analysis in timeframe_analysis.items():
                trend_direction = analysis.get('trend', {}).get('trend_direction', 'sideways')
                momentum_direction = 'bullish' if analysis.get('momentum', {}).get('rsi_momentum', 0.5) > 0.6 else 'bearish' if analysis.get('momentum', {}).get('rsi_momentum', 0.5) < 0.4 else 'neutral'
                price_action_direction = 'bullish' if analysis.get('price_action', {}).get('bullish_bias', 0.5) > 0.6 else 'bearish' if analysis.get('price_action', {}).get('bullish_bias', 0.5) < 0.4 else 'neutral'
                
                if trend_direction == 'up' or momentum_direction == 'bullish' or price_action_direction == 'bullish':
                    bullish_signals += 1
                elif trend_direction == 'down' or momentum_direction == 'bearish' or price_action_direction == 'bearish':
                    bearish_signals += 1
            
            # Add continuation pattern influence
            pattern_type = continuation_analysis.get('pattern_type', 'none')
            if pattern_type == 'bullish_continuation':
                bullish_signals += 2
            elif pattern_type == 'bearish_continuation':
                bearish_signals += 2
            
            # Make decision
            if bullish_signals > bearish_signals and bullish_signals >= 2:
                return 'BUY'
            elif bearish_signals > bullish_signals and bearish_signals >= 2:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            self.logger.error(f"Error determining trade action: {e}")
            return 'HOLD'

    def _calculate_dynamic_stop_loss(self, symbol: str, action: str, entry_price: float, timeframe_analysis: Dict[str, Any]) -> float:
        """Calculate dynamic stop loss based on multi-timeframe analysis"""
        try:
            # Base stop loss using ATR from multiple timeframes
            atr_values = []
            
            for tf_name, analysis in timeframe_analysis.items():
                atr = analysis.get('volatility', {}).get('atr', 0.0)
                if atr > 0:
                    atr_values.append(atr)
            
            if atr_values:
                avg_atr = np.mean(atr_values)
                
                # Dynamic stop loss based on market volatility
                if action == 'BUY':
                    stop_loss = entry_price - (avg_atr * 2.0)
                else:  # SELL
                    stop_loss = entry_price + (avg_atr * 2.0)
            else:
                # Fallback to percentage-based stop loss
                if action == 'BUY':
                    stop_loss = entry_price * 0.98  # 2% stop loss
                else:  # SELL
                    stop_loss = entry_price * 1.02  # 2% stop loss
            
            # Ensure minimum stop loss distance
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            min_stop_distance = pip_value * 20  # 20 pips minimum
            
            if action == 'BUY':
                min_stop = entry_price - min_stop_distance
                stop_loss = max(stop_loss, min_stop)
            else:  # SELL
                min_stop = entry_price + min_stop_distance
                stop_loss = min(stop_loss, min_stop)
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic stop loss: {e}")
            # Fallback stop loss
            if action == 'BUY':
                return entry_price * 0.98
            else:  # SELL
                return entry_price * 1.02

    def _calculate_maximum_profit_target(self, symbol: str, action: str, entry_price: float, timeframe_analysis: Dict[str, Any]) -> float:
        """Calculate maximum profit target (no fixed limit)"""
        try:
            # Analyze resistance/support levels
            resistance_support_data = []
            
            for tf_name, analysis in timeframe_analysis.items():
                sr_analysis = analysis.get('support_resistance', {})
                resistance = sr_analysis.get('resistance_level', entry_price * 1.05)
                support = sr_analysis.get('support_level', entry_price * 0.95)
                resistance_support_data.append((resistance, support))
            
            if resistance_support_data:
                # Take the most significant level in the direction of trade
                if action == 'BUY':
                    # Target resistance levels
                    targets = [resistance for resistance, _ in resistance_support_data if resistance > entry_price]
                    if targets:
                        max_target = max(targets)
                        profit_distance = max_target - entry_price
                        
                        # Ensure reasonable profit target (minimum 30 pips, maximum based on analysis)
                        min_profit = 0.0030  # 30 pips
                        max_profit = profit_distance * 0.8  # 80% of distance to resistance
                        
                        return entry_price + max(min_profit, max_profit)
                
                else:  # SELL
                    # Target support levels
                    targets = [support for _, support in resistance_support_data if support < entry_price]
                    if targets:
                        min_target = min(targets)
                        profit_distance = entry_price - min_target
                        
                        # Ensure reasonable profit target (minimum 30 pips, maximum based on analysis)
                        min_profit = 0.0030  # 30 pips
                        max_profit = profit_distance * 0.8  # 80% of distance to support
                        
                        return entry_price - max(min_profit, max_profit)
            
            # Fallback: Use trend strength for profit target
            trend_strength = 0
            for analysis in timeframe_analysis.values():
                trend_strength += analysis.get('trend', {}).get('trend_strength', 0.0)
            
            trend_strength /= len(timeframe_analysis)
            
            if action == 'BUY':
                profit_multiplier = 2.0 + (trend_strength * 3.0)  # 2x to 5x stop loss
                return entry_price + (profit_multiplier * (entry_price - self._calculate_dynamic_stop_loss(symbol, action, entry_price, timeframe_analysis)))
            else:  # SELL
                profit_multiplier = 2.0 + (trend_strength * 3.0)
                return entry_price - (profit_multiplier * (self._calculate_dynamic_stop_loss(symbol, action, entry_price, timeframe_analysis) - entry_price))
            
        except Exception as e:
            self.logger.error(f"Error calculating maximum profit target: {e}")
            # Conservative fallback
            if action == 'BUY':
                return entry_price * 1.03  # 30 pips
            else:  # SELL
                return entry_price * 0.97  # 30 pips

    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, symbol_info: Dict[str, Any]) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account info
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                return 0.0
            
            balance = account_info['balance']
            risk_amount = balance * self.risk_per_trade
            
            # Calculate pip value
            if 'JPY' in symbol:
                pip_value = 0.01
            else:
                pip_value = 0.0001
            
            # Calculate stop loss in pips
            stop_loss_pips = abs(entry_price - stop_loss) / pip_value
            
            if stop_loss_pips == 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / (stop_loss_pips * pip_value * 100000)  # Standard lot calculation
            
            # Apply symbol-specific constraints
            symbol_constraints = symbol_info.get('trade_contract_size', 100000)
            
            return min(position_size, balance * 0.1 / symbol_constraints)  # Max 10% of balance per trade
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _monitor_positions_maximum_profit(self):
        """Monitor positions for maximum profit potential"""
        try:
            positions_to_close = []
            
            for ticket, position in list(self.positions.items()):
                # Get current position data
                mt5_positions = self.mt5_connector.get_positions(symbol=position.symbol)
                if not mt5_positions:
                    continue
                
                mt5_pos = next((p for p in mt5_positions if p['ticket'] == ticket), None)
                if not mt5_pos:
                    continue
                
                current_price = mt5_pos['price_current']
                profit = mt5_pos['profit']
                
                # Update peak profit
                if profit > position.peak_profit:
                    position.peak_profit = profit
                
                # Maximum profit taking logic
                should_close = self._should_close_for_maximum_profit(position, mt5_pos, current_price)
                
                if should_close:
                    positions_to_close.append(ticket)
            
            # Close positions
            for ticket in positions_to_close:
                self._close_position(ticket)
                
        except Exception as e:
            self.logger.error(f"Error monitoring positions for maximum profit: {e}")

    def _should_close_for_maximum_profit(self, position: Position, mt5_pos: Dict[str, Any], current_price: float) -> bool:
        """Determine if position should be closed for maximum profit"""
        try:
            current_profit = mt5_pos['profit']
            
            # If we're in profit, protect the peak
            if current_profit > 0:
                # Update take profit based on maximum profit achieved
                if current_profit > position.peak_profit * 0.8:  # Still near peak
                    # Let it run - we're near maximum profit
                    return False
                elif current_profit < position.peak_profit * 0.5:  # Given back 50% of peak profit
                    # Take profit to protect gains
                    self.logger.info(f"Taking maximum profit: Peak ${position.peak_profit:.2f}, Current ${current_profit:.2f}")
                    return True
            
            # Check stop loss
            if self._is_stop_loss_hit(position, current_price):
                return True
            
            # Check original take profit (as absolute minimum)
            if self._is_take_profit_hit(position, current_price):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking maximum profit conditions: {e}")
            return False

    def _is_stop_loss_hit(self, position: Position, current_price: float) -> bool:
        """Check if stop loss is hit"""
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

    def _confirm_pattern_across_timeframes(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Confirm patterns across multiple timeframes"""
        try:
            confirmation_score = 0.0
            total_timeframes = 0
            
            for tf_name, data in market_data.items():
                if tf_name == 'symbol_info':
                    continue
                
                total_timeframes += 1
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                # Quick pattern check
                if len(df) >= 4:
                    last_4 = df.iloc[-4:]
                    bullish_count = sum(1 for _, candle in last_4.iterrows() if candle['close'] > candle['open'])
                    
                    if bullish_count >= 3:
                        confirmation_score += 1.0
                    elif bullish_count <= 1:
                        confirmation_score -= 1.0
            
            return {
                'confirmation_strength': abs(confirmation_score) / total_timeframes if total_timeframes > 0 else 0.0,
                'pattern_consensus': 'bullish' if confirmation_score > 0 else 'bearish' if confirmation_score < 0 else 'mixed'
            }
            
        except Exception as e:
            self.logger.error(f"Error confirming patterns across timeframes: {e}")
            return {'confirmation_strength': 0.0, 'pattern_consensus': 'mixed'}

    def _execute_signal(self, signal: TradingSignal) -> bool:
        """Execute trading signal"""
        try:
            # Open position
            order_result = self.mt5_connector.place_order(
                symbol=signal.symbol,
                action=signal.action,
                volume=signal.position_size,
                price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order_result and order_result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                ticket = order_result['order']
                
                # Create position object
                position = Position(
                    ticket=ticket,
                    symbol=signal.symbol,
                    action=signal.action,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=signal.position_size
                )
                
                self.positions[ticket] = position
                self.signals_history.append(signal)
                
                self.logger.info(f"Position opened: {signal.action} {signal.symbol} at {signal.entry_price}")
                return True
            else:
                self.logger.error(f"Failed to open position: {order_result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return False

    def _close_position(self, ticket: int) -> bool:
        """Close position"""
        try:
            if ticket not in self.positions:
                return False
            
            position = self.positions[ticket]
            
            # Close position
            close_result = self.mt5_connector.close_position(ticket)
            
            if close_result and close_result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                # Update metrics
                self.performance_metrics['total_trades'] += 1
                
                # Check if it was profitable
                profit = close_result.get('profit', 0.0)
                if profit > 0:
                    self.performance_metrics['winning_trades'] += 1
                    self.performance_metrics['total_pnl'] += profit
                else:
                    self.performance_metrics['losing_trades'] += 1
                    self.performance_metrics['total_pnl'] += profit
                
                # Calculate win rate
                total = self.performance_metrics['total_trades']
                if total > 0:
                    self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / total
                
                # Remove position
                del self.positions[ticket]
                
                self.logger.info(f"Position closed: Ticket {ticket}, Profit: ${profit:.2f}")
                return True
            else:
                self.logger.error(f"Failed to close position {ticket}: {close_result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return False