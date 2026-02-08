#!/usr/bin/env python3
"""
LIVE NEURAL TRADING BOT - AUTOMATED MT5 TRADING
===============================================

This script implements automated trading using the neural system
designed to achieve 78%+ accuracy for forex trading.

Features:
- Real-time market data analysis
- Neural network signal generation
- Automated trade execution
- Risk management
- Performance monitoring
- Error handling and recovery

Author: Neural Trading System
Date: 2026-02-05
Target: 78%+ Win Rate
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import threading
from dataclasses import dataclass
from enum import Enum

# Single instance check
LOCK_FILE = ".neural_trading_bot.lock"

def check_single_instance():
    """Check if another instance is already running"""
    import socket
    try:
        # Try to create a lock file with port info
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, 'r') as f:
                lock_info = f.read().strip()
            print(f"Warning: Another instance may be running (lock file exists: {lock_info})")
            return False
        
        # Create lock file
        hostname = socket.gethostname()
        pid = os.getpid()
        with open(LOCK_FILE, 'w') as f:
            f.write(f"{hostname}:{pid}:{datetime.now().isoformat()}")
        print(f"Lock file created: {LOCK_FILE}")
        return True
    except Exception as e:
        print(f"Warning: Could not create lock file: {e}")
        return True

def release_lock():
    """Release the lock file"""
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            logger.info("Lock file released")
    except Exception as e:
        logger.error(f"Could not release lock file: {e}")

# Check single instance at module load
if __name__ != "__main__":
    check_single_instance()

# Import our neural system components
try:
    from enhanced_neural_architecture import EnhancedTradingBrain
    from feature_engineering_pipeline import FeatureEngineeringPipeline
    from contextual_trading_brain import ContextualTradingBrain
    from neural_ai_brain_integration import NeuralAIBrain
except ImportError as e:
    print(f"Warning: Could not import neural components: {e}")
    print("Using simplified neural trading...")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neural_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"

class TradeResult(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

@dataclass
class TradeSignal:
    symbol: str
    action: TradeResult
    confidence: float
    lot_size: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime

@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str

@dataclass
class MarketData:
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    h1_data: pd.DataFrame
    h4_data: pd.DataFrame
    d1_data: pd.DataFrame

class LiveNeuralTradingBot:
    """
    LIVE NEURAL TRADING BOT
    
    Automated trading system using neural networks for 78%+ accuracy.
    """
    
    def __init__(self, 
                 trading_mode: TradingMode = TradingMode.DEMO,
                 confidence_threshold: float = 0.78,  # 78% threshold
                 max_risk_per_trade: float = 0.02,
                 symbols: List[str] = None):
        
        self.trading_mode = trading_mode
        self.confidence_threshold = confidence_threshold
        self.max_risk_per_trade = max_risk_per_trade
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY']
        
        # Neural system components
        self.neural_brain = None
        self.feature_pipeline = None
        self.is_training_mode = False
        
        # Trading state
        self.is_running = False
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
        
        # Risk management
        self.max_open_positions = 5
        self.max_daily_risk = 0.05
        
        # Sync positions with MT5 on startup
        self._sync_positions_with_mt5()
    
    def _sync_positions_with_mt5(self):
        """Sync bot's internal position tracking with actual MT5 positions."""
        try:
            positions = mt5.positions_get()
            if positions:
                # Clear stale positions first
                self.positions.clear()
                
                for position in positions:
                    symbol = position.symbol
                    self.positions[symbol] = {
                        'ticket': position.ticket,
                        'symbol': symbol,
                        'type': position.type,
                        'volume': position.volume,
                        'price_open': position.price_open,
                        'profit': position.profit,
                        'time': position.time
                    }
                    logger.info(f"Synced existing position: {symbol} (Ticket: {position.ticket})")
                
                logger.info(f"Synced {len(positions)} positions from MT5")
            else:
                # No positions in MT5, ensure our tracking is clean
                if self.positions:
                    logger.info("No MT5 positions found, clearing internal tracking")
                    self.positions.clear()
        except Exception as e:
            logger.error(f"Error syncing positions with MT5: {e}")
        
        # Risk management
        self.max_drawdown_limit = 0.15
        
        # Initialize neural system
        self._initialize_neural_system()
        
        logger.info("Live Neural Trading Bot Initialized")
        logger.info(f"Trading Mode: {self.trading_mode.value}")
        logger.info(f"Confidence Threshold: {self.confidence_threshold:.1%}")
        logger.info(f"Max Risk Per Trade: {self.max_risk_per_trade:.1%}")
        logger.info(f"Target Symbols: {', '.join(self.symbols)}")
    
    def _initialize_neural_system(self):
        """Initialize the neural trading system"""
        try:
            logger.info("Initializing Neural System...")
            
            # Initialize neural brain
            self.neural_brain = NeuralAIBrain(
                use_neural=True,
                fallback_to_original=False
            )
            
            # Initialize feature pipeline
            self.feature_pipeline = FeatureEngineeringPipeline()
            
            logger.info("Neural System Initialized Successfully")
            
        except Exception as e:
            logger.error(f"Neural System Initialization Failed: {e}")
            # Fallback to simplified trading
            self.neural_brain = None
            self.is_training_mode = True
    
    def connect_to_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        try:
            logger.info("Connecting to MetaTrader 5...")
            
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Could not get account info")
                return False
            
            logger.info(f"Connected to MT5")
            logger.info(f"Account: {account_info.login}")
            logger.info(f"Server: {account_info.server}")
            logger.info(f"Balance: {account_info.balance}")
            logger.info(f"Currency: {account_info.currency}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 Connection Error: {e}")
            return False
    
    def get_account_info(self, max_retries: int = 3, retry_delay: float = 1.0) -> Optional[AccountInfo]:
        """Get current account information with retry logic"""
        for attempt in range(max_retries):
            try:
                account_info = mt5.account_info()
                if account_info is None:
                    if attempt < max_retries - 1:
                        logger.warning(f"Account info is None, retrying ({attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return None
                
                return AccountInfo(
                    balance=account_info.balance,
                    equity=account_info.equity,
                    margin=account_info.margin,
                    free_margin=account_info.margin_free,
                    margin_level=account_info.margin_level,
                    currency=account_info.currency
                )
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error getting account info (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to get account info after {max_retries} attempts: {e}")
                    return None
        return None
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for symbol"""
        try:
            # Get current prices
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Symbol info not found: {symbol}")
                return None
            
            # Get quote data
            quote = mt5.symbol_info_tick(symbol)
            if quote is None:
                logger.warning(f"Quote data not found: {symbol}")
                return None
            
            # Calculate spread
            spread = quote.ask - quote.bid
            
            # Get historical data - ALL TIMEFRAMES USER TRADES ON
            m15_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)  # 15-minute
            h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)   # 1-hour
            h4_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)   # 4-hour
            d1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 100)   # 1-day
            
            # Convert to DataFrames
            m15_data = pd.DataFrame(m15_rates) if m15_rates is not None else pd.DataFrame()
            h1_data = pd.DataFrame(h1_rates) if h1_rates is not None else pd.DataFrame()
            h4_data = pd.DataFrame(h4_rates) if h4_rates is not None else pd.DataFrame()
            d1_data = pd.DataFrame(d1_rates) if d1_rates is not None else pd.DataFrame()
            
            return MarketData(
                symbol=symbol,
                bid=quote.bid,
                ask=quote.ask,
                spread=spread,
                timestamp=datetime.fromtimestamp(quote.time),
                h1_data=h1_data,  # 1-hour data
                h4_data=h4_data,  # 4-hour data
                d1_data=d1_data   # 1-day data (kept for additional context)
            )
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _get_m15_data(self, symbol: str) -> pd.DataFrame:
        """Get 15-minute timeframe data for analysis"""
        try:
            m15_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
            return pd.DataFrame(m15_rates) if m15_rates is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting M15 data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_neural_signal(self, market_data: MarketData, account_info: AccountInfo) -> Optional[TradeSignal]:
        """Generate trading signal using neural network or fallback to simple analysis"""
        try:
            # Always try simple signal first (multi-timeframe consensus) - it's more reliable
            simple_signal = self._generate_simple_signal(market_data)
            
            # If simple signal has high confidence, use it
            if simple_signal and simple_signal.confidence >= self.confidence_threshold:
                return simple_signal
            
            # If neural brain is None or simple signal has low confidence, try neural
            if self.neural_brain is None:
                return simple_signal
            
            # Prepare data for neural analysis - ALL TIMEFRAMES USER TRADES ON
            m15_data = self._get_m15_data(market_data.symbol)
            h1_data = market_data.h1_data.copy() if not market_data.h1_data.empty else pd.DataFrame()
            h4_data = market_data.h4_data.copy() if not market_data.h4_data.empty else pd.DataFrame()
            d1_data = market_data.d1_data.copy() if not market_data.d1_data.empty else pd.DataFrame()
            
            # Generate neural signal with ALL TIMEFRAMES
            result = self.neural_brain.think(
                symbol=market_data.symbol,
                data_h1=m15_data,    # 15-minute timeframe as primary
                data_h4=h1_data,     # 1-hour timeframe as secondary
                data_d1=h4_data,     # 4-hour timeframe as tertiary (context)
                account_info={
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'free_margin': account_info.free_margin,
                    'margin_level': account_info.margin_level,
                    'currency': account_info.currency
                },
                symbol_info={
                    'bid': market_data.bid,
                    'ask': market_data.ask,
                    'spread': market_data.spread,
                    'volume': 0.01  # Default lot size
                }
            )
            
            if result is None:
                return None
            
            # Extract signal components
            decision = result.get('decision', 'HOLD')
            confidence = result.get('confidence', 0.0)
            lot_size = result.get('lot', 0.0)
            reason = result.get('reason', 'Neural analysis')
            
            # Only trade if confidence is high enough
            if confidence < self.confidence_threshold:
                return TradeSignal(
                    symbol=market_data.symbol,
                    action=TradeResult.HOLD,
                    confidence=confidence,
                    lot_size=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    reason=f"Low confidence: {confidence:.1%}",
                    timestamp=datetime.now()
                )
            
            # Calculate stop loss and take profit
            if decision == 'BUY':
                stop_loss = market_data.bid - (market_data.spread * 3)
                take_profit = market_data.ask + (market_data.spread * 6)
                action = TradeResult.BUY
            elif decision == 'SELL':
                stop_loss = market_data.ask + (market_data.spread * 3)
                take_profit = market_data.bid - (market_data.spread * 6)
                action = TradeResult.SELL
            else:
                # HOLD case - no stop loss/take profit needed
                return TradeSignal(
                    symbol=market_data.symbol,
                    action=TradeResult.HOLD,
                    confidence=confidence,
                    lot_size=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    reason=reason,
                    timestamp=datetime.now()
                )

            return TradeSignal(
                symbol=market_data.symbol,
                action=action,
                confidence=confidence,
                lot_size=max(0.01, float(lot_size) if lot_size else 0.01),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating neural signal: {e}")
            return None
    
    def _generate_simple_signal(self, market_data: MarketData) -> TradeSignal:
        """Generate simple signal as fallback using user's timeframes"""
        try:
            # Get 15-minute data for analysis
            m15_data = self._get_m15_data(market_data.symbol)
            
            # Check if we have data from the timeframes user trades on
            if market_data.h1_data.empty or market_data.h4_data.empty or m15_data.empty:
                return TradeSignal(
                    symbol=market_data.symbol,
                    action=TradeResult.HOLD,
                    confidence=0.0,
                    lot_size=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    reason="Insufficient data from user's timeframes (15m, 1h, 4h)",
                    timestamp=datetime.now()
                )
            
            # Multi-timeframe momentum analysis (user's timeframes)
            # 15-minute momentum
            m15_closes = m15_data['close'].values
            m15_ma5 = np.mean(m15_closes[-5:]) if len(m15_closes) >= 5 else 0
            m15_ma10 = np.mean(m15_closes[-10:]) if len(m15_closes) >= 10 else 0
            
            # 1-hour momentum
            h1_closes = market_data.h1_data['close'].values
            h1_ma5 = np.mean(h1_closes[-5:]) if len(h1_closes) >= 5 else 0
            h1_ma10 = np.mean(h1_closes[-10:]) if len(h1_closes) >= 10 else 0
            
            # 4-hour momentum
            h4_closes = market_data.h4_data['close'].values
            h4_ma5 = np.mean(h4_closes[-5:]) if len(h4_closes) >= 5 else 0
            h4_ma10 = np.mean(h4_closes[-10:]) if len(h4_closes) >= 10 else 0
            
            # Multi-timeframe consensus analysis
            # Check for agreement across timeframes
            m15_bullish = m15_ma5 > m15_ma10
            h1_bullish = h1_ma5 > h1_ma10
            h4_bullish = h4_ma5 > h4_ma10
            
            # Log the analysis for debugging
            logger.info(f"Symbol Analysis: {market_data.symbol}")
            logger.info(f"  M15: {m15_bullish} (MA5={m15_ma5:.5f}, MA10={m15_ma10:.5f})")
            logger.info(f"  H1: {h1_bullish} (MA5={h1_ma5:.5f}, MA10={h1_ma10:.5f})")
            logger.info(f"  H4: {h4_bullish} (MA5={h4_ma5:.5f}, MA10={h4_ma10:.5f})")
            
            # Determine majority direction (prevents bullish bias bug).
            bullish_count = int(m15_bullish) + int(h1_bullish) + int(h4_bullish)
            bearish_count = 3 - bullish_count

            if bullish_count >= 2:
                action = TradeResult.BUY
                confidence = 0.90 if bullish_count == 3 else 0.80
                reason = "Strong bullish (3/3 timeframes)" if bullish_count == 3 else "Bullish majority (2/3 timeframes)"
            elif bearish_count >= 2:
                action = TradeResult.SELL
                confidence = 0.90 if bearish_count == 3 else 0.80
                reason = "Strong bearish (3/3 timeframes)" if bearish_count == 3 else "Bearish majority (2/3 timeframes)"
            else:
                return TradeSignal(
                    symbol=market_data.symbol,
                    action=TradeResult.HOLD,
                    confidence=0.2,
                    lot_size=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    reason=f"No timeframe agreement (15m: {m15_bullish}, 1h: {h1_bullish}, 4h: {h4_bullish})",
                    timestamp=datetime.now()
                )

            # Calculate stop loss and take profit
            if action == TradeResult.BUY:
                stop_loss = market_data.bid - (market_data.spread * 3)
                take_profit = market_data.ask + (market_data.spread * 6)
            else:
                stop_loss = market_data.ask + (market_data.spread * 3)
                take_profit = market_data.bid - (market_data.spread * 6)

            return TradeSignal(
                symbol=market_data.symbol,
                action=action,
                confidence=confidence,
                lot_size=0.01,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                timestamp=datetime.now()
            )
                
        except Exception as e:
            logger.error(f"Error generating simple signal: {e}")
            return None
    
    def execute_trade(self, signal: TradeSignal, account_info: AccountInfo) -> bool:
        """Execute trade based on signal"""
        try:
            if signal.action == TradeResult.HOLD:
                return True
            
            # Risk management checks
            if not self._check_risk_conditions(signal, account_info):
                logger.warning(f"Trade rejected due to risk management: {signal.symbol}")
                return False
            
            # Prepare trade request
            symbol_info = mt5.symbol_info(signal.symbol)
            if symbol_info is None:
                logger.error(f"Symbol info not found: {signal.symbol}")
                return False
            
            # Calculate lot size based on risk
            lot_size = self._calculate_position_size(signal, account_info, symbol_info)
            
            if lot_size <= 0:
                logger.warning(f"Invalid lot size calculated: {lot_size}")
                return False
            
            # Prepare request - USE CORRECT FILLING MODE (FOK=0, IOC=1, RETURN=2)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if signal.action == TradeResult.BUY else mt5.ORDER_TYPE_SELL,
                "price": symbol_info.ask if signal.action == TradeResult.BUY else symbol_info.bid,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Neural-{signal.confidence:.1%}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,  # Use FOK (0) instead of IOC (1)
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logger.error(f"Order send failed: {signal.symbol}")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Trade failed: {result.retcode} - {result.comment}")
                return False
            
            # Log successful trade
            logger.info(f"Trade Executed: {signal.action.value} {signal.symbol}")
            logger.info(f"   Lot Size: {lot_size}")
            logger.info(f"   Confidence: {signal.confidence:.1%}")
            logger.info(f"   Reason: {signal.reason}")
            
            # Update positions
            self.positions[signal.symbol] = {
                'ticket': result.order,
                'action': signal.action,
                'lot_size': lot_size,
                'entry_price': result.price,
                'timestamp': datetime.now(),
                'confidence': signal.confidence,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            self.trade_history.append({
                'ticket': result.order,
                'symbol': signal.symbol,
                'action': signal.action.value,
                'lot_size': lot_size,
                'entry_price': result.price,
                'confidence': signal.confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def _check_risk_conditions(self, signal: TradeSignal, account_info: AccountInfo) -> bool:
        """Check if trade meets risk management conditions"""
        try:
            # Check max open positions
            if len(self.positions) >= self.max_open_positions:
                logger.warning(f"Risk check failed: Max positions reached ({len(self.positions)}/{self.max_open_positions})")
                return False
            
            # Check if already have position in this symbol
            if signal.symbol in self.positions:
                logger.warning(f"Risk check failed: Already have position in {signal.symbol}")
                return False
            
            # Check minimum confidence
            if signal.confidence < self.confidence_threshold:
                logger.warning(f"Risk check failed: Low confidence ({signal.confidence:.1%} < {self.confidence_threshold:.1%})")
                return False
            
            # Check account health (margin_level can be 0 when no positions are open)
            # Only check margin if margin is actually being used
            if account_info.margin > 0 and account_info.margin_level < 100:
                logger.warning(f"Risk check failed: Low margin level ({account_info.margin_level:.1f}%)")
                return False
            
            logger.info(f"Risk check passed for {signal.symbol}: confidence={signal.confidence:.1%}, positions={len(self.positions)}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk conditions: {e}")
            return False
    
    def _calculate_position_size(self, signal: TradeSignal, account_info: AccountInfo, symbol_info, base_lot: float = 0.01) -> float:
        """Calculate position size based on risk management"""
        try:
            # Base lot size
            lot_size = base_lot
            
            # Adjust based on confidence
            confidence_multiplier = signal.confidence / 0.78  # Scale to confidence threshold
            lot_size *= confidence_multiplier
            
            # Risk-based adjustment - USE CONFIGURED RISK (2% default)
            risk_factor = self.max_risk_per_trade  # Use configured risk, not 10%
            balance_risk = account_info.balance * risk_factor
            
            # Calculate pip distance
            pip_distance = abs(signal.take_profit - signal.stop_loss)
            if pip_distance == 0:
                pip_distance = symbol_info.point * 100  # Default to 100 pips
            
            # Calculate pip value for risk calculation
            tick_value = symbol_info.trade_tick_value
            risk_amount = pip_distance * tick_value * lot_size
            
            # Adjust lot size to match risk
            if risk_amount > 0:
                lot_size = min(lot_size, balance_risk / risk_amount)
            
            # Round to valid lot size
            lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))
            
            logger.info(f"Position size: {lot_size:.2f} lots (risk: {risk_factor:.1%}, balance: ${account_info.balance:.2f}")
            return round(lot_size, 2)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Safe fallback
    
    def monitor_positions(self):
        """Monitor open positions and manage exits"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
            
            for position in positions:
                symbol = position.symbol
                if symbol not in self.positions:
                    continue
                
                # Check if we should close position
                should_close = self._should_close_position(position)
                
                if should_close:
                    self._close_position(position)
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def _should_close_position(self, position) -> bool:
        """Determine if position should be closed"""
        try:
            current_price = position.price_current
            entry_price = position.price_open
            
            # Close if stop loss hit
            if position.sl > 0:
                if position.type == mt5.POSITION_TYPE_BUY and current_price <= position.sl:
                    return True
                elif position.type == mt5.POSITION_TYPE_SELL and current_price >= position.sl:
                    return True
            
            # Close if take profit hit
            if position.tp > 0:
                if position.type == mt5.POSITION_TYPE_BUY and current_price >= position.tp:
                    return True
                elif position.type == mt5.POSITION_TYPE_SELL and current_price <= position.tp:
                    return True
            
            # Close if profit target reached
            profit = position.profit
            if profit > 0:
                # Close at 50% of max expected profit (risk management)
                expected_profit = abs((position.tp - entry_price) if position.tp > 0 else 0)
                if expected_profit > 0 and profit >= expected_profit * 0.5:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking position closure: {e}")
            return False
    
    def _close_position(self, position):
        """Close a position"""
        try:
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "deviation": 20,
                "magic": 123456,
                "comment": "Neural Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,  # Use FOK (0) instead of IOC (1)
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position closed: {position.symbol} - Profit: {position.profit:.2f}")
                
                # Update performance metrics
                if position.profit > 0:
                    self.performance_metrics['winning_trades'] += 1
                else:
                    self.performance_metrics['losing_trades'] += 1
                
                self.performance_metrics['total_profit'] += position.profit
                
                # Remove from positions
                if position.symbol in self.positions:
                    del self.positions[position.symbol]
            else:
                logger.error(f"Failed to close position: {position.symbol}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            total_trades = self.performance_metrics['total_trades']
            winning_trades = self.performance_metrics['winning_trades']
            losing_trades = self.performance_metrics['losing_trades']
            
            if total_trades > 0:
                self.performance_metrics['win_rate'] = winning_trades / total_trades
            
            # Calculate proper profit factor
            # Profit Factor = Gross Profit / Gross Loss
            # Need to track separate totals
            total_profit = self.performance_metrics['total_profit']
            
            # If no losing trades and we have profit, profit factor is infinity (or very high)
            # If no losing trades and we have loss, profit factor is negative
            # If no winning trades, profit factor is 0
            
            if winning_trades > 0 and losing_trades > 0:
                # Simple estimate based on win rate and total profit
                if total_profit > 0:
                    # Assume asymmetric risk/reward ratio (e.g., 1:2)
                    expected_loss_per_trade = abs(total_profit) / (winning_trades / self.performance_metrics['win_rate'] - 1) if self.performance_metrics['win_rate'] < 1 else 0
                    if expected_loss_per_trade > 0:
                        avg_win = total_profit / winning_trades
                        self.performance_metrics['profit_factor'] = avg_win / expected_loss_per_trade
                    else:
                        self.performance_metrics['profit_factor'] = float('inf') if total_profit > 0 else 0
                else:
                    self.performance_metrics['profit_factor'] = 0.0
            elif winning_trades > 0 and total_profit > 0:
                # No losing trades yet = infinite profit factor
                self.performance_metrics['profit_factor'] = 99.99  # Cap for display
            elif losing_trades > 0 and total_profit < 0:
                # Only losing trades
                self.performance_metrics['profit_factor'] = 0.0
            else:
                self.performance_metrics['profit_factor'] = 0.0
            
            # Log performance summary
            logger.info("Performance Summary:")
            logger.info(f"   Total Trades: {self.performance_metrics['total_trades']}")
            logger.info(f"   Winning Trades: {self.performance_metrics['winning_trades']}")
            logger.info(f"   Losing Trades: {self.performance_metrics['losing_trades']}")
            logger.info(f"   Win Rate: {self.performance_metrics['win_rate']:.1%}")
            logger.info(f"   Total Profit: {self.performance_metrics['total_profit']:.2f}")
            pf = self.performance_metrics['profit_factor']
            pf_display = "INF" if pf > 99 else f"{pf:.2f}"
            logger.info(f"   Profit Factor: {pf_display}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def trading_loop(self):
        """Main trading loop"""
        logger.info("Starting Trading Loop...")
        cycle_count = 0
        
        while self.is_running:
            try:
                # Get account info
                account_info = self.get_account_info()
                if account_info is None:
                    logger.error("Could not get account info, skipping cycle")
                    time.sleep(5)
                    continue
                
                # Monitor and manage existing positions
                self.monitor_positions()
                
                # Process each symbol
                for symbol in self.symbols:
                    try:
                        # Get market data
                        market_data = self.get_market_data(symbol)
                        if market_data is None:
                            continue
                        
                        # Generate neural signal
                        signal = self.generate_neural_signal(market_data, account_info)
                        if signal is None:
                            continue
                        
                        # Log signal
                        if signal.action != TradeResult.HOLD:
                            logger.info(f"Signal: {signal.action.value} {symbol} - Confidence: {signal.confidence:.1%}")
                            logger.info(f"   Reason: {signal.reason}")
                        
                        # Execute trade if signal is strong
                        if signal.action in [TradeResult.BUY, TradeResult.SELL]:
                            success = self.execute_trade(signal, account_info)
                            if success:
                                logger.info(f"Trade executed successfully")
                            else:
                                logger.warning(f"Trade execution failed")
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Update performance metrics every 10 cycles
                cycle_count += 1
                if cycle_count % 10 == 0:
                    self.update_performance_metrics()
                
                # Sleep between cycles
                time.sleep(30)  # 30 second intervals
                
            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(10)
        
        logger.info("Trading Loop Stopped")
    
    def start_trading(self):
        """Start the trading bot"""
        try:
            logger.info("Starting Live Neural Trading Bot...")
            
            # Connect to MT5
            if not self.connect_to_mt5():
                logger.error("Failed to connect to MT5")
                return False

            # Sync after MT5 connection so existing positions are visible immediately.
            self._sync_positions_with_mt5()
            
            # Set running flag
            self.is_running = True
            
            # Start trading loop in separate thread
            trading_thread = threading.Thread(target=self.trading_loop)
            trading_thread.daemon = True
            trading_thread.start()
            
            logger.info("Trading Bot Started Successfully!")
            logger.info("Press Ctrl+C to stop trading")
            
            # Keep main thread alive
            try:
                while self.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping trading bot...")
                self.stop_trading()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            return False
    
    def stop_trading(self):
        """Stop the trading bot"""
        logger.info("Stopping Trading Bot...")
        self.is_running = False
        
        # Close all positions
        try:
            positions = mt5.positions_get()
            if positions is not None:
                for position in positions:
                    self._close_position(position)
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
        
        # Shutdown MT5
        mt5.shutdown()
        
        # Release lock file
        release_lock()
        
        logger.info("Trading Bot Stopped")

def main():
    """Main function to run the trading bot"""
    print("Live Neural Trading Bot - Starting...")
    print("Target: 78%+ Win Rate")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Check for single instance
    if not check_single_instance():
        print("ERROR: Another instance is already running!")
        print("Please close the other instance or remove the lock file: .neural_trading_bot.lock")
        return
    
    # Initialize trading bot with ALL FOREX PAIRS and enhanced features
    bot = LiveNeuralTradingBot(
        trading_mode=TradingMode.DEMO,  # DEMO trading for safety
        confidence_threshold=0.78,
        max_risk_per_trade=0.02,
        symbols=[
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY', 'CADCHF',
            'EURAUD', 'EURCAD', 'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPNZD',
            'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADJPY', 'CHFJPY', 'NZDJPY'
        ]  # ALL MAJOR FOREX PAIRS
    )
    
    try:
        # Start trading
        success = bot.start_trading()
        if success:
            print("Trading bot is now running!")
        else:
            print("Failed to start trading bot")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Release lock file
        release_lock()
        print("Trading bot shutdown complete")

if __name__ == "__main__":
    main()
