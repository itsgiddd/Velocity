"""
Enhanced AI Brain with Multi-Timeframe Neural Network Analysis
Replaces traditional pattern recognition with neural network decision making
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from enhanced_neural_network import (
    CLASS_LABELS,
    EnhancedNeuralTrainer,
    MultiTimeframeFeatureExtractor,
)
from market_context import MarketContextAnalyzer
from trade_validator import TradeValidator
from adaptive_risk import AdaptiveRiskManager
from trading_memory import TradingMemory
from daily_planner import DailyPlanner

class EnhancedAIBrain:
    """
    Enhanced AI Brain using multi-timeframe neural network analysis
    and dynamic profit prediction
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Initialize components
        self.market_analyzer = MarketContextAnalyzer()
        self.trade_validator = TradeValidator()
        self.risk_manager = AdaptiveRiskManager()
        self.memory = TradingMemory()
        self.planner = DailyPlanner()
        
        # Initialize enhanced neural network
        self.neural_trainer = EnhancedNeuralTrainer(model_path)
        self.feature_extractor = MultiTimeframeFeatureExtractor()
        
        # Required data columns
        self.required_columns = {"open", "high", "low", "close"}

        # Decision quality guardrails
        self.min_rr = 2.0
        self.min_decision_margin = 0.08
        self.max_mean_risk = 0.75
        self.min_reward_to_cost = 1.2
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced AI Brain initialized with multi-timeframe neural network")
        
    def _sanitize_data(self, data: pd.DataFrame, name: str) -> pd.DataFrame:
        """Clean and validate market data"""
        if data is None or data.empty:
            raise ValueError(f"{name} data is empty")
            
        missing = self.required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} data missing columns: {', '.join(sorted(missing))}")
            
        cleaned = data.dropna(subset=list(self.required_columns)).copy()
        if cleaned.empty:
            raise ValueError(f"{name} data has no usable rows after cleanup")
            
        if "time" in cleaned.columns:
            cleaned = cleaned.sort_values("time")
            
        return cleaned
        
    def _validate_symbol_info(self, symbol_info) -> None:
        """Validate symbol information"""
        required_attrs = ("point", "volume_step", "volume_min", "volume_max", "trade_tick_value")
        missing = [attr for attr in required_attrs if not hasattr(symbol_info, attr)]
        if missing:
            raise ValueError(f"symbol_info missing fields: {', '.join(missing)}")
            
    def think(self, symbol: str, data_m15: pd.DataFrame, data_h1: pd.DataFrame, 
              data_h4: pd.DataFrame, data_d1: pd.DataFrame, account_info, symbol_info) -> Dict:
        """
        Enhanced thinking process using multi-timeframe neural network
        """
        try:
            # Validate and clean data
            data_m15 = self._sanitize_data(data_m15, "M15")
            data_h1 = self._sanitize_data(data_h1, "H1") 
            data_h4 = self._sanitize_data(data_h4, "H4")
            data_d1 = self._sanitize_data(data_d1, "D1")
            self._validate_symbol_info(symbol_info)
        except ValueError as exc:
            return {"decision": "REJECT", "reason": str(exc)}
            
        # Check memory (Revenge Trading / Kill Switch)
        if not self.memory.can_trade(symbol):
            return {"decision": "REJECT", "reason": "Memory Block (Loss Streak or Cooldown)"}
            
        # Check if we have enough data
        if len(data_m15) < 50 or len(data_h1) < 50 or len(data_h4) < 50 or len(data_d1) < 50:
            return {"decision": "WAIT", "reason": "Insufficient market history"}
            
        # Get daily bias (LONG / SHORT / NEUTRAL)
        daily_bias = self.planner.get_bias(symbol)
            
        # Get market context
        market_state = self.market_analyzer.get_market_state(symbol, data_h1, data_h4, data_d1)
        
        # Enhanced neural network analysis
        neural_analysis = self._neural_analysis(
            data_m15, data_h1, data_h4, data_d1, market_state, daily_bias
        )
        
        if not neural_analysis["should_trade"]:
            return {
                "decision": "REJECT", 
                "reason": neural_analysis["reason"],
                "neural_confidence": neural_analysis["confidence"],
                "timeframe_attention": neural_analysis.get("timeframe_attention", [0, 0, 0, 0])
            }
            
        # Calculate dynamic profit targets
        price = data_h1['close'].iloc[-1]
        sl, tp = self._calculate_dynamic_targets(neural_analysis, price, data_h1, symbol_info)
        
        # Validate risk/reward ratio
        risk = abs(price - sl)
        if risk == 0:
            return {"decision": "REJECT", "reason": "Zero risk distance"}
            
        reward = abs(tp - price)
        rr = reward / risk
        if rr < self.min_rr:
            return {
                "decision": "REJECT", 
                "reason": f"RR below 1:2 ({rr:.2f})",
                "neural_confidence": neural_analysis["confidence"]
            }

        spread_points = float(getattr(symbol_info, "spread", 0.0))
        point = float(getattr(symbol_info, "point", 0.0))
        transaction_cost = spread_points * point if spread_points > 0 and point > 0 else price * 0.00005
        reward_to_cost = reward / max(transaction_cost, 1e-8)
        if reward_to_cost < self.min_reward_to_cost:
            return {
                "decision": "REJECT",
                "reason": f"Reward/Cost too low ({reward_to_cost:.2f})",
                "neural_confidence": neural_analysis["confidence"],
            }
            
        # Calculate position size
        lot = self.risk_manager.calculate_lot_size(
            symbol, price, sl, neural_analysis["confidence"], account_info, symbol_info
        )
        
        if lot <= 0:
            return {"decision": "REJECT", "reason": "Risk Calc = 0 Lot"}
            
        return {
            "decision": "TRADE",
            "lot": lot,
            "sl": sl,
            "tp": tp,
            "reason": f"Neural Score {neural_analysis['confidence']:.3f} | RR {rr:.2f} | {daily_bias}",
            "confidence": neural_analysis["confidence"],
            "neural_analysis": neural_analysis,
            "market_state": market_state,
            "dynamic_targets": {
                "entry": price,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_reward": rr,
                "profit_potential": reward,
                "reward_to_cost": reward_to_cost
            }
        }
        
    def _neural_analysis(self, data_m15: pd.DataFrame, data_h1: pd.DataFrame, 
                        data_h4: pd.DataFrame, data_d1: pd.DataFrame, 
                        market_state: Dict, daily_bias: str) -> Dict:
        """
        Perform multi-timeframe neural network analysis
        """
        try:
            # Get neural network predictions
            predictions = self.neural_trainer.predict(data_m15, data_h1, data_h4, data_d1)
            
            # Extract prediction components
            trading_decision = predictions["trading_decision"]
            profit_targets = predictions["profit_targets"]
            risk_assessment = predictions["risk_assessment"]
            timeframe_attention = predictions["timeframe_attention"]
            confidence = predictions["confidence"]
            decision_margin = float(predictions.get("decision_margin", 0.0))
            uncertainty = float(predictions.get("uncertainty", 1.0))
            
            # Map neural decision to trading action
            decision_index = int(predictions.get("decision_index", int(np.argmax(trading_decision))))
            if decision_index < 0 or decision_index >= len(CLASS_LABELS):
                decision_index = int(np.argmax(trading_decision))
            neural_decision = CLASS_LABELS[decision_index]
            
            if neural_decision == "HOLD":
                return {
                    "should_trade": False,
                    "reason": "Neural decision is HOLD",
                    "confidence": confidence,
                    "timeframe_attention": timeframe_attention,
                }

            dynamic_margin_floor = self.min_decision_margin + (0.05 if daily_bias not in {"LONG", "SHORT"} else 0.0)
            if decision_margin < dynamic_margin_floor:
                return {
                    "should_trade": False,
                    "reason": (
                        f"Low decision margin ({decision_margin:.3f} < {dynamic_margin_floor:.2f})"
                    ),
                    "confidence": confidence,
                    "timeframe_attention": timeframe_attention,
                }

            # Enforce bias only when operator provides a directional bias.
            if daily_bias in {"LONG", "SHORT"}:
                bias_aligned = (
                    (daily_bias == "LONG" and neural_decision == "BUY") or
                    (daily_bias == "SHORT" and neural_decision == "SELL")
                )
            else:
                bias_aligned = True

            if not bias_aligned:
                return {
                    "should_trade": False,
                    "reason": f"Neural decision {neural_decision} conflicts with {daily_bias} bias",
                    "confidence": confidence,
                    "timeframe_attention": timeframe_attention
                }
                
            # Additional validation using market context
            context_valid = self._validate_neural_with_context(neural_decision, market_state)
            if not context_valid:
                return {
                    "should_trade": False,
                    "reason": "Neural decision rejected by market context validation",
                    "confidence": confidence * 0.5,  # Reduce confidence
                    "timeframe_attention": timeframe_attention
                }
                
            # Check minimum confidence threshold
            min_conf = 0.6 if daily_bias in {"LONG", "SHORT"} else 0.65
            if confidence < min_conf:
                return {
                    "should_trade": False,
                    "reason": f"Neural confidence too low ({confidence:.3f} < {min_conf:.2f})",
                    "confidence": confidence,
                    "timeframe_attention": timeframe_attention
                }

            mean_risk = float(np.mean(risk_assessment)) if len(np.atleast_1d(risk_assessment)) else 0.5
            if mean_risk > self.max_mean_risk:
                return {
                    "should_trade": False,
                    "reason": f"Risk assessment too high ({mean_risk:.3f})",
                    "confidence": confidence,
                    "timeframe_attention": timeframe_attention,
                }
                
            return {
                "should_trade": True,
                "decision": neural_decision,
                "confidence": confidence,
                "decision_margin": decision_margin,
                "uncertainty": uncertainty,
                "mean_risk": mean_risk,
                "profit_targets": profit_targets,
                "risk_assessment": risk_assessment,
                "timeframe_attention": timeframe_attention,
                "trading_decision_vector": trading_decision,
                "reason": f"Neural {neural_decision} with {confidence:.3f} confidence"
            }
            
        except Exception as e:
            self.logger.error(f"Neural analysis failed: {str(e)}")
            return {
                "should_trade": False,
                "reason": f"Neural analysis error: {str(e)}",
                "confidence": 0.0,
                "timeframe_attention": [0.25, 0.25, 0.25, 0.25]
            }
            
    def _validate_neural_with_context(self, neural_decision: str, market_state: Dict) -> bool:
        """
        Validate neural network decision with market context
        """
        # Get market session score
        session_score = market_state.get('session', 0)
        
        # Get strength score
        strength_score = market_state.get('strength', 0)
        
        # Require minimum market quality for trading
        if session_score < 0.3:
            return False
            
        if strength_score < 0.4:
            return False
            
        # For BUY decisions, require some bullish sentiment
        if neural_decision == "BUY" and market_state.get('sentiment', 'neutral') == 'bearish':
            return False
            
        # For SELL decisions, require some bearish sentiment
        if neural_decision == "SELL" and market_state.get('sentiment', 'neutral') == 'bullish':
            return False
            
        return True
        
    def _calculate_dynamic_targets(
        self,
        neural_analysis: Dict,
        current_price: float,
        data_h1: pd.DataFrame,
        symbol_info=None,
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit based on neural predictions
        """
        profit_targets = neural_analysis.get("profit_targets", [0, 0, 0, 0])
        
        # Extract neural predictions: [entry_signal, sl_offset, tp_offset, confidence]
        if len(profit_targets) >= 4:
            entry_signal = profit_targets[0]  # -1 to 1
            sl_offset = profit_targets[1]    # -1 to 1  
            tp_offset = profit_targets[2]    # -1 to 1
            neural_confidence = abs(profit_targets[3])  # 0 to 1
        else:
            # Fallback to default values
            entry_signal = 0
            sl_offset = -0.01  # 1% stop loss
            tp_offset = 0.02   # 2% take profit
            neural_confidence = 0.5
            
        decision = neural_analysis.get("decision", "HOLD")
        neural_confidence = float(np.clip(neural_confidence, 0.0, 1.0))
        sl_offset = float(np.clip(sl_offset, -1.0, 1.0))
        tp_offset = float(np.clip(tp_offset, -1.0, 1.0))

        atr = self._estimate_atr(data_h1)
        volatility = self._calculate_recent_volatility(data_h1)
        # Base distance scales by ATR and recent volatility, avoiding pair-dependent %
        base_distance = max(atr, current_price * max(volatility * 2.0, 0.0005))
        point = float(getattr(symbol_info, "point", 0.0)) if symbol_info is not None else 0.0
        min_distance = max(base_distance * 0.25, point * 20 if point > 0 else 0.0, current_price * 0.0002)
        mean_risk = float(neural_analysis.get("mean_risk", 0.5))
        risk_adjustment = float(np.clip(1.0 + (mean_risk - 0.5), 0.7, 1.4))

        # Keep SL tighter than TP by default to protect RR.
        sl_multiplier = (0.9 + abs(sl_offset) * 0.8) * risk_adjustment
        tp_multiplier = (1.8 + abs(tp_offset) * 1.6) * (1.2 - min(mean_risk, 0.8) * 0.25)
        confidence_multiplier = 0.8 + neural_confidence * 0.6
         
        # Calculate dynamic stop loss and take profit
        if decision == "BUY":
            sl = current_price - max(base_distance * sl_multiplier, min_distance)
            tp = current_price + max(base_distance * tp_multiplier * confidence_multiplier, min_distance * 2.0)
             
        elif decision == "SELL":
            sl = current_price + max(base_distance * sl_multiplier, min_distance)
            tp = current_price - max(base_distance * tp_multiplier * confidence_multiplier, min_distance * 2.0)
             
        else:
            # Fallback for HOLD or unknown decisions
            sl = current_price - max(base_distance, min_distance)
            tp = current_price + max(base_distance * 2.0, min_distance * 2.0)
             
        return sl, tp

    def _estimate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        if len(data) < period + 1:
            close = float(data["close"].iloc[-1]) if len(data) > 0 else 1.0
            return close * 0.001

        highs = data["high"].astype(float).values
        lows = data["low"].astype(float).values
        closes = data["close"].astype(float).values
        tr_values = []
        for i in range(len(data) - period, len(data)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1]) if i > 0 else 0.0
            low_close = abs(lows[i] - closes[i - 1]) if i > 0 else 0.0
            tr_values.append(max(high_low, high_close, low_close))
        atr = float(np.mean(tr_values)) if tr_values else float(data["close"].iloc[-1]) * 0.001
        return max(atr, float(data["close"].iloc[-1]) * 0.0002)
        
    def _calculate_recent_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate recent price volatility"""
        if len(data) < period:
            return 0.01  # Default 1% volatility
            
        closes = data['close'].values[-period:]
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns)
        
        return max(float(volatility), 0.0005)  # Minimum 0.05% volatility
        
    def log_result(self, symbol: str, result: Dict, profit: float):
        """Log trading result for memory system"""
        self.memory.close_trade(symbol, profit)
        
    def set_daily_plan(self, plan: dict):
        """Set the daily plan from external source"""
        self.planner.set_plan(plan)
        
    def save_model(self, path: str):
        """Save the trained neural network model"""
        self.neural_trainer.save_model(path)
        self.logger.info(f"Enhanced neural model saved to {path}")
        
    def load_model(self, path: str):
        """Load a pre-trained neural network model"""
        self.neural_trainer.load_model(path)
        self.logger.info(f"Enhanced neural model loaded from {path}")

def create_enhanced_training_data(symbol: str, historical_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """
    Create training data for the enhanced neural network
    """
    training_data = []
    
    # Get all timeframes
    timeframes = ['m15_data', 'h1_data', 'h4_data', 'd1_data']
    
    for i in range(100, min(len(historical_data['h1_data']), 1000)):  # Use last 900 data points
        sample = {}
        
        # Extract data for each timeframe
        for tf in timeframes:
            if tf in historical_data:
                data = historical_data[tf]
                start_idx = max(0, i - 100)
                end_idx = i
                sample[tf] = data.iloc[start_idx:end_idx].copy()
        
        # Create labels based on future price movement
        future_data = historical_data['h1_data'].iloc[i:i+24]  # 24 hours ahead
        
        if len(future_data) >= 24:
            current_price = historical_data['h1_data']['close'].iloc[i]
            future_price = future_data['close'].iloc[-1]
            
            price_change = (future_price - current_price) / current_price
            
            # Create decision label
            if price_change > 0.002:  # > 0.2% gain
                decision = 0  # BUY
            elif price_change < -0.002:  # < -0.2% loss
                decision = 1  # SELL
            else:
                decision = 2  # HOLD
                
            future_high = float(future_data['high'].max())
            future_low = float(future_data['low'].min())
            downside = max((current_price - future_low) / (current_price + 1e-8), 0.0)
            upside = max((future_high - current_price) / (current_price + 1e-8), 0.0)

            # Create profit target labels
            profit_targets = [
                float(np.clip(price_change, -1.0, 1.0)),           # Entry signal
                float(np.clip(-downside * 40.0, -1.0, 1.0)),       # Stop-loss offset proxy
                float(np.clip(upside * 40.0, -1.0, 1.0)),          # Take-profit offset proxy
                float(np.clip(abs(price_change) * 20.0, 0.0, 1.0))  # Confidence proxy
            ]
            
            sample['decision'] = decision
            sample['profit_targets'] = profit_targets
            
            training_data.append(sample)
    
    return training_data

def main():
    """Demo the enhanced AI brain"""
    # Create sample market data
    np.random.seed(42)
    sample_size = 200
    
    def create_sample_data(base_price=150):
        data = {
            'open': np.random.normal(base_price, 0.5, sample_size),
            'high': np.random.normal(base_price + 0.3, 0.5, sample_size),
            'low': np.random.normal(base_price - 0.3, 0.5, sample_size),
            'close': np.random.normal(base_price, 0.5, sample_size),
            'tick_volume': np.random.randint(100, 1000, sample_size)
        }
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        for i in range(sample_size):
            data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
            data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
            
        return pd.DataFrame(data)
    
    # Create sample data for all timeframes
    data_m15 = create_sample_data(150)
    data_h1 = create_sample_data(150)
    data_h4 = create_sample_data(150)
    data_d1 = create_sample_data(150)
    
    # Create mock account and symbol info
    class MockAccountInfo:
        balance = 10000
        equity = 10000
        margin = 0
        free_margin = 10000
        
    class MockSymbolInfo:
        point = 0.00001
        volume_step = 0.01
        volume_min = 0.01
        volume_max = 100.0
        trade_tick_value = 1.0
    
    # Initialize enhanced AI brain
    ai_brain = EnhancedAIBrain()
    
    # Set daily plan
    daily_plan = {"EURUSD": "LONG", "GBPUSD": "LONG", "USDJPY": "LONG"}
    ai_brain.set_daily_plan(daily_plan)
    
    # Make trading decision
    result = ai_brain.think(
        "EURUSD", data_m15, data_h1, data_h4, data_d1, 
        MockAccountInfo(), MockSymbolInfo()
    )
    
    print("Enhanced AI Brain Decision:")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result.get('reason', 'N/A')}")
    if 'confidence' in result:
        print(f"Neural Confidence: {result['confidence']:.3f}")
    if 'dynamic_targets' in result:
        targets = result['dynamic_targets']
        print(f"Entry: {targets['entry']:.5f}")
        print(f"Stop Loss: {targets['stop_loss']:.5f}")
        print(f"Take Profit: {targets['take_profit']:.5f}")
        print(f"Risk/Reward: {targets['risk_reward']:.2f}")

if __name__ == "__main__":
    main()
