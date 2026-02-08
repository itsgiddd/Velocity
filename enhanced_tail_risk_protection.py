#!/usr/bin/env python3
"""
Enhanced Tail Risk Protection System
===================================

Advanced risk management system specifically designed to eliminate large downside scenarios
while maintaining upside potential. This system implements:

1. Dynamic confidence adjustment based on market volatility
2. Volatility clustering detection and position sizing
3. Advanced tail risk controls with real-time monitoring
4. Multi-timeframe risk validation
5. Performance-based risk scaling

Target: Transform p10 from -85.18% to positive territory while maintaining p90 potential
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NORMAL_VOLATILITY = "NORMAL_VOLATILITY" 
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"

class RiskLevel(Enum):
    """Risk level classification"""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class TailRiskProtector:
    """
    Advanced tail risk protection system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Risk thresholds for tail protection
        self.tail_risk_thresholds = {
            'max_daily_loss_pct': 0.02,      # 2% max daily loss
            'max_weekly_loss_pct': 0.05,      # 5% max weekly loss
            'max_drawdown_pct': 0.10,         # 10% max drawdown
            'min_win_rate': 0.65,             # 65% minimum win rate
            'max_consecutive_losses': 3,       # Max 3 consecutive losses
            'volatility_spike_threshold': 2.0  # 2x normal volatility
        }
        
        # Dynamic confidence adjustment parameters
        self.confidence_adjustment = {
            'base_threshold': 0.78,           # Base confidence threshold
            'volatility_multiplier': 0.15,     # How much volatility affects confidence
            'drawdown_penalty': 0.20,         # Penalty for drawdown
            'loss_streak_penalty': 0.25,      # Penalty for loss streaks
            'regime_adjustment': 0.10          # Adjustment based on regime
        }
        
        # Volatility clustering parameters
        self.volatility_clustering = {
            'short_window': 10,               # Short-term volatility window
            'long_window': 50,                # Long-term volatility window
            'cluster_threshold': 1.5,         # Volatility clustering threshold
            'persistence_threshold': 0.7      # Volatility persistence threshold
        }
        
        # Performance tracking
        self.performance_metrics = {
            'daily_returns': [],
            'weekly_returns': [],
            'drawdown_series': [],
            'volatility_regimes': [],
            'confidence_adjustments': [],
            'risk_scaling_events': []
        }
        
        # Risk state tracking
        self.risk_state = {
            'current_drawdown': 0.0,
            'current_volatility_regime': VolatilityRegime.NORMAL_VOLATILITY,
            'consecutive_losses': 0,
            'days_since_last_win': 0,
            'current_risk_level': RiskLevel.MODERATE,
            'last_confidence_adjustment': datetime.now(),
            'volatility_spike_detected': False,
            'account_reference_balance': 10000.0
        }
    
    def analyze_market_conditions(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive market condition analysis for risk assessment
        """
        try:
            if price_data is None or price_data.empty or 'close' not in price_data.columns:
                return self._get_default_risk_indicators()

            returns = price_data['close'].pct_change().dropna()
            if returns.empty:
                return self._get_default_risk_indicators()
            
            # Calculate multiple volatility measures
            short_vol = returns.rolling(self.volatility_clustering['short_window']).std().iloc[-1]
            long_vol = returns.rolling(self.volatility_clustering['long_window']).std().iloc[-1]
            current_vol = returns.rolling(20).std().iloc[-1]

            # Handle short history / NaN edge cases safely
            fallback_vol = float(returns.std()) if float(returns.std()) > 0 else 1e-8
            if np.isnan(short_vol) or short_vol <= 0:
                short_vol = fallback_vol
            if np.isnan(long_vol) or long_vol <= 0:
                long_vol = fallback_vol
            if np.isnan(current_vol) or current_vol <= 0:
                current_vol = fallback_vol
            
            # Volatility clustering detection
            vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
            vol_persistence = self._calculate_volatility_persistence(returns)
            
            # Market regime detection
            volatility_regime = self._classify_volatility_regime(current_vol, long_vol)
            trend_strength = self._calculate_trend_strength(price_data)
            
            # Risk indicators
            risk_indicators = {
                'current_volatility': current_vol,
                'volatility_ratio': vol_ratio,
                'volatility_persistence': vol_persistence,
                'trend_strength': trend_strength,
                'volatility_regime': volatility_regime,
                'market_stress': self._calculate_market_stress(returns, volatility_regime),
                'liquidity_risk': self._assess_liquidity_risk(price_data)
            }
            
            return risk_indicators
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return self._get_default_risk_indicators()
    
    def calculate_dynamic_confidence(self, base_confidence: float, 
                                   market_conditions: Dict[str, Any],
                                   performance_history: List[float]) -> float:
        """
        Calculate dynamic confidence threshold based on market conditions
        """
        try:
            adjusted_confidence = base_confidence
            
            # Volatility adjustment
            vol_regime = market_conditions.get('volatility_regime', VolatilityRegime.NORMAL_VOLATILITY)
            vol_multiplier = self._get_volatility_multiplier(vol_regime)
            adjusted_confidence += vol_multiplier * self.confidence_adjustment['volatility_multiplier']
            
            # Drawdown adjustment
            current_dd = self._calculate_current_drawdown(performance_history)
            if current_dd > 0.05:  # 5% drawdown
                drawdown_penalty = (current_dd - 0.05) * self.confidence_adjustment['drawdown_penalty']
                adjusted_confidence -= drawdown_penalty
            
            # Loss streak adjustment
            consecutive_losses = self.risk_state['consecutive_losses']
            if consecutive_losses > 0:
                loss_penalty = consecutive_losses * self.confidence_adjustment['loss_streak_penalty']
                adjusted_confidence -= loss_penalty
            
            # Market stress adjustment
            market_stress = market_conditions.get('market_stress', 0.0)
            if market_stress > 0.7:
                stress_penalty = (market_stress - 0.7) * 0.15
                adjusted_confidence -= stress_penalty
            
            # Ensure confidence stays within reasonable bounds
            adjusted_confidence = max(0.65, min(0.95, adjusted_confidence))
            
            # Record adjustment
            self.performance_metrics['confidence_adjustments'].append({
                'timestamp': datetime.now(),
                'base_confidence': base_confidence,
                'adjusted_confidence': adjusted_confidence,
                'volatility_regime': vol_regime.value,
                'drawdown': current_dd,
                'loss_streak': consecutive_losses,
                'market_stress': market_stress
            })
            
            return adjusted_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic confidence: {e}")
            return base_confidence
    
    def calculate_dynamic_position_size(self, base_lot_size: float,
                                       confidence: float,
                                       market_conditions: Dict[str, Any],
                                       account_balance: float) -> float:
        """
        Calculate dynamic position size based on risk factors
        """
        try:
            # Base risk per trade (1% of account)
            base_risk_pct = 0.01
            
            # Volatility adjustment
            vol_regime = market_conditions.get('volatility_regime', VolatilityRegime.NORMAL_VOLATILITY)
            vol_multiplier = self._get_volatility_position_multiplier(vol_regime)
            
            # Confidence adjustment
            confidence_multiplier = confidence ** 2  # Square confidence for stronger scaling
            
            # Drawdown adjustment
            current_dd = self._calculate_current_drawdown([])
            dd_multiplier = max(0.3, 1.0 - current_dd * 2)  # Reduce size as drawdown increases
            
            # Calculate risk-adjusted position size
            risk_adjusted_risk_pct = base_risk_pct * vol_multiplier * confidence_multiplier * dd_multiplier
            risk_adjusted_risk_pct = max(0.003, min(0.025, risk_adjusted_risk_pct))  # 0.3% to 2.5%
            
            # Calculate position size
            position_size = (account_balance * risk_adjusted_risk_pct) / (account_balance * 0.01) * base_lot_size
            
            # Ensure reasonable bounds
            position_size = max(0.01, min(1.0, position_size))
            
            # Record risk scaling event
            self.performance_metrics['risk_scaling_events'].append({
                'timestamp': datetime.now(),
                'base_lot_size': base_lot_size,
                'confidence': confidence,
                'volatility_regime': vol_regime.value,
                'drawdown': current_dd,
                'final_position_size': position_size,
                'risk_multiplier': risk_adjusted_risk_pct / base_risk_pct
            })
            
            return round(position_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic position size: {e}")
            return base_lot_size
    
    def validate_trade_signal(
        self,
        signal_confidence: Optional[float] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
        recent_performance: Optional[List[float]] = None,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive trade validation with tail risk protection
        """
        try:
            if signal_confidence is None:
                signal_confidence = confidence if confidence is not None else 0.0
            market_conditions = market_conditions or {}
            recent_performance = recent_performance or []

            validation_result = {
                'approved': False,
                'confidence_threshold': 0.78,
                'position_size_multiplier': 1.0,
                'risk_level': RiskLevel.MODERATE,
                'rejection_reasons': [],
                'warnings': []
            }
            
            # Check if we're in emergency mode
            if self._is_emergency_mode():
                validation_result['rejection_reasons'].append("Emergency mode active - trading suspended")
                return validation_result
            
            # Calculate dynamic confidence threshold
            dynamic_confidence = self.calculate_dynamic_confidence(
                self.confidence_adjustment['base_threshold'],
                market_conditions,
                recent_performance
            )
            validation_result['confidence_threshold'] = dynamic_confidence
            
            # Check confidence threshold
            if signal_confidence < dynamic_confidence:
                validation_result['rejection_reasons'].append(
                    f"Signal confidence {signal_confidence:.3f} below threshold {dynamic_confidence:.3f}"
                )
                return validation_result
            
            # Check volatility regime
            vol_regime = market_conditions.get('volatility_regime', VolatilityRegime.NORMAL_VOLATILITY)
            if vol_regime == VolatilityRegime.EXTREME_VOLATILITY:
                validation_result['rejection_reasons'].append("Extreme volatility - trading suspended")
                return validation_result
            elif vol_regime == VolatilityRegime.HIGH_VOLATILITY:
                validation_result['warnings'].append("High volatility detected - position size reduced")
                validation_result['position_size_multiplier'] = 0.5
            
            # Check current drawdown
            current_dd = self._calculate_current_drawdown(recent_performance)
            if current_dd > self.tail_risk_thresholds['max_drawdown_pct']:
                validation_result['rejection_reasons'].append(
                    f"Current drawdown {current_dd:.1%} exceeds limit"
                )
                return validation_result
            elif current_dd > 0.07:  # 7% drawdown
                validation_result['warnings'].append("High drawdown - position size reduced")
                validation_result['position_size_multiplier'] *= 0.6
            
            # Check loss streak
            if self.risk_state['consecutive_losses'] >= self.tail_risk_thresholds['max_consecutive_losses']:
                validation_result['rejection_reasons'].append(
                    f"Loss streak of {self.risk_state['consecutive_losses']} exceeds limit"
                )
                return validation_result
            
            # Check daily loss limit
            if self._check_daily_loss_limit():
                validation_result['rejection_reasons'].append("Daily loss limit reached")
                return validation_result
            
            # If we get here, trade is approved
            validation_result['approved'] = True
            validation_result['risk_level'] = self._determine_risk_level(market_conditions, recent_performance)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error in trade validation: {e}")
            return {'approved': False, 'rejection_reasons': [f"System error: {e}"]}
    
    def update_risk_state(self, trade_result: Dict[str, Any]):
        """
        Update risk state after trade execution
        """
        try:
            current_time = datetime.now()
            account_balance = float(trade_result.get('account_balance', 0.0) or 0.0)
            if account_balance > 0:
                self.risk_state['account_reference_balance'] = account_balance
            
            # Update consecutive losses
            if trade_result.get('is_loss', False):
                self.risk_state['consecutive_losses'] += 1
                self.risk_state['days_since_last_win'] += 1
            else:
                self.risk_state['consecutive_losses'] = 0
                self.risk_state['days_since_last_win'] = 0
            
            # Update drawdown
            pnl = trade_result.get('pnl', 0.0)
            self.performance_metrics['daily_returns'].append({
                'timestamp': trade_result.get('timestamp', current_time),
                'pnl': float(pnl),
                'is_loss': bool(trade_result.get('is_loss', False))
            })
            
            # Keep only recent returns (last 30 days)
            cutoff_date = current_time - timedelta(days=30)
            self.performance_metrics['daily_returns'] = [
                ret for ret in self.performance_metrics['daily_returns']
                if ret.get('timestamp', current_time) > cutoff_date
            ]
            
            # Update current drawdown
            pnl_history = [ret.get('pnl', 0.0) for ret in self.performance_metrics['daily_returns']]
            self.risk_state['current_drawdown'] = self._calculate_current_drawdown(
                pnl_history
            )
            
            # Update risk level
            self.risk_state['current_risk_level'] = self._determine_risk_level({}, pnl_history)
            
            self.logger.info(f"Risk state updated - Drawdown: {self.risk_state['current_drawdown']:.1%}, "
                           f"Loss streak: {self.risk_state['consecutive_losses']}")
                           
        except Exception as e:
            self.logger.error(f"Error updating risk state: {e}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report
        """
        try:
            return {
                'timestamp': datetime.now(),
                'current_risk_state': {
                    'drawdown': self.risk_state['current_drawdown'],
                    'volatility_regime': self.risk_state['current_volatility_regime'].value,
                    'consecutive_losses': self.risk_state['consecutive_losses'],
                    'risk_level': self.risk_state['current_risk_level'].value,
                    'days_since_last_win': self.risk_state['days_since_last_win']
                },
                'tail_risk_thresholds': self.tail_risk_thresholds,
                'recent_performance': {
                    'avg_daily_return': np.mean([r.get('pnl', 0) for r in self.performance_metrics['daily_returns'][-7:]]),
                    'win_rate': self._calculate_recent_win_rate(),
                    'max_drawdown': self._calculate_current_drawdown(
                        [r.get('pnl', 0) for r in self.performance_metrics['daily_returns']]
                    )
                },
                'confidence_adjustments': self.performance_metrics['confidence_adjustments'][-5:],
                'risk_scaling_events': self.performance_metrics['risk_scaling_events'][-5:]
            }
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {'error': str(e)}
    
    # Helper methods
    def _classify_volatility_regime(self, current_vol: float, long_vol: float) -> VolatilityRegime:
        """Classify current volatility regime"""
        vol_ratio = current_vol / long_vol if long_vol > 0 else 1.0
        
        # Use both relative and absolute volatility to avoid misclassifying
        # consistently volatile datasets as normal.
        if current_vol >= 0.008 or vol_ratio > 3.0:
            return VolatilityRegime.EXTREME_VOLATILITY
        elif current_vol >= 0.003 or vol_ratio > 1.6:
            return VolatilityRegime.HIGH_VOLATILITY
        elif current_vol <= 0.0005 or vol_ratio < 0.7:
            return VolatilityRegime.LOW_VOLATILITY
        else:
            return VolatilityRegime.NORMAL_VOLATILITY
    
    def _calculate_volatility_persistence(self, returns: pd.Series) -> float:
        """Calculate volatility clustering persistence"""
        try:
            vol = returns.rolling(10).std()
            vol_changes = vol.pct_change().dropna()
            
            # Calculate persistence as correlation of squared returns
            if len(vol_changes) > 10:
                persistence = vol_changes.autocorr(lag=1)
                return abs(persistence) if not np.isnan(persistence) else 0.0
            return 0.0
        except:
            return 0.0
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate trend strength indicator"""
        try:
            sma_20 = price_data['close'].rolling(20).mean().iloc[-1]
            sma_50 = price_data['close'].rolling(50).mean().iloc[-1]
            current_price = price_data['close'].iloc[-1]
            
            trend_strength = abs((sma_20 - sma_50) / sma_50)
            price_deviation = abs((current_price - sma_20) / sma_20)
            
            return (trend_strength + price_deviation) / 2
        except:
            return 0.0
    
    def _calculate_market_stress(self, returns: pd.Series, vol_regime: VolatilityRegime) -> float:
        """Calculate market stress indicator"""
        try:
            recent_vol = returns.rolling(10).std().iloc[-1]
            historical_vol = returns.rolling(50).std().mean()
            vol_spike = recent_vol / historical_vol if historical_vol > 0 else 1.0
            
            # Combine volatility spike with recent drawdown indicators
            volatility_stress = min(vol_spike / 2.0, 1.0)  # Normalize to 0-1
            
            # Add stress from extreme regimes
            regime_stress = {
                VolatilityRegime.EXTREME_VOLATILITY: 1.0,
                VolatilityRegime.HIGH_VOLATILITY: 0.7,
                VolatilityRegime.NORMAL_VOLATILITY: 0.3,
                VolatilityRegime.LOW_VOLATILITY: 0.1
            }
            
            total_stress = (volatility_stress + regime_stress.get(vol_regime, 0.5)) / 2
            return min(total_stress, 1.0)
        except:
            return 0.5
    
    def _assess_liquidity_risk(self, price_data: pd.DataFrame) -> float:
        """Assess liquidity risk based on price gaps and spreads"""
        try:
            # Calculate price gaps
            gaps = abs(price_data['close'].diff()) / price_data['close'].shift(1)
            avg_gap = gaps.rolling(20).mean().iloc[-1]
            recent_gap = gaps.iloc[-1]
            
            # Liquidity risk increases with unusual gaps
            liquidity_risk = min(recent_gap / (avg_gap * 3), 1.0) if avg_gap > 0 else 0.5
            return liquidity_risk
        except:
            return 0.5
    
    def _get_volatility_multiplier(self, vol_regime: VolatilityRegime) -> float:
        """Get confidence adjustment multiplier for volatility regime"""
        multipliers = {
            VolatilityRegime.EXTREME_VOLATILITY: -0.3,
            VolatilityRegime.HIGH_VOLATILITY: -0.15,
            VolatilityRegime.NORMAL_VOLATILITY: 0.0,
            VolatilityRegime.LOW_VOLATILITY: 0.1
        }
        return multipliers.get(vol_regime, 0.0)
    
    def _get_volatility_position_multiplier(self, vol_regime: VolatilityRegime) -> float:
        """Get position size multiplier for volatility regime"""
        multipliers = {
            VolatilityRegime.EXTREME_VOLATILITY: 0.0,  # No trading
            VolatilityRegime.HIGH_VOLATILITY: 0.3,
            VolatilityRegime.NORMAL_VOLATILITY: 0.8,
            VolatilityRegime.LOW_VOLATILITY: 1.2
        }
        return multipliers.get(vol_regime, 0.8)
    
    def _calculate_current_drawdown(self, performance_history: List[float]) -> float:
        """Calculate current drawdown from performance history"""
        if not performance_history:
            return self.risk_state['current_drawdown']
        
        try:
            reference_balance = float(self.risk_state.get('account_reference_balance', 10000.0) or 10000.0)
            equity_curve = reference_balance + np.cumsum(performance_history)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / np.maximum(peak, 1.0)
            return float(np.max(drawdown))
        except:
            return self.risk_state['current_drawdown']
    
    def _is_emergency_mode(self) -> bool:
        """Check if emergency mode should be activated"""
        return (
            self.risk_state['current_drawdown'] > 0.15 or  # 15% drawdown
            self.risk_state['consecutive_losses'] > 5 or   # 5+ consecutive losses
            self.risk_state['days_since_last_win'] > 10   # 10+ days without a win
        )
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        try:
            today = datetime.now().date()
            daily_returns = [
                ret for ret in self.performance_metrics['daily_returns']
                if ret.get('timestamp', datetime.now()).date() == today
            ]
            daily_loss = sum(ret.get('pnl', 0) for ret in daily_returns if ret.get('pnl', 0) < 0)
            reference_balance = float(self.risk_state.get('account_reference_balance', 10000.0) or 10000.0)
            daily_loss_pct = abs(daily_loss) / max(reference_balance, 1.0)
            return daily_loss_pct > self.tail_risk_thresholds['max_daily_loss_pct']
        except:
            return False
    
    def _determine_risk_level(self, market_conditions: Dict[str, Any], 
                             performance_history: List[float]) -> RiskLevel:
        """Determine current risk level"""
        drawdown = self._calculate_current_drawdown(performance_history)
        consecutive_losses = self.risk_state['consecutive_losses']
        
        if drawdown > 0.10 or consecutive_losses > 4:
            return RiskLevel.CRITICAL
        elif drawdown > 0.07 or consecutive_losses > 2:
            return RiskLevel.HIGH
        elif drawdown > 0.04 or consecutive_losses > 1:
            return RiskLevel.MODERATE
        elif drawdown > 0.02:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _calculate_recent_win_rate(self) -> float:
        """Calculate recent win rate"""
        try:
            recent_trades = self.performance_metrics['daily_returns'][-10:]
            if not recent_trades:
                return 0.5
            
            wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
            return wins / len(recent_trades)
        except:
            return 0.5
    
    def _get_default_risk_indicators(self) -> Dict[str, Any]:
        """Get default risk indicators for error cases"""
        return {
            'current_volatility': 0.01,
            'volatility_ratio': 1.0,
            'volatility_persistence': 0.5,
            'trend_strength': 0.0,
            'volatility_regime': VolatilityRegime.NORMAL_VOLATILITY,
            'market_stress': 0.5,
            'liquidity_risk': 0.5
        }
