#!/usr/bin/env python3
"""
Advanced Performance Tracking and Alerting System
==============================================

Comprehensive performance monitoring system with predictive analytics,
intelligent alerting, and real-time risk assessment for the neural trading system.

Features:
1. Real-time performance metrics calculation
2. Predictive analytics for performance forecasting
3. Intelligent alerting based on risk thresholds
4. Advanced risk scoring and early warning systems
5. Performance attribution analysis
6. Regime-based performance tracking
7. Portfolio-level risk monitoring
8. Automated risk mitigation triggers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class PerformanceMetric(Enum):
    """Key performance metrics"""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    RECOVERY_FACTOR = "recovery_factor"
    TAIL_RISK_SCORE = "tail_risk_score"
    VOLATILITY_REGIME_SCORE = "volatility_regime_score"

@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    metric: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    action_required: bool = False
    auto_resolved: bool = False

@dataclass
class PerformanceSnapshot:
    """Performance snapshot data"""
    timestamp: datetime
    total_return: float
    daily_return: float
    cumulative_return: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    tail_risk_score: float
    volatility_regime: str
    risk_level: str

class AdvancedPerformanceTracker:
    """
    Advanced performance tracking and alerting system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Performance data storage
        self.performance_history: List[PerformanceSnapshot] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.risk_events: List[Alert] = []
        self.performance_alerts: List[Alert] = []
        
        # Performance thresholds for alerting
        self.alert_thresholds = {
            'max_daily_loss': -0.03,      # -3% daily loss
            'max_drawdown': -0.10,        # -10% drawdown
            'min_sharpe_ratio': 0.5,     # Minimum Sharpe ratio
            'min_win_rate': 0.60,         # Minimum win rate
            'max_tail_risk_score': 0.7,   # Maximum tail risk score
            'min_profit_factor': 1.2,      # Minimum profit factor
            'max_volatility_regime_risk': 0.8  # Maximum volatility regime risk
        }
        
        # Predictive analytics parameters
        self.prediction_window = 30  # days
        self.regime_detection_window = 50  # trades
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Performance attribution
        self.regime_performance: Dict[str, Dict[str, float]] = {}
        self.symbol_performance: Dict[str, Dict[str, float]] = {}
        self.time_based_performance: Dict[str, Dict[str, float]] = {}
        
        self.logger.info("Advanced Performance Tracker initialized")
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """
        Record a trade for performance tracking
        
        Args:
            trade_data: Dictionary containing trade information
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now()
            
            # Calculate additional metrics
            trade_data['return_pct'] = trade_data.get('pnl', 0) / trade_data.get('entry_value', 1)
            trade_data['risk_reward'] = abs(trade_data.get('pnl', 0) / trade_data.get('risk_amount', 1))
            
            self.trade_history.append(trade_data)
            
            # Keep only recent trades (last 1000)
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
            # Update performance tracking
            self._update_performance_metrics()
            
            # Check for alerts
            self._check_trade_alerts(trade_data)
            
            # Update regime performance
            self._update_regime_performance(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def record_risk_event(self, risk_data: Dict[str, Any]):
        """
        Record a risk event for tracking and alerting
        
        Args:
            risk_data: Dictionary containing risk event information
        """
        try:
            alert = Alert(
                timestamp=datetime.now(),
                level=AlertLevel(risk_data.get('level', 'WARNING')),
                title=risk_data.get('title', 'Risk Event'),
                message=risk_data.get('message', ''),
                metric=risk_data.get('metric'),
                value=risk_data.get('value'),
                threshold=risk_data.get('threshold'),
                action_required=risk_data.get('action_required', False)
            )
            
            self.risk_events.append(alert)
            
            # Trigger alerts
            self._trigger_alert(alert)
            
            # Keep only recent events (last 100)
            if len(self.risk_events) > 100:
                self.risk_events = self.risk_events[-100:]
                
        except Exception as e:
            self.logger.error(f"Error recording risk event: {e}")
    
    def get_current_performance_snapshot(self) -> PerformanceSnapshot:
        """
        Calculate current performance snapshot
        """
        try:
            if not self.trade_history:
                return self._get_default_snapshot()
            
            recent_trades = self.trade_history[-30:]  # Last 30 trades
            
            # Calculate basic metrics
            total_return = sum(trade.get('pnl', 0) for trade in recent_trades)
            daily_returns = self._calculate_daily_returns(recent_trades)
            
            # Calculate performance metrics
            win_rate = self._calculate_win_rate(recent_trades)
            max_drawdown = self._calculate_max_drawdown(recent_trades)
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            sortino_ratio = self._calculate_sortino_ratio(daily_returns)
            profit_factor = self._calculate_profit_factor(recent_trades)
            
            # Advanced metrics
            tail_risk_score = self._calculate_tail_risk_score(daily_returns)
            volatility_regime = self._detect_current_volatility_regime()
            risk_level = self._assess_current_risk_level(
                max_drawdown=max_drawdown,
                tail_risk_score=tail_risk_score
            )
            
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                total_return=total_return,
                daily_return=np.mean(daily_returns) if daily_returns else 0,
                cumulative_return=self._calculate_cumulative_return(recent_trades),
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                profit_factor=profit_factor,
                tail_risk_score=tail_risk_score,
                volatility_regime=volatility_regime,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance snapshot: {e}")
            return self._get_default_snapshot()
    
    def predict_performance(self, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Predict future performance based on historical data
        """
        try:
            if len(self.trade_history) < 20:
                return {'error': 'Insufficient data for prediction'}
            
            # Prepare data for prediction
            daily_returns = self._calculate_daily_returns(self.trade_history)
            if len(daily_returns) < 10:
                return {'error': 'Insufficient return data for prediction'}
            
            # Simple trend analysis (can be enhanced with ML models)
            recent_returns = daily_returns[-20:]
            avg_return = np.mean(recent_returns)
            return_volatility = np.std(recent_returns)
            
            # Calculate trend
            x = np.arange(len(recent_returns))
            trend_coef = np.polyfit(x, recent_returns, 1)[0]
            
            # Predict metrics
            predicted_return = avg_return * days_ahead + trend_coef * days_ahead * 0.5
            predicted_volatility = return_volatility * np.sqrt(days_ahead)
            
            # Risk predictions
            predicted_max_drawdown = min(0, predicted_return - 2 * predicted_volatility)
            predicted_sharpe = predicted_return / predicted_volatility if predicted_volatility > 0 else 0
            
            # Confidence intervals (simplified)
            confidence_lower = predicted_return - 1.96 * predicted_volatility
            confidence_upper = predicted_return + 1.96 * predicted_volatility
            
            return {
                'prediction_horizon_days': days_ahead,
                'predicted_return': predicted_return,
                'predicted_volatility': predicted_volatility,
                'predicted_max_drawdown': predicted_max_drawdown,
                'predicted_sharpe_ratio': predicted_sharpe,
                'confidence_interval_95': {
                    'lower': confidence_lower,
                    'upper': confidence_upper
                },
                'trend_direction': 'positive' if trend_coef > 0 else 'negative',
                'confidence_level': min(len(daily_returns) / 100, 1.0),
                'risk_factors': self._identify_risk_factors()
            }
            
        except Exception as e:
            self.logger.error(f"Error in performance prediction: {e}")
            return {'error': str(e)}
    
    def get_performance_attribution(self) -> Dict[str, Any]:
        """
        Analyze performance attribution by different factors
        """
        try:
            attribution = {
                'regime_attribution': self.regime_performance,
                'symbol_attribution': self.symbol_performance,
                'time_based_attribution': self.time_based_performance,
                'risk_contribution': self._calculate_risk_contribution(),
                'overall_metrics': self._calculate_overall_metrics()
            }
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"Error calculating performance attribution: {e}")
            return {'error': str(e)}
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health check
        """
        try:
            current_snapshot = self.get_current_performance_snapshot()
            
            health_status = {
                'overall_status': 'HEALTHY',
                'timestamp': datetime.now(),
                'alerts': [],
                'recommendations': []
            }
            
            # Check various health metrics
            health_checks = [
                self._check_drawdown_health(current_snapshot),
                self._check_win_rate_health(current_snapshot),
                self._check_sharpe_health(current_snapshot),
                self._check_tail_risk_health(current_snapshot),
                self._check_volatility_health(current_snapshot)
            ]
            
            for check in health_checks:
                if not check['healthy']:
                    health_status['alerts'].append(check)
                    if check['severity'] == 'CRITICAL':
                        health_status['overall_status'] = 'CRITICAL'
                    elif check['severity'] == 'WARNING' and health_status['overall_status'] == 'HEALTHY':
                        health_status['overall_status'] = 'WARNING'
            
            # Add recommendations
            health_status['recommendations'] = self._generate_health_recommendations(health_status['alerts'])
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in system health check: {e}")
            return {'overall_status': 'ERROR', 'error': str(e)}
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Add callback function for alerts
        """
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """
        Get active alerts, optionally filtered by level
        """
        all_alerts = self.risk_events + self.performance_alerts
        if level:
            all_alerts = [alert for alert in all_alerts if alert.level == level]
        return sorted(all_alerts, key=lambda x: x.timestamp, reverse=True)
    
    # Private helper methods
    
    def _update_performance_metrics(self):
        """Update performance metrics after new trade"""
        try:
            current_snapshot = self.get_current_performance_snapshot()
            self.performance_history.append(current_snapshot)
            
            # Keep only recent snapshots (last 100)
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _check_trade_alerts(self, trade_data: Dict[str, Any]):
        """Check for alerts based on trade data"""
        try:
            pnl = trade_data.get('pnl', 0)
            symbol = trade_data.get('symbol', 'UNKNOWN')
            
            # Large loss alert
            if pnl < -0.02 * trade_data.get('entry_value', 10000):  # 2% loss
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    title="Large Loss Detected",
                    message=f"Large loss of {pnl:.2f} on {symbol}",
                    metric="daily_pnl",
                    value=pnl,
                    threshold=-0.02,
                    action_required=True
                )
                self.performance_alerts.append(alert)
                self._trigger_alert(alert)
            
            # Winning streak check
            recent_trades = [t for t in self.trade_history[-10:] if t.get('pnl', 0) > 0]
            if len(recent_trades) >= 7:  # 7 winning trades in a row
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.INFO,
                    title="Winning Streak Detected",
                    message=f"7 winning trades in a row detected",
                    action_required=False
                )
                self.performance_alerts.append(alert)
                self._trigger_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Error checking trade alerts: {e}")
    
    def _update_regime_performance(self, trade_data: Dict[str, Any]):
        """Update performance attribution by regime"""
        try:
            # This would be enhanced with actual regime detection
            regime = trade_data.get('volatility_regime', 'NORMAL')
            symbol = trade_data.get('symbol', 'UNKNOWN')
            
            # Update regime performance
            if regime not in self.regime_performance:
                self.regime_performance[regime] = {'trades': 0, 'total_pnl': 0, 'wins': 0}
            
            self.regime_performance[regime]['trades'] += 1
            self.regime_performance[regime]['total_pnl'] += trade_data.get('pnl', 0)
            if trade_data.get('pnl', 0) > 0:
                self.regime_performance[regime]['wins'] += 1
            
            # Update symbol performance
            if symbol not in self.symbol_performance:
                self.symbol_performance[symbol] = {'trades': 0, 'total_pnl': 0, 'wins': 0}
            
            self.symbol_performance[symbol]['trades'] += 1
            self.symbol_performance[symbol]['total_pnl'] += trade_data.get('pnl', 0)
            if trade_data.get('pnl', 0) > 0:
                self.symbol_performance[symbol]['wins'] += 1
                
        except Exception as e:
            self.logger.error(f"Error updating regime performance: {e}")
    
    def _trigger_alert(self, alert: Alert):
        """Trigger alert and notify callbacks"""
        try:
            # Log alert
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.CRITICAL: logging.ERROR,
                AlertLevel.EMERGENCY: logging.CRITICAL
            }.get(alert.level, logging.WARNING)
            
            self.logger.log(log_level, f"ALERT [{alert.level.value}] {alert.title}: {alert.message}")
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    def _calculate_daily_returns(self, trades: List[Dict[str, Any]]) -> List[float]:
        """Calculate daily returns from trade history"""
        try:
            # Group trades by date
            daily_pnls = {}
            for trade in trades:
                timestamp = trade.get('timestamp', datetime.now())
                if not isinstance(timestamp, datetime):
                    try:
                        timestamp = pd.to_datetime(timestamp).to_pydatetime()
                    except Exception:
                        timestamp = datetime.now()
                date = timestamp.date()
                daily_pnls[date] = daily_pnls.get(date, 0) + trade.get('pnl', 0)
            
            # Convert to returns (assuming $10,000 base)
            base_value = 10000
            ordered_dates = sorted(daily_pnls.keys())
            returns = [daily_pnls[day] / base_value for day in ordered_dates]
            return returns
            
        except Exception as e:
            self.logger.error(f"Error calculating daily returns: {e}")
            return []
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate"""
        try:
            if not trades:
                return 0.0
            
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            return winning_trades / len(trades)
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative_pnl = 0
            peak = 0
            max_drawdown = 0
            
            for trade in trades:
                cumulative_pnl += trade.get('pnl', 0)
                peak = max(peak, cumulative_pnl)
                drawdown = (peak - cumulative_pnl) / 10000  # Assuming $10k base
                max_drawdown = max(max_drawdown, drawdown)
            
            return -max_drawdown  # Negative value for drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            avg_return = np.mean(returns)
            return_vol = np.std(returns)
            
            if return_vol == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming daily returns)
            return avg_return / return_vol * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            avg_return = np.mean(returns)
            downside_returns = [r for r in returns if r < 0]
            
            if not downside_returns:
                return float('inf') if avg_return > 0 else 0.0
            
            downside_deviation = np.std(downside_returns)
            if downside_deviation == 0:
                return 0.0
            
            # Annualized Sortino ratio
            return avg_return / downside_deviation * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor"""
        try:
            gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
            gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            return gross_profit / gross_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {e}")
            return 0.0
    
    def _calculate_tail_risk_score(self, returns: List[float]) -> float:
        """Calculate tail risk score (Value at Risk based)"""
        try:
            if len(returns) < 10:
                return 0.5
            
            # Calculate 5th percentile (VaR)
            var_5 = np.percentile(returns, 5)
            
            # Normalize to 0-1 scale
            # Worst case scenario: -10% daily loss = score of 1.0
            tail_risk_score = min(abs(var_5) / 0.10, 1.0)
            
            return tail_risk_score
            
        except Exception as e:
            self.logger.error(f"Error calculating tail risk score: {e}")
            return 0.5
    
    def _detect_current_volatility_regime(self) -> str:
        """Detect current volatility regime"""
        try:
            if len(self.trade_history) < 20:
                return 'UNKNOWN'
            
            recent_returns = [trade.get('pnl', 0) for trade in self.trade_history[-20:]]
            volatility = np.std(recent_returns)
            
            if volatility > 200:
                return 'HIGH_VOLATILITY'
            elif volatility < 50:
                return 'LOW_VOLATILITY'
            else:
                return 'NORMAL_VOLATILITY'
                
        except Exception as e:
            self.logger.error(f"Error detecting volatility regime: {e}")
            return 'UNKNOWN'
    
    def _assess_current_risk_level(
        self,
        max_drawdown: Optional[float] = None,
        tail_risk_score: Optional[float] = None
    ) -> str:
        """Assess current risk level"""
        try:
            if max_drawdown is None or tail_risk_score is None:
                if self.performance_history:
                    latest_snapshot = self.performance_history[-1]
                    max_drawdown = latest_snapshot.max_drawdown
                    tail_risk_score = latest_snapshot.tail_risk_score
                elif self.trade_history:
                    recent_trades = self.trade_history[-30:]
                    max_drawdown = self._calculate_max_drawdown(recent_trades)
                    daily_returns = self._calculate_daily_returns(recent_trades)
                    tail_risk_score = self._calculate_tail_risk_score(daily_returns)
                else:
                    return 'LOW'
            
            if max_drawdown < -0.08:
                return 'HIGH'
            elif max_drawdown < -0.04:
                return 'MODERATE'
            elif tail_risk_score > 0.7:
                return 'HIGH'
            else:
                return 'LOW'
                
        except Exception as e:
            self.logger.error(f"Error assessing risk level: {e}")
            return 'UNKNOWN'
    
    def _calculate_cumulative_return(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate cumulative return"""
        try:
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            return total_pnl / 10000  # Assuming $10k base
            
        except Exception as e:
            self.logger.error(f"Error calculating cumulative return: {e}")
            return 0.0
    
    def _identify_risk_factors(self) -> List[str]:
        """Identify current risk factors"""
        risk_factors = []
        
        try:
            current_snapshot = self.get_current_performance_snapshot()
            
            if current_snapshot.max_drawdown < -0.05:
                risk_factors.append("High drawdown")
            
            if current_snapshot.tail_risk_score > 0.6:
                risk_factors.append("Elevated tail risk")
            
            if current_snapshot.win_rate < 0.6:
                risk_factors.append("Low win rate")
            
            if current_snapshot.volatility_regime == 'HIGH_VOLATILITY':
                risk_factors.append("High volatility environment")
            
            if current_snapshot.sharpe_ratio < 0.5:
                risk_factors.append("Poor risk-adjusted returns")
                
        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {e}")
        
        return risk_factors
    
    def _calculate_risk_contribution(self) -> Dict[str, float]:
        """Calculate risk contribution by different factors"""
        try:
            if not self.trade_history:
                return {}
            
            total_volatility = np.std([trade.get('pnl', 0) for trade in self.trade_history])
            
            # Calculate contribution by symbol
            symbol_contributions = {}
            for symbol in self.symbol_performance:
                symbol_volatility = np.std([
                    trade.get('pnl', 0) for trade in self.trade_history 
                    if trade.get('symbol') == symbol
                ])
                symbol_contributions[symbol] = symbol_volatility / total_volatility if total_volatility > 0 else 0
            
            return symbol_contributions
            
        except Exception as e:
            self.logger.error(f"Error calculating risk contribution: {e}")
            return {}
    
    def _calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        try:
            current_snapshot = self.get_current_performance_snapshot()
            
            return {
                'total_trades': len(self.trade_history),
                'total_return': current_snapshot.cumulative_return,
                'max_drawdown': current_snapshot.max_drawdown,
                'win_rate': current_snapshot.win_rate,
                'sharpe_ratio': current_snapshot.sharpe_ratio,
                'sortino_ratio': current_snapshot.sortino_ratio,
                'profit_factor': current_snapshot.profit_factor,
                'tail_risk_score': current_snapshot.tail_risk_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating overall metrics: {e}")
            return {}
    
    def _check_drawdown_health(self, snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        """Check drawdown health"""
        if snapshot.max_drawdown < self.alert_thresholds['max_drawdown']:
            return {
                'healthy': False,
                'severity': 'CRITICAL',
                'metric': 'max_drawdown',
                'message': f"Maximum drawdown of {snapshot.max_drawdown:.1%} exceeds threshold"
            }
        return {'healthy': True}
    
    def _check_win_rate_health(self, snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        """Check win rate health"""
        if snapshot.win_rate < self.alert_thresholds['min_win_rate']:
            return {
                'healthy': False,
                'severity': 'WARNING',
                'metric': 'win_rate',
                'message': f"Win rate of {snapshot.win_rate:.1%} below threshold"
            }
        return {'healthy': True}
    
    def _check_sharpe_health(self, snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        """Check Sharpe ratio health"""
        if snapshot.sharpe_ratio < self.alert_thresholds['min_sharpe_ratio']:
            return {
                'healthy': False,
                'severity': 'WARNING',
                'metric': 'sharpe_ratio',
                'message': f"Sharpe ratio of {snapshot.sharpe_ratio:.2f} below threshold"
            }
        return {'healthy': True}
    
    def _check_tail_risk_health(self, snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        """Check tail risk health"""
        if snapshot.tail_risk_score > self.alert_thresholds['max_tail_risk_score']:
            return {
                'healthy': False,
                'severity': 'CRITICAL',
                'metric': 'tail_risk_score',
                'message': f"Tail risk score of {snapshot.tail_risk_score:.2f} exceeds threshold"
            }
        return {'healthy': True}
    
    def _check_volatility_health(self, snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        """Check volatility regime health"""
        if snapshot.volatility_regime == 'HIGH_VOLATILITY':
            return {
                'healthy': False,
                'severity': 'WARNING',
                'metric': 'volatility_regime',
                'message': "System operating in high volatility regime"
            }
        return {'healthy': True}
    
    def _generate_health_recommendations(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate health recommendations based on alerts"""
        recommendations = []
        
        try:
            alert_metrics = [alert.get('metric') for alert in alerts]
            
            if 'max_drawdown' in alert_metrics:
                recommendations.append("Consider reducing position sizes")
                recommendations.append("Review stop-loss settings")
            
            if 'tail_risk_score' in alert_metrics:
                recommendations.append("Increase confidence thresholds")
                recommendations.append("Implement additional risk controls")
            
            if 'win_rate' in alert_metrics:
                recommendations.append("Review entry criteria")
                recommendations.append("Consider market regime filtering")
            
            if 'volatility_regime' in alert_metrics:
                recommendations.append("Adjust trading frequency")
                recommendations.append("Consider switching to lower volatility periods")
                
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _get_default_snapshot(self) -> PerformanceSnapshot:
        """Get default performance snapshot"""
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            total_return=0.0,
            daily_return=0.0,
            cumulative_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            profit_factor=0.0,
            tail_risk_score=0.5,
            volatility_regime='UNKNOWN',
            risk_level='UNKNOWN'
        )
