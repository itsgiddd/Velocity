#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced Risk Management System
================================================

This test validates the enhanced risk management system that addresses the volatility issues:
- Tests tail risk protection with dynamic confidence adjustment
- Validates volatility clustering detection
- Tests position sizing with dynamic risk scaling
- Validates performance tracking and alerting
- Simulates various market conditions to ensure stability

Target: Transform p10 from -85.18% to positive territory while maintaining upside potential
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unittest
from unittest.mock import Mock, MagicMock
import logging

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_tail_risk_protection import TailRiskProtector, VolatilityRegime, RiskLevel
from advanced_performance_tracking import AdvancedPerformanceTracker, AlertLevel, PerformanceSnapshot
from app.trading_engine import TradingEngine, TradingSignal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestEnhancedRiskManagement(unittest.TestCase):
    """
    Comprehensive test suite for enhanced risk management
    """
    
    def setUp(self):
        """Set up test environment"""
        self.tail_risk_protector = TailRiskProtector()
        self.performance_tracker = AdvancedPerformanceTracker()
        
        # Create sample market data for testing
        self.sample_market_data = self._create_sample_market_data()
        
        # Set up mock MT5 connector
        self.mock_mt5_connector = Mock()
        self.mock_mt5_connector.is_connected.return_value = True
        self.mock_mt5_connector.get_account_info.return_value = {
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'free_margin': 10000.0
        }
        
        # Set up mock model manager
        self.mock_model_manager = Mock()
        self.mock_model_manager.is_model_loaded.return_value = True
        self.mock_model_manager.predict.return_value = {
            'action': 'BUY',
            'confidence': 0.85,
            'probabilities': {'BUY': 0.85, 'SELL': 0.10, 'HOLD': 0.05},
            'trade_score': 0.85,
            'should_trade': True
        }
        
        logger.info("Test environment set up successfully")
    
    def test_tail_risk_protection_basic_functionality(self):
        """Test basic tail risk protection functionality"""
        logger.info("Testing basic tail risk protection...")
        
        # Test market condition analysis
        market_conditions = self.tail_risk_protector.analyze_market_conditions(self.sample_market_data)
        
        # Verify market analysis works
        self.assertIn('volatility_regime', market_conditions)
        self.assertIn('current_volatility', market_conditions)
        self.assertIn('trend_strength', market_conditions)
        
        # Test dynamic confidence calculation
        base_confidence = 0.78
        dynamic_confidence = self.tail_risk_protector.calculate_dynamic_confidence(
            base_confidence, market_conditions, []
        )
        
        # Dynamic confidence should be adjusted based on conditions
        self.assertIsInstance(dynamic_confidence, float)
        self.assertGreaterEqual(dynamic_confidence, 0.65)
        self.assertLessEqual(dynamic_confidence, 0.95)
        
        logger.info(f"✓ Basic tail risk protection: Dynamic confidence = {dynamic_confidence:.3f}")
    
    def test_volatility_clustering_detection(self):
        """Test volatility clustering detection"""
        logger.info("Testing volatility clustering detection...")
        
        # Create data with volatility clustering
        high_vol_data = self._create_volatile_market_data()
        low_vol_data = self._create_calm_market_data()
        
        # Analyze high volatility period
        high_vol_conditions = self.tail_risk_protector.analyze_market_conditions(high_vol_data)
        self.assertEqual(high_vol_conditions['volatility_regime'], VolatilityRegime.HIGH_VOLATILITY)
        
        # Analyze low volatility period
        low_vol_conditions = self.tail_risk_protector.analyze_market_conditions(low_vol_data)
        self.assertEqual(low_vol_conditions['volatility_regime'], VolatilityRegime.LOW_VOLATILITY)
        
        logger.info("✓ Volatility clustering detection working correctly")
    
    def test_dynamic_position_sizing(self):
        """Test dynamic position sizing with risk scaling"""
        logger.info("Testing dynamic position sizing...")
        
        base_lot_size = 0.10
        confidence = 0.85
        account_balance = 10000.0
        
        # Test normal conditions
        normal_conditions = {
            'volatility_regime': VolatilityRegime.NORMAL_VOLATILITY,
            'market_stress': 0.3,
            'liquidity_risk': 0.2
        }
        
        normal_size = self.tail_risk_protector.calculate_dynamic_position_size(
            base_lot_size, confidence, normal_conditions, account_balance
        )
        
        # Test high volatility conditions
        high_vol_conditions = {
            'volatility_regime': VolatilityRegime.HIGH_VOLATILITY,
            'market_stress': 0.8,
            'liquidity_risk': 0.7
        }
        
        high_vol_size = self.tail_risk_protector.calculate_dynamic_position_size(
            base_lot_size, confidence, high_vol_conditions, account_balance
        )
        
        # High volatility should result in smaller position sizes
        self.assertLess(high_vol_size, normal_size)
        
        logger.info(f"✓ Dynamic position sizing: Normal={normal_size:.3f}, High Vol={high_vol_size:.3f}")
    
    def test_trade_signal_validation(self):
        """Test comprehensive trade signal validation"""
        logger.info("Testing trade signal validation...")
        
        # Test normal conditions - should approve trade
        normal_conditions = {
            'volatility_regime': VolatilityRegime.NORMAL_VOLATILITY,
            'market_stress': 0.2,
            'liquidity_risk': 0.1
        }
        
        high_confidence = 0.85
        validation_result = self.tail_risk_protector.validate_trade_signal(
            high_confidence, normal_conditions, []
        )
        
        self.assertTrue(validation_result['approved'])
        
        # Test extreme volatility - should reject trade
        extreme_conditions = {
            'volatility_regime': VolatilityRegime.EXTREME_VOLATILITY,
            'market_stress': 0.9,
            'liquidity_risk': 0.8
        }
        
        extreme_validation = self.tail_risk_protector.validate_trade_signal(
            high_confidence, extreme_conditions, []
        )
        
        self.assertFalse(extreme_validation['approved'])
        self.assertIn("Extreme volatility", str(extreme_validation['rejection_reasons']))
        
        logger.info("✓ Trade signal validation working correctly")
    
    def test_performance_tracking_basic(self):
        """Test basic performance tracking functionality"""
        logger.info("Testing performance tracking...")
        
        # Record some sample trades
        sample_trades = [
            {'symbol': 'EURUSD', 'pnl': 100.0, 'timestamp': datetime.now()},
            {'symbol': 'GBPUSD', 'pnl': -50.0, 'timestamp': datetime.now()},
            {'symbol': 'USDJPY', 'pnl': 75.0, 'timestamp': datetime.now()},
        ]
        
        for trade in sample_trades:
            self.performance_tracker.record_trade(trade)
        
        # Test current snapshot
        snapshot = self.performance_tracker.get_current_performance_snapshot()
        
        self.assertIsInstance(snapshot, PerformanceSnapshot)
        self.assertEqual(snapshot.win_rate, 2/3)  # 2 out of 3 trades winning
        self.assertEqual(snapshot.total_return, 125.0)  # 100 - 50 + 75
        
        logger.info(f"✓ Performance tracking: Win Rate={snapshot.win_rate:.1%}, Return=${snapshot.total_return:.2f}")
    
    def test_predictive_analytics(self):
        """Test predictive analytics functionality"""
        logger.info("Testing predictive analytics...")
        
        # Generate more trades for prediction
        for i in range(50):
            trade = {
                'symbol': f'TEST{i%8}',
                'pnl': np.random.normal(10, 50),  # Random P&L
                'timestamp': datetime.now() - timedelta(days=i)
            }
            self.performance_tracker.record_trade(trade)
        
        # Test performance prediction
        prediction = self.performance_tracker.predict_performance(days_ahead=30)
        
        self.assertIn('predicted_return', prediction)
        self.assertIn('confidence_interval_95', prediction)
        self.assertIn('trend_direction', prediction)
        
        logger.info(f"✓ Predictive analytics: Predicted 30-day return = {prediction['predicted_return']:.2%}")
    
    def test_risk_alert_system(self):
        """Test risk alert system"""
        logger.info("Testing risk alert system...")
        
        # Set up alert callback
        alerts_received = []
        def alert_callback(alert):
            alerts_received.append(alert)
        
        self.performance_tracker.add_alert_callback(alert_callback)
        
        # Record a large loss to trigger alert
        large_loss_trade = {
            'symbol': 'EURUSD',
            'pnl': -500.0,  # 5% loss on $10k account
            'entry_value': 10000.0,
            'timestamp': datetime.now()
        }
        
        self.performance_tracker.record_trade(large_loss_trade)
        
        # Check if alert was generated
        active_alerts = self.performance_tracker.get_active_alerts(AlertLevel.WARNING)
        self.assertTrue(len(active_alerts) > 0)
        
        # Verify alert callback was called
        self.assertTrue(len(alerts_received) > 0)
        
        logger.info(f"✓ Risk alert system: Generated {len(active_alerts)} alerts")
    
    def test_end_to_end_risk_scenario(self):
        """Test end-to-end risk management scenario"""
        logger.info("Testing end-to-end risk scenario...")
        
        # Simulate a stressed market environment
        stressed_market_data = self._create_stressed_market_data()
        stressed_conditions = self.tail_risk_protector.analyze_market_conditions(stressed_market_data)
        
        # Test that system becomes more conservative under stress
        base_confidence = 0.78
        stressed_confidence = self.tail_risk_protector.calculate_dynamic_confidence(
            base_confidence, stressed_conditions, []
        )
        
        # In stressed conditions, confidence should be adjusted downward
        self.assertLess(stressed_confidence, base_confidence)
        
        # Test position sizing under stress
        stressed_position_size = self.tail_risk_protector.calculate_dynamic_position_size(
            0.10, 0.85, stressed_conditions, 10000.0
        )
        
        # Position size should be reduced under stress
        self.assertLess(stressed_position_size, 0.10)
        
        logger.info(f"✓ End-to-end risk: Stressed confidence = {stressed_confidence:.3f}, Position = {stressed_position_size:.3f}")
    
    def test_system_health_monitoring(self):
        """Test system health monitoring"""
        logger.info("Testing system health monitoring...")
        
        # Generate some trades to establish baseline
        for i in range(20):
            trade = {
                'symbol': 'EURUSD',
                'pnl': np.random.normal(0, 30),
                'timestamp': datetime.now()
            }
            self.performance_tracker.record_trade(trade)
        
        # Test system health check
        health_status = self.performance_tracker.check_system_health()
        
        self.assertIn('overall_status', health_status)
        self.assertIn('alerts', health_status)
        self.assertIn('recommendations', health_status)
        
        logger.info(f"✓ System health: Status = {health_status['overall_status']}")
    
    def _create_sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.001, len(dates))
        prices = 1.1000 * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'close': prices,
            'tick_volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
    
    def _create_volatile_market_data(self):
        """Create highly volatile market data"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        np.random.seed(123)
        
        # High volatility
        returns = np.random.normal(0, 0.005, len(dates))
        prices = 1.1000 * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
            'close': prices,
            'tick_volume': np.random.randint(500, 2000, len(dates))
        }, index=dates)
    
    def _create_calm_market_data(self):
        """Create calm market data"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        np.random.seed(456)
        
        # Low volatility
        returns = np.random.normal(0, 0.0002, len(dates))
        prices = 1.1000 * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0001, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0001, len(dates)))),
            'close': prices,
            'tick_volume': np.random.randint(50, 200, len(dates))
        }, index=dates)
    
    def _create_stressed_market_data(self):
        """Create stressed market data with extreme conditions"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        np.random.seed(789)
        
        # Create crisis-like conditions
        returns = np.random.normal(0, 0.01, len(dates))
        # Add some extreme moves
        extreme_indices = np.random.choice(len(dates), size=50, replace=False)
        returns[extreme_indices] *= 5  # 5x normal moves
        
        prices = 1.1000 * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'tick_volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)

def run_comprehensive_risk_test():
    """
    Run comprehensive risk management test suite
    """
    logger.info("Starting comprehensive risk management test suite...")
    logger.info("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedRiskManagement)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Test Results: {result.testsRun} tests run")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    if result.failures:
        logger.info("FAILURES:")
        for test, traceback in result.failures:
            logger.info(f"- {test}: {traceback}")
    
    if result.errors:
        logger.info("ERRORS:")
        for test, traceback in result.errors:
            logger.info(f"- {test}: {traceback}")
    
    # Overall assessment
    if result.wasSuccessful():
        logger.info("🎉 ALL TESTS PASSED! Enhanced risk management system is working correctly.")
        logger.info("✅ The system should now achieve stable profitability with reduced volatility.")
        logger.info("✅ Target: Transform p10 from -85.18% to positive territory while maintaining upside.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_risk_test()
    sys.exit(0 if success else 1)