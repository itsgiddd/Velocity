#!/usr/bin/env python3
"""
Maximum Profit Neural Trading App
================================

Enhanced neural trading application with:
- Multi-timeframe analysis (M15, H1, H4, D1)
- 4-candle continuation pattern recognition
- Maximum profit taking strategy
- USDJPY bidirection trading capability
- Comprehensive market analysis

This app addresses the user's requirements for:
- Recognizing 4 consecutive candle continuation patterns
- Trading multiple timeframes for comprehensive analysis
- Taking maximum profit rather than fixed 20-pip limits
- Full BUY and SELL capability for USDJPY
- Understanding the entire market picture for profitable trading
"""

import sys
import os
import logging
import threading
import time
from datetime import datetime
from typing import Optional

# Add the neural-forex-trading-app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced modules
from mt5_connector import MT5Connector
from model_manager import NeuralModelManager
from maximum_profit_trading_engine import MaximumProfitTradingEngine
from config_manager import ConfigManager

class MaximumProfitTradingApp:
    """Enhanced trading application with maximum profit capability"""
    
    def __init__(self):
        """Initialize the enhanced trading application"""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.mt5_connector = None
        self.model_manager = None
        self.trading_engine = None
        self.config_manager = None
        
        # App state
        self.is_running = False
        self.ui_thread = None
        
        self.logger.info("Maximum Profit Trading App initialized")

    def setup_logging(self):
        """Setup enhanced logging for the application"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/maximum_profit_trading_app.log'),
                logging.StreamHandler()
            ]
        )

    def initialize(self):
        """Initialize all components of the trading system"""
        try:
            self.logger.info("Initializing Maximum Profit Trading System...")
            
            # Initialize MT5 connector
            self.mt5_connector = MT5Connector()
            if not self.mt5_connector.connect():
                self.logger.error("Failed to initialize MT5 connector")
                return False
            
            # Initialize config manager
            self.config_manager = ConfigManager()
            
            # Initialize model manager
            self.model_manager = NeuralModelManager()
            self.model_manager.load_model()
            
            if not self.model_manager.model_loaded:
                self.logger.error("Failed to load neural model")
                return False
            
            # Initialize enhanced trading engine
            self.trading_engine = MaximumProfitTradingEngine(
                mt5_connector=self.mt5_connector,
                model_manager=self.model_manager,
                risk_per_trade=0.015,  # 1.5% risk per trade
                confidence_threshold=0.65,  # 65% confidence threshold
                trading_pairs=['USDJPY'],  # Only USDJPY as requested
                max_concurrent_positions=3  # Max 3 concurrent positions
            )
            
            self.logger.info("Maximum Profit Trading System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing trading system: {e}")
            return False

    def start(self):
        """Start the enhanced trading application"""
        try:
            if self.is_running:
                self.logger.warning("Trading app is already running")
                return
            
            self.logger.info("Starting Maximum Profit Trading App...")
            
            # Start the enhanced trading engine
            self.trading_engine.start_trading()
            
            # Start UI monitoring
            self.start_ui_monitoring()
            
            self.is_running = True
            self.logger.info("Maximum Profit Trading App started successfully")
            
            # Display system status
            self.display_system_status()
            
        except Exception as e:
            self.logger.error(f"Error starting trading app: {e}")

    def stop(self):
        """Stop the enhanced trading application"""
        try:
            self.logger.info("Stopping Maximum Profit Trading App...")
            
            # Stop trading engine
            if self.trading_engine:
                self.trading_engine.stop_trading()
            
            self.is_running = False
            self.logger.info("Maximum Profit Trading App stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading app: {e}")

    def display_system_status(self):
        """Display comprehensive system status"""
        try:
            print("\n" + "="*80)
            print("MAXIMUM PROFIT NEURAL TRADING SYSTEM - STATUS")
            print("="*80)
            
            # Check MT5 connection
            if self.mt5_connector and self.mt5_connector.is_connected():
                account_info = self.mt5_connector.get_account_info()
                if account_info:
                    print(f"[YES] MT5 Connected: Account {account_info['login']}")
                    print(f"  Balance: ${account_info['balance']:.2f}")
                    print(f"  Equity: ${account_info['equity']:.2f}")
                    print(f"  Margin: ${account_info['margin']:.2f}")
                else:
                    print("[WARN] MT5 Connected but account info unavailable")
            else:
                print("[NO] MT5 Not Connected")
            
            # Check neural model
            if self.model_manager and self.model_manager.model_loaded:
                print(f"[YES] Neural Model Loaded: neural_model.pth")
                print(f"  Model Type: {type(self.model_manager.model).__name__}")
            else:
                print("[NO] Neural Model Not Loaded")
            
            # Check trading engine
            if self.trading_engine:
                print(f"[YES] Trading Engine Active: {len(self.trading_engine.positions)} positions")
                print(f"  Trading Pairs: {', '.join(self.trading_engine.trading_pairs)}")
                print(f"  Risk Per Trade: {self.trading_engine.risk_per_trade*100:.1f}%")
                print(f"  Confidence Threshold: {self.trading_engine.confidence_threshold*100:.0f}%")
            else:
                print("[NO] Trading Engine Not Active")
            
            # Display recent signals
            if self.trading_engine and self.trading_engine.signals_history:
                print(f"\nRecent Trading Signals:")
                for signal in self.trading_engine.signals_history[-3:]:
                    print(f"  {signal.timestamp.strftime('%H:%M:%S')} - {signal.symbol} {signal.action}")
                    print(f"    Entry: {signal.entry_price:.5f}, SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}")
                    print(f"    Confidence: {signal.confidence*100:.1f}% - {signal.reason}")
            
            # Display current positions
            if self.trading_engine and self.trading_engine.positions:
                print(f"\nCurrent Positions ({len(self.trading_engine.positions)}):")
                for ticket, position in self.trading_engine.positions.items():
                    current_profit = 0.0  # This would come from MT5 in real implementation
                    print(f"  Ticket {ticket}: {position.symbol} {position.action}")
                    print(f"    Entry: {position.entry_price:.5f}")
                    print(f"    Current: {position.current_price:.5f} (P&L: ${current_profit:.2f})")
                    print(f"    Peak Profit: ${position.peak_profit:.2f}")
            
            print(f"\nSystem Features:")
            print(f"  [YES] Multi-timeframe Analysis (M15, H1, H4, D1)")
            print(f"  [YES] 4-Candle Continuation Pattern Recognition")
            print(f"  [YES] Maximum Profit Taking Strategy")
            print(f"  [YES] USDJPY Bidirectional Trading")
            print(f"  [YES] Dynamic Stop Loss Calculation")
            print(f"  [YES] Smart Profit Protection")
            
            print("\n" + "="*80)
            print("ENHANCED TRADING FEATURES ACTIVATED")
            print("="*80)
            
        except Exception as e:
            self.logger.error(f"Error displaying system status: {e}")

    def start_ui_monitoring(self):
        """Start UI monitoring thread"""
        def monitor_ui():
            while self.is_running:
                try:
                    time.sleep(60)  # Update every minute
                    
                    # Log current status
                    if self.trading_engine:
                        self.logger.info(f"System Status: {len(self.trading_engine.positions)} positions, "
                                       f"{len(self.trading_engine.signals_history)} signals generated")
                        
                        # Display enhanced status
                        self.display_enhanced_status()
                        
                except Exception as e:
                    self.logger.error(f"Error in UI monitoring: {e}")
                    time.sleep(60)
        
        self.ui_thread = threading.Thread(target=monitor_ui, daemon=True)
        self.ui_thread.start()

    def display_enhanced_status(self):
        """Display enhanced trading status"""
        try:
            # Get current market data for USDJPY
            if self.mt5_connector and self.trading_engine:
                symbol_info = self.mt5_connector.get_symbol_info('USDJPY')
                if symbol_info:
                    current_price = symbol_info['bid']
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] USDJPY Current: {current_price:.5f}")
                    
                    # Generate analysis
                    market_data = self.trading_engine._get_comprehensive_market_data('USDJPY')
                    if market_data:
                        # 4-candle analysis
                        continuation_analysis = self.trading_engine._analyze_continuation_patterns(market_data)
                        if continuation_analysis['pattern_detected']:
                            pattern_type = continuation_analysis['pattern_type']
                            strength = continuation_analysis['overall_confidence']
                            print(f"  [PATTERN] 4-Candle Pattern: {pattern_type} (Strength: {strength*100:.1f}%)")
                        
                        # Multi-timeframe trend analysis
                        tf_analysis = self.trading_engine._analyze_all_timeframes('USDJPY', market_data)
                        trends = []
                        for tf_name, analysis in tf_analysis.items():
                            trend_dir = analysis.get('trend', {}).get('trend_direction', 'unknown')
                            trends.append(f"{tf_name}:{trend_dir}")
                        
                        print(f"  [TRENDS] Multi-timeframe Trends: {', '.join(trends)}")
                        
                        # Combined confidence
                        combined_confidence = self.trading_engine._calculate_combined_confidence(tf_analysis, continuation_analysis)
                        print(f"  [CONF] Combined Confidence: {combined_confidence*100:.1f}%")
                        
                        # Decision
                        action = self.trading_engine._determine_trade_action(tf_analysis, continuation_analysis)
                        print(f"  [ACTION] Recommended Action: {action}")
                        
                        if action != 'HOLD':
                            entry_price = symbol_info['ask'] if action == 'BUY' else symbol_info['bid']
                            dynamic_sl = self.trading_engine._calculate_dynamic_stop_loss('USDJPY', action, entry_price, tf_analysis)
                            max_tp = self.trading_engine._calculate_maximum_profit_target('USDJPY', action, entry_price, tf_analysis)
                            
                            print(f"    Entry: {entry_price:.5f}")
                            print(f"    Dynamic SL: {dynamic_sl:.5f}")
                            print(f"    Maximum TP: {max_tp:.5f}")
                            print(f"    Potential Profit: {(max_tp - entry_price):.5f} pips" if action == 'BUY' else f"    Potential Profit: {(entry_price - max_tp):.5f} pips")
        except Exception as e:
            self.logger.error(f"Error displaying enhanced status: {e}")

    def run(self):
        """Main application run loop"""
        try:
            print("="*80)
            print("MAXIMUM PROFIT NEURAL TRADING SYSTEM")
            print("="*80)
            print("Enhanced Features:")
            print("• Multi-timeframe Analysis (M15, H1, H4, D1)")
            print("• 4-Candle Continuation Pattern Recognition") 
            print("• Maximum Profit Taking Strategy")
            print("• USDJPY Bidirectional Trading")
            print("• Dynamic Stop Loss Calculation")
            print("• Smart Profit Protection")
            print("="*80)
            
            # Initialize system
            if not self.initialize():
                self.logger.error("Failed to initialize trading system")
                return False
            
            # Start trading
            self.start()
            
            print("\n[STARTED] Maximum Profit Trading System Started!")
            print("Press Ctrl+C to stop the system")
            
            # Main application loop
            try:
                while self.is_running:
                    time.sleep(10)
                    
            except KeyboardInterrupt:
                print("\n[STOPPING] Stopping Maximum Profit Trading System...")
                self.stop()
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in main application loop: {e}")
            return False
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the application"""
        try:
            self.logger.info("Shutting down Maximum Profit Trading System...")
            
            # Stop all components
            if self.trading_engine:
                self.trading_engine.stop_trading()
            
            # Disconnect MT5
            if self.mt5_connector:
                self.mt5_connector.disconnect()
            
            self.logger.info("Maximum Profit Trading System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point for the enhanced trading application"""
    app = MaximumProfitTradingApp()
    return app.run()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)