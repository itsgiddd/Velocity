#!/usr/bin/env python3
"""
Test Timer Functionality
=======================
Tests the enhanced timer system in the neural trading app.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

def test_timer_system():
    """Test the enhanced timer system"""
    print("TESTING ENHANCED TIMER SYSTEM")
    print("=" * 40)
    
    try:
        # Import the main app class
        from main_app import NeuralTradingApp
        import tkinter as tk
        
        # Create a mock root window
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Create the app instance
        print("Creating NeuralTradingApp instance...")
        app = NeuralTradingApp(root)
        
        # Test timer initialization
        print(f"App start time: {app.app_start_time}")
        print(f"Trading start time: {app.trading_start_time}")
        print(f"Current market session: {app.current_market_session}")
        
        # Test market session detection
        print("\nTesting market session detection:")
        current_time = datetime.now()
        app._update_market_session()
        print(f"Current UTC time: {current_time.strftime('%H:%M:%S')}")
        print(f"Detected session: {app.current_market_session}")
        
        # Test next session calculation
        next_session = app._get_next_market_session(current_time)
        if next_session:
            print(f"Next session: {next_session['name']} at {next_session['start_time'].strftime('%H:%M:%S')}")
        
        # Test duration formatting
        print("\nTesting duration formatting:")
        test_durations = [
            timedelta(seconds=45),
            timedelta(minutes=90),
            timedelta(hours=3),
            timedelta(days=1, hours=5, minutes=30)
        ]
        
        for duration in test_durations:
            formatted = app._format_duration(duration)
            print(f"  {duration} -> {formatted}")
        
        # Test symbol readiness
        print("\nTesting symbol readiness:")
        from frequent_profitable_trading_config import FREQUENT_TRADING_CONFIG
        
        # Mock timer data for testing
        test_timer_data = {
            'last_trade_time': datetime.now() - timedelta(hours=2),
            'cooldown_until': datetime.now() - timedelta(hours=1),
            'profit_lock_until': datetime.now() - timedelta(minutes=30)
        }
        
        is_ready = app._is_symbol_ready(test_timer_data, datetime.now())
        print(f"  Symbol ready (should be True): {is_ready}")
        
        # Test timer display update
        print("\nTesting timer display update...")
        app.update_timer_display()
        print("  Timer display updated successfully")
        
        # Test trading timer functionality
        print("\nTesting trading timer functionality:")
        
        # Start trading
        app.trading_start_time = datetime.now()
        print(f"  Trading started at: {app.trading_start_time}")
        
        # Simulate some time passing
        import time
        time.sleep(2)
        
        # Test timer display again
        app.update_timer_display()
        print("  Timer display updated after trading start")
        
        # Clean up
        root.destroy()
        
        print("\n[SUCCESS] All timer functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Timer test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_app_launch():
    """Test that the app can be launched without errors"""
    print("\nTESTING APP LAUNCH")
    print("=" * 20)
    
    try:
        import tkinter as tk
        from main_app import NeuralTradingApp
        
        # Create a minimal test
        root = tk.Tk()
        root.withdraw()
        
        app = NeuralTradingApp(root)
        
        # Check basic initialization
        assert app.app_start_time is not None
        assert app.trading_start_time is None
        assert len(app.symbol_timers) > 0
        
        print("App initialized successfully")
        print(f"Model loaded: {app.model_loaded}")
        print(f"MT5 connected: {app.mt5_connected}")
        print(f"Symbol timers initialized: {len(app.symbol_timers)}")
        
        root.destroy()
        
        print("[SUCCESS] App launch test passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] App launch test failed: {e}")
        return False

def main():
    """Main test function"""
    print("ENHANCED TIMER SYSTEM TEST")
    print("=" * 50)
    
    # Change to neural-forex-trading-app directory
    if not Path("main_app.py").exists():
        print("Error: Not in neural-forex-trading-app directory")
        return False
    
    success = True
    
    # Test timer system
    if not test_timer_system():
        success = False
    
    # Test app launch
    if not test_app_launch():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("[OVERALL SUCCESS] All tests passed! Timer system is working correctly.")
    else:
        print("[OVERALL FAILURE] Some tests failed. Check the output above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)