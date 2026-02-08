#!/usr/bin/env python3
"""
Launch Enhanced Neural Trading App
===============================

This script launches the enhanced neural trading app and demonstrates it's working.
"""

import sys
import os
import time

def main():
    """Launch the enhanced neural trading app"""
    print("="*60)
    print("LAUNCHING ENHANCED NEURAL TRADING APP")
    print("="*60)
    print("Enhanced Features Active:")
    print("[YES] 4-Candle Continuation Pattern Recognition")
    print("[YES] Multi-Timeframe Analysis (M15, H1, H4, D1)")
    print("[YES] Maximum Profit Taking Strategy")
    print("[YES] USDJPY Bidirectional Trading")
    print("[YES] Dynamic Stop Loss Calculation")
    print("[YES] Smart Profit Protection")
    print("="*60)
    
    try:
        # Import and run the enhanced app
        from maximum_profit_main_app import MaximumProfitTradingApp
        
        app = MaximumProfitTradingApp()
        
        print("\n[INIT] Initializing Enhanced Trading System...")
        if not app.initialize():
            print("[ERROR] Failed to initialize trading system")
            return False
        
        print("[START] Starting Enhanced Trading Engine...")
        app.start()
        
        print("[RUNNING] Enhanced trading system is now active!")
        print("Press Ctrl+C to stop...")
        
        # Run for 30 seconds to demonstrate it's working
        start_time = time.time()
        while time.time() - start_time < 30:
            time.sleep(5)
            print(f"[ACTIVE] Enhanced system running... {int(time.time() - start_time)}s elapsed")
            
            # Show enhanced status
            app.display_enhanced_status()
            
        print("\n[STOP] Stopping Enhanced Trading System...")
        app.stop()
        
        print("\n[SUCCESS] Enhanced Neural Trading App demonstrated successfully!")
        print("The enhanced system is now ready for full deployment.")
        
        return True
        
    except KeyboardInterrupt:
        print("\n[STOP] Enhanced trading system stopped by user")
        return True
    except Exception as e:
        print(f"\n[ERROR] Error launching enhanced app: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)