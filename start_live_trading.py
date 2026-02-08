#!/usr/bin/env python3
"""
START LIVE NEURAL TRADING
=========================

Simple script to start the automated neural trading bot.
This script will:
1. Load configuration settings
2. Initialize the neural trading system
3. Start automated trading on MT5
4. Monitor and manage trades

SAFETY FIRST: Always test in demo mode before live trading!
"""

import sys
import os
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from trading_config import *

# Import the live trading bot
try:
    from clean_live_trading_bot import LiveNeuralTradingBot, TradingMode
    print("Live Neural Trading Bot imported successfully")
except ImportError as e:
    print("Error importing trading bot: {}".format(e))
    print("Make sure all neural trading files are in the same directory")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOGGING['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGGING['file']),
            logging.StreamHandler() if LOGGING['console'] else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_config():
    """Validate configuration settings"""
    logger = logging.getLogger(__name__)
    
    errors = []
    warnings = []
    
    # Check trading mode
    if TRADING_MODE not in ['demo', 'live', 'backtest']:
        errors.append("Invalid trading mode. Must be 'demo', 'live', or 'backtest'")
    
    # Check confidence threshold
    if not (0.5 <= CONFIDENCE_THRESHOLD <= 1.0):
        errors.append("Confidence threshold must be between 0.5 and 1.0")
    
    # Check risk settings
    if not (0.001 <= MAX_RISK_PER_TRADE <= 0.1):
        errors.append("Max risk per trade must be between 0.1% and 10%")
    
    # Check symbols
    if not SYMBOLS or len(SYMBOLS) == 0:
        errors.append("At least one symbol must be configured")
    
    # Safety check for live trading
    if TRADING_MODE == 'live':
        warnings.append("LIVE TRADING MODE DETECTED")
        warnings.append("This will execute REAL trades with REAL money")
        warnings.append("Press Ctrl+C now to cancel if this is not intentional")
    
    if errors:
        logger.error("Configuration errors found:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    if warnings:
        logger.warning("Configuration warnings:")
        for warning_msg in warnings:
            logger.warning(f"  - {warning_msg}")
    
    return True

def print_startup_banner():
    """Print startup banner"""
    print("=" * 60)
    print("LIVE NEURAL TRADING BOT - STARTING")
    print("=" * 60)
    print(f"Mode: {TRADING_MODE.upper()}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.1%}")
    print(f"Max Risk Per Trade: {MAX_RISK_PER_TRADE:.1%}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Max Positions: {MAX_OPEN_POSITIONS}")
    print(f"Target Win Rate: {TARGETS['win_rate']:.1%}")
    print("=" * 60)

def safety_check():
    """Perform safety checks before starting"""
    print("Performing Safety Checks...")
    
    # Demo mode warning
    if TRADING_MODE == 'demo':
        print("Demo mode - No real money at risk")
    elif TRADING_MODE == 'live':
        print("LIVE MODE - REAL MONEY TRADING!")
        print("Are you sure? Press Ctrl+C to cancel...")
        import time
        time.sleep(3)  # Give user time to cancel
    
    # Configuration validation
    if not validate_config():
        print("Configuration validation failed")
        return False
    
    print("Safety checks passed")
    return True

def initialize_bot():
    """Initialize the trading bot"""
    print("Initializing Neural Trading Bot...")
    
    try:
        # Determine trading mode
        mode = TradingMode.DEMO if TRADING_MODE == 'demo' else TradingMode.LIVE
        
        # Create bot instance
        bot = LiveNeuralTradingBot(
            trading_mode=mode,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            max_risk_per_trade=MAX_RISK_PER_TRADE,
            symbols=SYMBOLS
        )
        
        print("Neural Trading Bot initialized successfully")
        return bot
        
    except Exception as e:
        print("Error initializing trading bot: {}".format(e))
        return None

def start_trading(bot):
    """Start the trading process"""
    print("Starting Automated Trading...")
    
    try:
        # Start trading
        success = bot.start_trading()
        
        if success:
            print("Trading bot started successfully!")
            print("Monitoring trades and performance...")
            print("Press Ctrl+C to stop trading")
            
            # Keep the script running
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping trading...")
                bot.stop_trading()
        else:
            print("Failed to start trading bot")
            
    except Exception as e:
        print("Error during trading: {}".format(e))
    finally:
        print("Trading session ended")

def main():
    """Main function"""
    # Setup logging
    logger = setup_logging()
    
    try:
        # Print startup banner
        print_startup_banner()
        
        # Safety check
        if not safety_check():
            print("Safety check failed. Aborting startup.")
            return 1
        
        # Initialize bot
        bot = initialize_bot()
        if bot is None:
            print("Bot initialization failed")
            return 1
        
        # Start trading
        start_trading(bot)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
        return 0
    except Exception as e:
        print("Unexpected error: {}".format(e))
        logger.exception("Unexpected error occurred")
        return 1

if __name__ == "__main__":
    print("Neural Trading System - Live Trading Bot")
    print("Target: 78%+ Win Rate")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    exit_code = main()
    sys.exit(exit_code)
