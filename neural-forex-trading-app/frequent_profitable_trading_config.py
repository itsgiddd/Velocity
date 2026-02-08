#!/usr/bin/env python3
"""
Frequent Profitable Trading Configuration
======================================

Balance profitability with trading frequency based on extensive historical learning.

SEQUENTIAL THINKING: Historical Learning → Pattern Recognition → Frequent Profitable Trading
"""

# ENHANCED NEURAL MODEL SETTINGS
NEURAL_MODEL_CONFIG = {
    "model_path": "enhanced_neural_model.pth",
    "confidence_threshold": 0.55,  # Lowered from 0.65 for more frequent trading
    "required_accuracy": 60.0,     # Minimum accuracy for profitable trading
}

# FREQUENT PROFITABLE TRADING SETTINGS
# Balance between profitability and trading frequency
FREQUENT_TRADING_CONFIG = {
    # Profit requirements (more flexible than extreme profitability)
    "MIN_PROFIT_R": 1.2,        # Lowered from 2.0 for more frequent exits
    "TIER1_CLOSE_PCT": 0.40,    # Increased from 0.25 for more profit taking
    "TIER2_CLOSE_PCT": 0.30,    # Increased from 0.25 
    "TIER3_CLOSE_PCT": 0.30,    # Changed from 0.50 for more balanced exits
    "TRAILING_PROFIT_R": 2.0,   # Lowered from 3.0 for more flexibility
    
    # Trading frequency settings (less restrictive timers)
    "MIN_HOLD_TIME": 1.0,       # Lowered from 4.0 hours for more frequent trading
    "PROFIT_LOCK_TIME": 0.5,    # Lowered from 2.0 hours
    "COOLDOWN_AFTER_LOSS": 2.0,  # Lowered from 12.0 hours
    "MIN_TIME_BETWEEN_WINS": 0.25,  # Lowered from 1.0 hours
    
    # Enhanced risk management
    "MAX_CONCURRENT_POSITIONS": 8,  # Increased from 5 for more activity
    "RISK_PER_TRADE": 0.020,         # Increased from 0.015 (2% vs 1.5%)
    "MAX_DAILY_RISK": 0.08,         # 8% max daily risk
    "MAX_WEEKLY_RISK": 0.20,        # 20% max weekly risk
}

# HISTORICAL LEARNING ENHANCEMENT
HISTORICAL_LEARNING_CONFIG = {
    # Data requirements for better pattern recognition
    "MIN_HISTORICAL_DATA_YEARS": 3,      # Use 3+ years of data
    "MIN_DATA_POINTS_PER_PAIR": 5000,    # Minimum data points per currency pair
    "TIMEFRAMES_TO_ANALYZE": ["M5", "M15", "M30", "H1", "H4"],  # Multiple timeframes
    "TECHNICAL_INDICATORS": [
        "price_momentum", "price_zscore", "sma_ratios", 
        "rsi", "volatility", "trend_strength", "bollinger_bands"
    ],
    
    # Model training parameters
    "EPOCHS": 200,                    # More training for better learning
    "BATCH_SIZE": 64,                 # Optimal batch size
    "LEARNING_RATE": 0.001,           # Stable learning rate
    "VALIDATION_SPLIT": 0.2,          # 20% for validation
    
    # Pattern recognition enhancement
    "LOOKAHEAD_PERIODS": 5,            # Predict 5 periods ahead
    "PROFIT_THRESHOLD": 0.005,        # 0.5% minimum profit for signal
    "LOSS_THRESHOLD": -0.005,         # -0.5% maximum loss threshold
}

# TRADING FREQUENCY TARGETS
FREQUENCY_TARGETS = {
    # Realistic trading frequency goals
    "TRADES_PER_DAY_TARGET": 8,       # Target 8 trades per day
    "TRADES_PER_WEEK_TARGET": 50,     # Target 50 trades per week
    "MIN_TRADES_PER_DAY": 3,          # Minimum 3 trades per day
    "MAX_TRADES_PER_DAY": 15,         # Maximum 15 trades per day
    
    # Profitability targets (adjusted for frequency)
    "TARGET_WIN_RATE": 65,            # 65% win rate (realistic for frequent trading)
    "TARGET_PROFIT_FACTOR": 2.0,      # 2.0 profit factor (realistic target)
    "TARGET_AVAILABLE_TRADES": 80,     # 80% of available trades taken
    "AVOID_OVERTRADING": True,        # Prevent excessive trading
}

# SEQUENTIAL LOGIC IMPROVEMENTS
SEQUENTIAL_IMPROVEMENTS = {
    # Smarter entry timing
    "ENTRY_TIME_WINDOWS": {
        "LONDON_OPEN": {"start": "08:00", "end": "12:00", "multiplier": 1.2},
        "NEW_YORK_OPEN": {"start": "13:00", "end": "17:00", "multiplier": 1.3},
        "OVERLAP": {"start": "13:00", "end": "16:00", "multiplier": 1.5},
        "ASIAN_SESSION": {"start": "00:00", "end": "08:00", "multiplier": 0.8},
    },
    
    # Dynamic confidence adjustment
    "CONFIDENCE_ADJUSTMENT": {
        "HIGH_VOLATILITY": 0.60,      # Higher threshold during high volatility
        "LOW_VOLATILITY": 0.50,       # Lower threshold during low volatility
        "TREND_STRONG": 0.55,         # Standard threshold for strong trends
        "TREND_WEAK": 0.65,           # Higher threshold for weak trends
    },
    
    # Market condition awareness
    "MARKET_CONDITIONS": {
        "TRENDING_MARKET": {
            "min_trend_strength": 0.02,
            "preferred_pairs": ["EURUSD", "GBPUSD", "USDJPY"],
            "time_multiplier": 1.0
        },
        "RANGING_MARKET": {
            "max_range_size": 0.015,
            "preferred_pairs": ["USDCAD"],  # Focus on USDCAD for ranging
            "time_multiplier": 0.8
        },
        "HIGH_VOLATILITY": {
            "min_volatility": 0.02,
            "risk_reduction": 0.5,
            "time_multiplier": 0.7
        }
    }
}

# PERFORMANCE MONITORING
PERFORMANCE_MONITORING = {
    # Real-time performance tracking
    "DAILY_TARGETS": {
        "min_trades": 3,
        "max_trades": 15,
        "target_win_rate": 65,
        "max_drawdown": 0.05,  # 5% max daily drawdown
        "target_profit": 100,   # $100 target daily profit
    },
    
    # Weekly performance targets
    "WEEKLY_TARGETS": {
        "min_trades": 20,
        "max_trades": 80,
        "target_win_rate": 65,
        "max_drawdown": 0.10,  # 10% max weekly drawdown
        "target_profit": 500,  # $500 target weekly profit
    },
    
    # Alert conditions
    "ALERTS": {
        "low_win_rate": 55,           # Alert if win rate drops below 55%
        "high_drawdown": 0.08,       # Alert if drawdown exceeds 8%
        "overtrading": 20,            # Alert if more than 20 trades in a day
        "losing_streak": 5,           # Alert after 5 consecutive losses
    }
}

# IMPLEMENTATION SUMMARY
IMPLEMENTATION_SUMMARY = {
    "approach": "Historical Learning → Frequent Profitable Trading",
    "key_changes": {
        "neural_model": "Enhanced with 3+ years historical data",
        "trading_frequency": "Increased from 0.1 to 8+ trades per day",
        "profit_requirements": "More flexible (1.2R vs 2.0R minimum)",
        "timer_settings": "Less restrictive for higher activity",
        "risk_management": "Balanced for frequent trading",
        "pattern_recognition": "Multiple timeframes and indicators"
    },
    
    "expected_improvements": {
        "trading_frequency": "80x increase (0.1 → 8 trades/day)",
        "pattern_learning": "Extensive historical data training",
        "profitability": "Maintained with better risk management",
        "market_adaptation": "Dynamic confidence and timing",
        "consistency": "Regular profitable trading activity"
    }
}

print("FREQUENT PROFITABLE TRADING CONFIGURATION LOADED")
print("Historical Learning -> Pattern Recognition -> Frequent Profitable Trading")
print("=" * 70)
print(f"Target Trading Frequency: {FREQUENCY_TARGETS['TRADES_PER_DAY_TARGET']} trades/day")
print(f"Minimum Profit Requirement: {FREQUENT_TRADING_CONFIG['MIN_PROFIT_R']}R")
print(f"Enhanced Neural Model: {NEURAL_MODEL_CONFIG['model_path']}")
print(f"Historical Data Period: {HISTORICAL_LEARNING_CONFIG['MIN_HISTORICAL_DATA_YEARS']} years")
print("Configuration optimized for frequent profitable trading")
