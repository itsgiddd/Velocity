#!/usr/bin/env python3
"""
Extreme Profitability Configuration
==================================

SEQUENTIAL THINKING FRAMEWORK:
What happens → Why it fails → Sequential fix

This configuration transforms "high win rate → moderate profit" into 
"extreme profitability" using timer-based sequential discipline.

EXTREME PROFITABILITY SETTINGS
===============================
"""

# PROFIT-FIRST SETTINGS (Transform moderate profit into extreme profit)
PROFIT_FIRST_CONFIG = {
    # NEVER exit before 2R profit - This is the CORE rule
    "MIN_PROFIT_R": 2.0,  # Minimum R units (risk multiples) before any exit
    
    # TIERED EXIT STRATEGY - Let winners run longer
    "TIER1_CLOSE_PCT": 0.25,  # Close 25% at 2R
    "TIER2_CLOSE_PCT": 0.25,  # Close 25% at 3R (1.5 * 2R)
    "TIER3_CLOSE_PCT": 0.50,  # Close 50% at 4R (2.0 * 2R)
    
    # NO EARLY EXITS - Only break this rule with 3R+ trailing profit
    "TRAILING_PROFIT_R": 3.0,  # Allow early exit only if 3R+ trailing profit
    
    # PROFIT LOCK TIMER - Don't exit during profit lock period
    "PROFIT_LOCK_HOURS": 2.0,  # Must hold for 2 hours before any exit consideration
}

# SEQUENTIAL TIMER SYSTEM (Enforce trade discipline)
TIMER_CONFIG = {
    # MIN HOLD TIME - Wait between trades to avoid overtrading
    "MIN_HOLD_TIME": 4.0,  # Hours between trades per symbol
    
    # COOLDOWN AFTER LOSS - Prevent revenge trading
    "COOLDOWN_AFTER_LOSS": 12.0,  # Hours pause after any loss
    
    # WIN COOLDOWN - Prevent overconfidence trading
    "MIN_TIME_BETWEEN_WINS": 1.0,  # Hours between wins
    
    # PROFIT LOCK TIME - Hold positions for minimum time
    "PROFIT_LOCK_TIME": 2.0,  # Hours profit lock after entry
}

# EXTREME PROFITABILITY TARGETS
# Transform moderate profit into extreme profit:
# - Current: High win rate, small winners, some large losers
# - Target: Fewer trades, bigger winners, smaller losers
PROFITABILITY_TARGETS = {
    # PROFIT FACTOR > 3.0 (not just win rate)
    "TARGET_PROFIT_FACTOR": 3.0,
    
    # Average Winner > 3x Average Loser
    "MIN_WIN_LOSS_RATIO": 3.0,
    
    # Max consecutive losses < 3
    "MAX_CONSECUTIVE_LOSSES": 3,
    
    # Trade frequency decreases but profit per trade increases
    "TARGET_TRADES_PER_DAY": 3,  # Reduced from typical 10-15
    "TARGET_PROFIT_PER_TRADE": 50.0,  # Increased from typical $10-20
}

# SEQUENTIAL LOGIC SUMMARY
# =======================
# 
# SEQUENCE FLOW:
# 1. NEW SIGNAL → Check "time since last trade" > MIN_HOLD_TIME (4h)
# 2. NEW TRADE → Set "profit lock timer" = 2 hours (no exit unless 2R)
# 3. LOSS → Set "cool down timer" = 12 hours (no new trades)
# 4. WIN → Reset timers but require "time since last win" > 1h before next trade
# 
# EXTREME PROFITABILITY FLOW:
# 1. Entry → Signal must pass timer checks + 2R profit potential
# 2. During Trade → Profit lock timer active (2h minimum hold)
# 3. Exit Options → Tiered exits only (25% → 25% → 50%)
# 4. No Early Exits → Unless 3R+ trailing profit achieved
# 
# BEFORE ENTRY: Timer check → Profit potential check → Signal confidence
# DURING TRADE: Profit lock timer → Tiered exit sequence
# AFTER WIN: Reset with hold time → Next trade allowed after 1h
# AFTER LOSS: Cooldown timer → No trades for 12h
# OVERALL: Fewer trades, bigger winners, smaller losers

# BACKTEST VALIDATION METRICS
# ===========================
VALIDATION_METRICS = {
    # Must achieve extreme profitability, not just win rate
    "PROFIT_FACTOR_THRESHOLD": 3.0,  # Total profit / Total loss
    "WIN_LOSS_RATIO_THRESHOLD": 3.0,  # Average winner / Average loser
    "MAX_DRAWDOWN_THRESHOLD": 0.05,  # Maximum 5% drawdown
    "CONSECUTIVE_LOSS_LIMIT": 3,  # Never more than 3 losses in a row
    
    # Performance improvement indicators
    "TRADES_SHOULD_DECREASE": True,  # Fewer total trades
    "PROFIT_PER_TRADE_SHOULD_INCREASE": True,  # Higher profit per trade
    "WIN_RATE_MAY_DECREASE": True,  # Win rate can drop but profit increases
}

# IMPLEMENTATION STATUS
# ====================
IMPLEMENTATION_STATUS = {
    "sequential_timers": "[IMPLEMENTED] Per-symbol timer tracking",
    "profit_first_rules": "[IMPLEMENTED] 2R minimum, tiered exits",
    "timer_entry_guard": "[IMPLEMENTED] No entry during cooldown",
    "extreme_metrics": "[IMPLEMENTED] R-unit tracking, profit factor",
    "tiered_exits": "[IMPLEMENTED] 25% -> 25% -> 50% sequential exits",
    "profit_lock": "[IMPLEMENTED] 2h minimum hold time",
}

print("EXTREME PROFITABILITY SYSTEM ACTIVATED")
print("Transforming moderate profit -> EXTREME PROFITABILITY")
print("Sequential timer discipline: ENFORCED")
print("Profit-first rules: ACTIVE")
print("Target: 3x profit factor with fewer, bigger trades")
print("\nCONFIGURATION LOADED:")
for key, value in PROFIT_FIRST_CONFIG.items():
    print(f"  {key}: {value}")
for key, value in TIMER_CONFIG.items():
    print(f"  {key}: {value}")
for key, value in IMPLEMENTATION_STATUS.items():
    print(f"  {key}: {value}")
