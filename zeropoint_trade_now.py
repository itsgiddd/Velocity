#!/usr/bin/env python3
"""
ZeroPoint One-Shot Trader
=========================
Loads the ZeroPoint neural model, scans all 9 symbols at confidence threshold 0.40,
picks the highest-confidence signal, and places a live trade.

Usage:
    python zeropoint_trade_now.py
"""

import sys
import os
import logging
import MetaTrader5 as mt5
import numpy as np
import time
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

from app.mt5_connector import MT5Connector
from app.model_manager import NeuralModelManager
from app.trading_engine import TradingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ZeroPointTradeNow")

# All 9 model symbols
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD",
]

CONFIDENCE_THRESHOLD = 0.40
RISK_PER_TRADE = 0.08  # 8% of account
MODEL_PATH = "zeropoint_neural_model.pth"


def main():
    print("=" * 60)
    print("ZeroPoint One-Shot Trader")
    print(f"Threshold: {CONFIDENCE_THRESHOLD:.0%} | Risk: {RISK_PER_TRADE:.0%}")
    print("=" * 60)

    # --- Connect to MT5 ---
    connector = MT5Connector()
    if not connector.connect():
        print("ERROR: Cannot connect to MT5")
        return False

    account = connector.get_account_info()
    if account:
        print(f"Account: {account.get('login', '?')}")
        print(f"Balance: ${account.get('balance', 0):.2f}")
        print(f"Equity:  ${account.get('equity', 0):.2f}")
    else:
        print("WARNING: Could not get account info")

    # --- Load ZeroPoint model ---
    model_mgr = NeuralModelManager()
    if not model_mgr.load_model(MODEL_PATH):
        print(f"ERROR: Cannot load model from {MODEL_PATH}")
        return False

    meta = getattr(model_mgr, 'metadata', {}) or {}
    trainer_type = meta.get('trainer_type', 'unknown')
    zp_feat_count = meta.get('zeropoint_feature_count', 0)
    feat_dim = meta.get('feature_dim', '?')
    print(f"Model loaded: trainer={trainer_type}, features={feat_dim}, zp_features={zp_feat_count}")

    # --- Create trading engine at 0.40 threshold ---
    engine = TradingEngine(
        mt5_connector=connector,
        model_manager=model_mgr,
        risk_per_trade=RISK_PER_TRADE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        trading_pairs=SYMBOLS,
        max_concurrent_positions=8,
    )

    # --- Scan all symbols ---
    print(f"\nScanning {len(SYMBOLS)} symbols for signals...")
    print("-" * 60)

    best_signal = None
    all_signals = []

    for symbol in SYMBOLS:
        try:
            signal = engine._generate_signal(symbol)
            if signal is not None:
                conf = getattr(signal, 'confidence', 0.0)
                action = getattr(signal, 'action', '?')
                reason = getattr(signal, 'reason', '')
                sl = getattr(signal, 'stop_loss', 0.0)
                tp = getattr(signal, 'take_profit', 0.0)
                size = getattr(signal, 'position_size', 0.0)
                print(
                    f"  {symbol:8s}: {action:4s} @ {conf:.1%} conf | "
                    f"SL={sl:.5f} TP={tp:.5f} | Size={size:.2f} lots | {reason}"
                )
                all_signals.append(signal)
                if best_signal is None or conf > getattr(best_signal, 'confidence', 0):
                    best_signal = signal
            else:
                print(f"  {symbol:8s}: HOLD (no signal)")
        except Exception as e:
            print(f"  {symbol:8s}: ERROR - {e}")

    print("-" * 60)

    if not all_signals:
        print("\nNo tradeable signals found. Market may be closed or all below threshold.")
        return False

    print(f"\nFound {len(all_signals)} signal(s)")

    # --- Show the best signal ---
    sym = getattr(best_signal, 'symbol', '?')
    act = getattr(best_signal, 'action', '?')
    conf = getattr(best_signal, 'confidence', 0.0)
    sl = getattr(best_signal, 'stop_loss', 0.0)
    tp = getattr(best_signal, 'take_profit', 0.0)
    size = getattr(best_signal, 'position_size', 0.0)
    reason = getattr(best_signal, 'reason', '')

    print(f"\n>>> BEST SIGNAL: {sym} {act}")
    print(f"    Confidence: {conf:.1%}")
    print(f"    Stop Loss:  {sl}")
    print(f"    Take Profit: {tp}")
    print(f"    Lot Size:   {size}")
    print(f"    Reason:     {reason}")

    # --- Get current price ---
    sym_info = connector.get_symbol_info(sym)
    if sym_info:
        bid = sym_info.get('bid', 0)
        ask = sym_info.get('ask', 0)
        spread_pts = sym_info.get('spread', 0)
        print(f"    Bid/Ask:    {bid}/{ask} (spread={spread_pts} pts)")

    # --- Execute the trade ---
    print(f"\n>>> EXECUTING {act} {sym} @ {size:.2f} lots...")
    result = engine._process_signal(best_signal)

    if result:
        print(f"\n*** TRADE PLACED SUCCESSFULLY ***")
        if isinstance(result, dict):
            ticket = result.get('ticket', result.get('order', '?'))
            print(f"    Ticket: {ticket}")
    else:
        # _process_signal returns None but may have executed — check positions
        time.sleep(1)
        positions = mt5.positions_get(symbol=sym)
        if positions and len(positions) > 0:
            latest = positions[-1]
            print(f"\n*** TRADE CONFIRMED ON MT5 ***")
            print(f"    Ticket:  {latest.ticket}")
            print(f"    Symbol:  {latest.symbol}")
            print(f"    Type:    {'BUY' if latest.type == 0 else 'SELL'}")
            print(f"    Volume:  {latest.volume}")
            print(f"    Price:   {latest.price_open}")
            print(f"    SL:      {latest.sl}")
            print(f"    TP:      {latest.tp}")
            print(f"    Profit:  ${latest.profit:.2f}")
        else:
            print(f"\nTrade may not have executed. Check MT5 terminal.")
            # Show last MT5 error
            last_err = mt5.last_error()
            if last_err:
                print(f"    MT5 last error: {last_err}")

    return True


if __name__ == "__main__":
    try:
        success = main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        success = False
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        mt5.shutdown()

    print("\nDone.")
    raise SystemExit(0 if success else 1)
