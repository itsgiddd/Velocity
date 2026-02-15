#!/usr/bin/env python3
"""
V4 Compounding Backtest — Proper Risk Sizing + Currency Conversion
==================================================================
Simulates the LIVE trading system behavior:
  - 30% risk per trade (adaptive: +25% after 3 wins, -37.5% after loss)
  - Lot sizing based on balance, SL distance, and tick_value
  - Proper JPY/CAD → USD conversion
  - Lot caps by balance tier (matching live system)
  - Single position at a time per symbol (matching live system)
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import MetaTrader5 as mt5
from app.zeropoint_signal import (
    compute_zeropoint_state,
    ATR_PERIOD, ATR_MULTIPLIER,
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,
    SL_BUFFER_PCT, SL_ATR_MIN_MULT, SWING_LOOKBACK,
    BE_TRIGGER_MULT, BE_BUFFER_MULT,
    PROFIT_TRAIL_DISTANCE_MULT, STALL_BARS,
    MICRO_TP_MULT, MICRO_TP_PCT,
)

SYMBOLS = ["AUDUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDJPY"]
FETCH_BARS = 5000
WARMUP_BARS = 50
STARTING_BALANCE = 200.0

# Adaptive risk parameters (matching live system)
BASE_RISK_PCT = 0.30          # 30% risk per trade
WIN_STREAK_BONUS = 0.25       # +25% after 3 consecutive wins
MAX_RISK_PCT = 0.40           # Cap at 40%
LOSS_PENALTY = 0.375          # -37.5% after a loss
HIGH_BALANCE_CAP_RISK = 0.20  # Cap at 20% above $50K

# Lot caps by balance tier (matching live _calc_lot_size)
LOT_CAP_TABLE = [
    (500, 0.10), (1000, 0.20), (3000, 0.50),
    (5000, 1.00), (10000, 2.00), (50000, 5.00),
    (float('inf'), 10.00),
]


def resolve_symbol(ticker):
    for c in [ticker, ticker + ".raw", ticker + "m", ticker + ".a", ticker + ".e"]:
        info = mt5.symbol_info(c)
        if info is not None:
            mt5.symbol_select(c, True)
            return c
    return None


def get_symbol_specs():
    """Get tick_value and tick_size for each symbol from MT5."""
    specs = {}
    for sym in SYMBOLS:
        resolved = resolve_symbol(sym)
        if resolved:
            info = mt5.symbol_info(resolved)
            if info:
                tick_size = info.trade_tick_size or info.point
                tick_value = info.trade_tick_value
                if tick_value <= 0:
                    tick_value = info.trade_contract_size * tick_size
                specs[sym] = {
                    "tick_size": tick_size,
                    "tick_value": tick_value,
                    "contract_size": info.trade_contract_size,
                    "volume_min": info.volume_min,
                    "volume_max": info.volume_max,
                    "volume_step": info.volume_step,
                    "point": info.point,
                    "currency_profit": info.currency_profit,
                }
                print(f"    {sym}: tick_val={tick_value:.6f} tick_sz={tick_size:.6f} "
                      f"contract={info.trade_contract_size:.0f} "
                      f"profit_currency={info.currency_profit} "
                      f"vol_min={info.volume_min} vol_step={info.volume_step}")
    return specs


def get_conversion_rates():
    """Get USDJPY and USDCAD rates for PnL conversion."""
    rates = {}
    for pair in ["USDJPY", "USDCAD"]:
        resolved = resolve_symbol(pair)
        if resolved:
            tick = mt5.symbol_info_tick(resolved)
            if tick:
                rates[pair] = (tick.bid + tick.ask) / 2.0
    return rates


def pnl_to_usd(raw_pnl, symbol, conv_rates):
    """Convert raw PnL (in quote currency) to USD."""
    sym = symbol.upper()
    if sym.endswith("JPY"):
        return raw_pnl / conv_rates.get("USDJPY", 152.0)
    elif sym.endswith("CAD"):
        return raw_pnl / conv_rates.get("USDCAD", 1.36)
    return raw_pnl  # USD-quote pairs


def calc_lot_size(balance, entry, sl, sym_specs, risk_pct):
    """Calculate lot size matching live system logic."""
    tick_size = sym_specs["tick_size"]
    tick_value = sym_specs["tick_value"]

    sl_distance = abs(entry - sl)
    sl_ticks = sl_distance / tick_size if tick_size > 0 else 0
    loss_per_lot = sl_ticks * tick_value

    if loss_per_lot <= 0:
        return sym_specs["volume_min"]

    risk_amount = balance * risk_pct
    lot = risk_amount / loss_per_lot

    # Round to volume step
    vol_step = sym_specs["volume_step"]
    lot = round(lot / vol_step) * vol_step

    # Clamp to broker limits
    lot = max(sym_specs["volume_min"], min(lot, sym_specs["volume_max"]))

    # Apply balance-tier caps
    for threshold, max_lot in LOT_CAP_TABLE:
        if balance <= threshold:
            lot = min(lot, max_lot)
            break

    return lot


def compute_smart_sl(df, bar_idx, direction, atr_val):
    lookback_start = max(0, bar_idx - SWING_LOOKBACK + 1)
    cur_close = float(df["close"].iloc[bar_idx])
    if direction == "BUY":
        recent_low = float(df["low"].iloc[lookback_start:bar_idx + 1].min())
        buffer = recent_low * SL_BUFFER_PCT
        structural_sl = recent_low - buffer
        atr_min_sl = cur_close - atr_val * SL_ATR_MIN_MULT
        return min(structural_sl, atr_min_sl)
    else:
        recent_high = float(df["high"].iloc[lookback_start:bar_idx + 1].max())
        buffer = recent_high * SL_BUFFER_PCT
        structural_sl = recent_high + buffer
        atr_max_sl = cur_close + atr_val * SL_ATR_MIN_MULT
        return max(structural_sl, atr_max_sl)


class V4Position:
    def __init__(self, sym, direction, entry, sl, atr_val, lot, contract_size):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.original_sl = sl
        self.atr_val = atr_val
        self.total_lot = lot
        self.remaining_lot = lot
        self.contract_size = contract_size
        sign = 1 if direction == "BUY" else -1
        self.tp1 = entry + sign * TP1_MULT_AGG * atr_val
        self.tp2 = entry + sign * TP2_MULT_AGG * atr_val
        self.tp3 = entry + sign * TP3_MULT_AGG * atr_val
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        self.closed = False
        self.partials = []
        self.bars_in_trade = 0
        self.exit_time = None
        self.entry_time = None
        self.max_profit_reached = 0.0
        self.max_favorable_price = entry
        self.sl_hit = False
        self.final_exit_type = None
        self.be_activated = False
        self.profit_lock_sl = None
        self.profit_lock_active = False
        self.micro_tp_hit = False
        self.stall_be_activated = False

    def partial_lot(self):
        return max(0.01, round(self.total_lot / 3, 2))

    def pnl_for_price(self, price, lot):
        if self.direction == 'BUY':
            return (price - self.entry) * lot * self.contract_size
        else:
            return (self.entry - price) * lot * self.contract_size

    def check_bar(self, high, low, close, confirmed_pos):
        if self.closed:
            return []
        events = []
        self.bars_in_trade += 1
        is_buy = self.direction == 'BUY'
        atr = self.atr_val if self.atr_val > 1e-12 else 1.0

        if is_buy:
            cur_profit = high - self.entry
            if high > self.max_favorable_price:
                self.max_favorable_price = high
        else:
            cur_profit = self.entry - low
            if low < self.max_favorable_price:
                self.max_favorable_price = low
        if cur_profit > self.max_profit_reached:
            self.max_profit_reached = cur_profit

        # Post-TP1 trailing stop
        if self.tp1_hit:
            trail_dist = PROFIT_TRAIL_DISTANCE_MULT * atr
            if is_buy:
                new_lock = self.max_favorable_price - trail_dist
                if new_lock > self.entry and (self.profit_lock_sl is None or new_lock > self.profit_lock_sl):
                    self.profit_lock_sl = new_lock
                    self.profit_lock_active = True
            else:
                new_lock = self.max_favorable_price + trail_dist
                if new_lock < self.entry and (self.profit_lock_sl is None or new_lock < self.profit_lock_sl):
                    self.profit_lock_sl = new_lock
                    self.profit_lock_active = True

        # Early breakeven
        if not self.be_activated:
            if self.max_profit_reached >= BE_TRIGGER_MULT * atr:
                be_buffer = BE_BUFFER_MULT * atr
                if is_buy:
                    new_sl = self.entry + be_buffer
                    if new_sl > self.sl:
                        self.sl = new_sl
                        self.be_activated = True
                else:
                    new_sl = self.entry - be_buffer
                    if new_sl < self.sl:
                        self.sl = new_sl
                        self.be_activated = True

        # Stall exit
        if not self.tp1_hit and not self.stall_be_activated:
            if self.bars_in_trade >= STALL_BARS:
                be_buffer = BE_BUFFER_MULT * atr
                if is_buy:
                    new_sl = self.entry + be_buffer
                    if new_sl > self.sl:
                        self.sl = new_sl
                        self.stall_be_activated = True
                        self.be_activated = True
                else:
                    new_sl = self.entry - be_buffer
                    if new_sl < self.sl:
                        self.sl = new_sl
                        self.stall_be_activated = True
                        self.be_activated = True

        # Micro-partial
        if not self.micro_tp_hit and not self.tp1_hit:
            micro_price = (self.entry + MICRO_TP_MULT * atr) if is_buy else (self.entry - MICRO_TP_MULT * atr)
            micro_triggered = (is_buy and high >= micro_price) or (not is_buy and low <= micro_price)
            if micro_triggered:
                self.micro_tp_hit = True
                micro_lot = round(self.total_lot * MICRO_TP_PCT, 2)
                micro_lot = max(0.01, min(micro_lot, self.remaining_lot))
                pnl = self.pnl_for_price(micro_price, micro_lot)
                self.partials.append((micro_price, micro_lot, pnl, 'MICRO_TP'))
                self.remaining_lot = round(self.remaining_lot - micro_lot, 2)
                events.append(('MICRO_TP', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'MICRO_TP'
                    return events

        # TP1
        if not self.tp1_hit:
            if (is_buy and high >= self.tp1) or (not is_buy and low <= self.tp1):
                self.tp1_hit = True
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp1, partial)
                self.partials.append((self.tp1, partial, pnl, 'TP1'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP1', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'TP1'
                    return events

        # TP2
        if self.tp1_hit and not self.tp2_hit:
            if (is_buy and high >= self.tp2) or (not is_buy and low <= self.tp2):
                self.tp2_hit = True
                self.sl = self.entry
                self.be_activated = True
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp2, partial)
                self.partials.append((self.tp2, partial, pnl, 'TP2'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP2', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'TP2'
                    return events

        # TP3
        if self.tp2_hit and not self.tp3_hit:
            if (is_buy and high >= self.tp3) or (not is_buy and low <= self.tp3):
                self.tp3_hit = True
                pnl = self.pnl_for_price(self.tp3, self.remaining_lot)
                self.partials.append((self.tp3, self.remaining_lot, pnl, 'TP3'))
                events.append(('TP3', pnl))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'TP3'
                return events

        # Profit lock SL
        if self.profit_lock_active and self.profit_lock_sl is not None:
            lock_hit = (is_buy and low <= self.profit_lock_sl) or \
                       (not is_buy and high >= self.profit_lock_sl)
            if lock_hit:
                pnl = self.pnl_for_price(self.profit_lock_sl, self.remaining_lot)
                self.partials.append((self.profit_lock_sl, self.remaining_lot, pnl, 'PROFIT_LOCK'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'PROFIT_LOCK'
                events.append(('PROFIT_LOCK', pnl))
                return events

        # SL
        if (is_buy and low <= self.sl) or (not is_buy and high >= self.sl):
            if self.stall_be_activated:
                exit_label = 'SL_STALL'
            elif self.be_activated:
                exit_label = 'SL_BE'
            elif self.tp1_hit:
                exit_label = 'SL_AFTER_TP'
            else:
                exit_label = 'SL'
                self.sl_hit = True
            pnl = self.pnl_for_price(self.sl, self.remaining_lot)
            self.partials.append((self.sl, self.remaining_lot, pnl, exit_label))
            self.remaining_lot = 0
            self.closed = True
            self.final_exit_type = exit_label
            events.append((exit_label, pnl))
            return events

        # ZP flip
        if confirmed_pos != 0:
            if (is_buy and confirmed_pos == -1) or (not is_buy and confirmed_pos == 1):
                pnl = self.pnl_for_price(close, self.remaining_lot)
                self.partials.append((close, self.remaining_lot, pnl, 'ZP_FLIP'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'ZP_FLIP'
                events.append(('ZP_FLIP', pnl))
                return events

        return events

    def force_close(self, price):
        if self.closed or self.remaining_lot <= 0:
            return 0
        pnl = self.pnl_for_price(price, self.remaining_lot)
        self.partials.append((price, self.remaining_lot, pnl, 'END'))
        self.remaining_lot = 0
        self.closed = True
        self.final_exit_type = 'END'
        return pnl

    @property
    def total_pnl(self):
        return sum(p[2] for p in self.partials)


def main():
    print("=" * 110)
    print("  V4 COMPOUNDING BACKTEST — Adaptive Risk + Currency Conversion")
    print(f"  Base risk: {BASE_RISK_PCT*100:.0f}% | Win streak bonus: +{WIN_STREAK_BONUS*100:.0f}% after 3 wins | Loss penalty: -{LOSS_PENALTY*100:.1f}%")
    print(f"  Starting balance: ${STARTING_BALANCE:.2f}")
    print("=" * 110)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    acct = mt5.account_info()
    if acct:
        print(f"  MT5: Account {acct.login} | Leverage 1:{acct.leverage}")

    # Get symbol specs and conversion rates
    print(f"\n  Symbol specs from broker:")
    sym_specs = get_symbol_specs()
    conv_rates = get_conversion_rates()
    print(f"\n  Conversion rates: USDJPY={conv_rates.get('USDJPY', 'N/A'):.4f}, USDCAD={conv_rates.get('USDCAD', 'N/A'):.4f}")

    # Fetch data
    symbol_data = {}
    print(f"\n  Fetching up to {FETCH_BARS} H4 bars per symbol...")
    for sym in SYMBOLS:
        resolved = resolve_symbol(sym)
        if resolved is None:
            continue
        rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H4, 0, FETCH_BARS)
        if rates is None or len(rates) < 100:
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df_zp = compute_zeropoint_state(df)
        if df_zp is None or len(df_zp) < WARMUP_BARS:
            continue
        symbol_data[sym] = df_zp
        days = (df_zp["time"].iloc[-1] - df_zp["time"].iloc[WARMUP_BARS]).days
        print(f"    {sym}: {len(df_zp)} bars ({days} days)")

    if not symbol_data:
        print("ERROR: No data loaded")
        mt5.shutdown()
        return

    # ═══════════════════════════════════════════════════════════════
    # Build a unified timeline of all signals across all symbols
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Building unified signal timeline...")

    signals = []  # (time, sym, direction, entry, sl, atr_val, bar_idx)
    for sym, df in symbol_data.items():
        n = len(df)
        for i in range(WARMUP_BARS, n):
            row = df.iloc[i]
            buy_sig = bool(row.get("buy_signal", False))
            sell_sig = bool(row.get("sell_signal", False))
            if buy_sig or sell_sig:
                atr_val = float(row["atr"])
                if np.isnan(atr_val) or atr_val <= 0:
                    continue
                direction = "BUY" if buy_sig else "SELL"
                entry = float(row["close"])
                smart_sl = compute_smart_sl(df, i, direction, atr_val)
                signals.append((row["time"], sym, direction, entry, smart_sl, atr_val, i))

    signals.sort(key=lambda x: x[0])
    print(f"  Total signals: {len(signals)}")

    # ═══════════════════════════════════════════════════════════════
    # Simulate with compounding
    # ═══════════════════════════════════════════════════════════════
    balance = STARTING_BALANCE
    risk_pct = BASE_RISK_PCT
    win_streak = 0
    all_trades = []
    open_positions = {}  # sym -> V4Position
    balance_curve = [(None, balance)]  # (time, balance)

    # We need to process bar-by-bar across all symbols in time order
    # Build a merged bar stream
    bar_events = []  # (time, sym, high, low, close, atr, pos, buy_sig, sell_sig, bar_idx)
    for sym, df in symbol_data.items():
        n = len(df)
        for i in range(WARMUP_BARS, n):
            row = df.iloc[i]
            atr_val = float(row["atr"])
            if np.isnan(atr_val) or atr_val <= 0:
                continue
            bar_events.append((
                row["time"], sym,
                float(row["high"]), float(row["low"]), float(row["close"]),
                atr_val, int(row.get("pos", 0)),
                bool(row.get("buy_signal", False)), bool(row.get("sell_signal", False)),
                i,
            ))

    bar_events.sort(key=lambda x: x[0])
    print(f"  Total bar events: {len(bar_events)}")
    print(f"\n  Simulating with compounding risk sizing...\n")

    for time, sym, high, low, close, atr_val, pos, buy_sig, sell_sig, bar_idx in bar_events:
        if sym not in sym_specs:
            continue

        # Check existing position for this symbol
        if sym in open_positions:
            p = open_positions[sym]
            if not p.closed:
                p.check_bar(high, low, close, pos)
                if p.closed:
                    p.exit_time = time
                    # Convert PnL to USD
                    raw_pnl = p.total_pnl
                    usd_pnl = pnl_to_usd(raw_pnl, sym, conv_rates)
                    p._usd_pnl = usd_pnl
                    balance += usd_pnl
                    all_trades.append(p)
                    balance_curve.append((time, balance))
                    del open_positions[sym]

                    # Adaptive risk
                    if usd_pnl > 0:
                        win_streak += 1
                        if win_streak >= 3:
                            risk_pct = min(BASE_RISK_PCT * (1 + WIN_STREAK_BONUS), MAX_RISK_PCT)
                    else:
                        win_streak = 0
                        risk_pct = max(BASE_RISK_PCT * (1 - LOSS_PENALTY), 0.05)

                    if balance > 50000:
                        risk_pct = min(risk_pct, HIGH_BALANCE_CAP_RISK)

        # Open new position on signal
        if buy_sig or sell_sig:
            direction = "BUY" if buy_sig else "SELL"
            entry = close
            smart_sl = compute_smart_sl(symbol_data[sym], bar_idx, direction, atr_val)

            # Close existing position on flip
            if sym in open_positions and not open_positions[sym].closed:
                p = open_positions[sym]
                p.force_close(close)
                p.exit_time = time
                raw_pnl = p.total_pnl
                usd_pnl = pnl_to_usd(raw_pnl, sym, conv_rates)
                p._usd_pnl = usd_pnl
                balance += usd_pnl
                all_trades.append(p)
                balance_curve.append((time, balance))

                if usd_pnl > 0:
                    win_streak += 1
                    if win_streak >= 3:
                        risk_pct = min(BASE_RISK_PCT * (1 + WIN_STREAK_BONUS), MAX_RISK_PCT)
                else:
                    win_streak = 0
                    risk_pct = max(BASE_RISK_PCT * (1 - LOSS_PENALTY), 0.05)

                if balance > 50000:
                    risk_pct = min(risk_pct, HIGH_BALANCE_CAP_RISK)

            # Don't trade if balance is too low
            if balance < 10:
                continue

            # Calculate lot size based on current balance
            lot = calc_lot_size(balance, entry, smart_sl, sym_specs[sym], risk_pct)

            p = V4Position(
                sym=sym, direction=direction, entry=entry, sl=smart_sl,
                atr_val=atr_val, lot=lot, contract_size=sym_specs[sym]["contract_size"],
            )
            p.entry_time = time
            open_positions[sym] = p

    # Force close remaining positions
    for sym, p in open_positions.items():
        if not p.closed:
            df = symbol_data[sym]
            p.force_close(float(df.iloc[-1]["close"]))
            p.exit_time = df.iloc[-1]["time"]
            raw_pnl = p.total_pnl
            usd_pnl = pnl_to_usd(raw_pnl, sym, conv_rates)
            p._usd_pnl = usd_pnl
            balance += usd_pnl
            all_trades.append(p)

    mt5.shutdown()

    if not all_trades:
        print("No trades generated!")
        return

    # Sort by exit time
    all_trades.sort(key=lambda t: t.exit_time if t.exit_time else pd.Timestamp.min)

    # ═══════════════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════════════
    usd_pnls = [t._usd_pnl for t in all_trades]
    wins = [p for p in usd_pnls if p > 0]
    losses = [p for p in usd_pnls if p <= 0]
    total_pnl = sum(usd_pnls)
    win_rate = len(wins) / len(usd_pnls) * 100
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = total_pnl / len(usd_pnls)

    # Max drawdown
    peak = STARTING_BALANCE
    max_dd = 0
    max_dd_pct = 0
    running = STARTING_BALANCE
    for p in usd_pnls:
        running += p
        if running > peak:
            peak = running
        dd = peak - running
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    first_time = min(t.exit_time for t in all_trades if t.exit_time is not None)
    last_time = max(t.exit_time for t in all_trades if t.exit_time is not None)
    days_tested = (last_time - first_time).days

    print(f"\n{'=' * 110}")
    print(f"  V4 COMPOUNDING RESULTS (all values in USD)")
    print(f"{'=' * 110}")
    print(f"\n  Period: {first_time.strftime('%Y-%m-%d')} -> {last_time.strftime('%Y-%m-%d')} ({days_tested} days / {days_tested/7:.0f} weeks)")
    print(f"  Total trades:      {len(all_trades)}")
    print(f"  Winners:           {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losers:            {len(losses)} ({100 - win_rate:.1f}%)")
    print(f"\n  Starting balance:  ${STARTING_BALANCE:.2f}")
    print(f"  Final balance:     ${balance:,.2f}")
    print(f"  Total PnL:         ${total_pnl:+,.2f}")
    print(f"  Return:            {(total_pnl / STARTING_BALANCE * 100):+,.1f}%")
    print(f"\n  Avg win:           ${np.mean(wins):,.2f}" if wins else "  Avg win:           N/A")
    print(f"  Avg loss:          ${np.mean(losses):,.2f}" if losses else "  Avg loss:          N/A")
    print(f"  Expectancy:        ${expectancy:,.2f} per trade")
    print(f"  Profit factor:     {profit_factor:.2f}")
    print(f"  Max drawdown:      ${max_dd:,.2f} ({max_dd_pct:.1f}%)")

    # Per-symbol
    print(f"\n  {'Symbol':>8} | {'Trades':>7} | {'Win%':>6} | {'USD PnL':>14} | {'Avg Win':>10} | {'Avg Loss':>10} | {'PF':>6}")
    print(f"  " + "-" * 80)
    for sym in sorted(set(t.sym for t in all_trades)):
        st = [t for t in all_trades if t.sym == sym]
        sp = [t._usd_pnl for t in st]
        sw = [p for p in sp if p > 0]
        sl = [p for p in sp if p <= 0]
        wr = len(sw) / len(sp) * 100
        pnl = sum(sp)
        aw = np.mean(sw) if sw else 0
        al = np.mean(sl) if sl else 0
        gp = sum(sw)
        gl = abs(sum(sl))
        pf = gp / gl if gl > 0 else float('inf')
        print(f"  {sym:>8} | {len(sp):>7} | {wr:>5.1f}% | ${pnl:>+12,.2f} | ${aw:>8,.2f} | ${al:>8,.2f} | {pf:>5.2f}")

    # Exit types
    print(f"\n  Exit Type Breakdown:")
    exit_counter = Counter()
    exit_pnl_usd = defaultdict(float)
    for t in all_trades:
        exit_counter[t.final_exit_type] += 1
        exit_pnl_usd[t.final_exit_type] += t._usd_pnl
    total_exits = len(all_trades)
    print(f"  {'Exit Type':>14} | {'Count':>7} | {'%':>6} | {'USD PnL':>14} | {'Avg USD':>10}")
    print(f"  " + "-" * 65)
    for et in sorted(exit_counter.keys(), key=lambda x: -exit_counter[x]):
        cnt = exit_counter[et]
        pct = cnt / total_exits * 100
        tot = exit_pnl_usd[et]
        avg = tot / cnt
        print(f"  {et:>14} | {cnt:>7} | {pct:>5.1f}% | ${tot:>+12,.2f} | ${avg:>+8,.2f}")

    # Balance milestones
    print(f"\n  Balance Milestones:")
    milestones = [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    running = STARTING_BALANCE
    trade_num = 0
    for t in all_trades:
        trade_num += 1
        running += t._usd_pnl
        for m in milestones[:]:
            if running >= m:
                days_to = (t.exit_time - first_time).days if t.exit_time and first_time else 0
                print(f"    ${m:>10,} reached at trade #{trade_num:>5} ({days_to:>4} days, {t.exit_time.strftime('%Y-%m-%d') if t.exit_time else 'N/A'})")
                milestones.remove(m)

    # Equity curve stats
    print(f"\n  Equity Curve Checkpoints:")
    for pct in [25, 50, 75, 100]:
        idx = int(len(all_trades) * pct / 100) - 1
        if 0 <= idx < len(all_trades):
            running_at = STARTING_BALANCE + sum(t._usd_pnl for t in all_trades[:idx+1])
            t = all_trades[idx]
            print(f"    {pct:>3}% of trades (#{idx+1}): ${running_at:>12,.2f} @ {t.exit_time.strftime('%Y-%m-%d') if t.exit_time else 'N/A'}")

    # Lot size evolution
    print(f"\n  Lot Size Samples (first, middle, last 3 trades):")
    for label, indices in [("First 3", range(min(3, len(all_trades)))),
                           ("Middle 3", range(max(0, len(all_trades)//2-1), min(len(all_trades), len(all_trades)//2+2))),
                           ("Last 3", range(max(0, len(all_trades)-3), len(all_trades)))]:
        for i in indices:
            t = all_trades[i]
            print(f"    [{label}] #{i+1} {t.sym} {t.direction} lot={t.total_lot:.2f} pnl=${t._usd_pnl:+,.2f}")

    # Losing trades
    losers = [(t, t._usd_pnl) for t in all_trades if t._usd_pnl <= 0]
    if losers:
        print(f"\n  ALL {len(losers)} LOSING TRADES:")
        print(f"  {'#':>3} | {'Symbol':>8} | {'Dir':>4} | {'Lot':>5} | {'Exit':>12} | {'USD PnL':>12} | {'Balance After':>14}")
        print(f"  " + "-" * 80)
        running = STARTING_BALANCE
        loss_idx = 0
        for i, t in enumerate(all_trades):
            running += t._usd_pnl
            if t._usd_pnl <= 0:
                loss_idx += 1
                print(f"  {loss_idx:>3} | {t.sym:>8} | {t.direction:>4} | {t.total_lot:>5.2f} | {t.final_exit_type:>12} | ${t._usd_pnl:>+10,.2f} | ${running:>12,.2f}")

    print(f"\n{'=' * 110}")
    print(f"  FINAL VERDICT")
    print(f"{'=' * 110}")
    print(f"  ${STARTING_BALANCE:.0f} -> ${balance:,.2f} over {days_tested} days ({days_tested/30:.0f} months)")
    print(f"  {win_rate:.1f}% win rate | PF {profit_factor:.2f} | Max DD ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
    print(f"  All PnL properly converted to USD (JPY÷{conv_rates.get('USDJPY', 152):.2f}, CAD÷{conv_rates.get('USDCAD', 1.36):.4f})")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
