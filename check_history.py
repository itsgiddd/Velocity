"""Check live trade history and open positions."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()

info = mt5.account_info()
print("=== ACCOUNT ===")
print(f"  Balance: ${info.balance:.2f}")
print(f"  Equity:  ${info.equity:.2f}")
print(f"  P/L:     ${info.equity - info.balance:+.2f}")
print()

now = datetime.now()
week_ago = now - timedelta(days=7)
deals = mt5.history_deals_get(week_ago, now)

if deals:
    print(f"=== TRADE HISTORY (last 7 days) - {len(deals)} deals ===")
    print(f"{'Time':<20} {'Symbol':<10} {'Dir':<5} {'I/O':<4} {'Vol':>5} {'Price':>12} {'P/L':>10} {'Comment'}")
    print("-" * 100)

    total_pnl = 0
    wins = 0
    losses = 0
    for d in deals:
        if d.symbol == '' and d.profit == 0:
            continue
        t = datetime.fromtimestamp(d.time)
        typ = "BUY" if d.type == 0 else "SELL" if d.type == 1 else str(d.type)
        entry = "IN" if d.entry == 0 else "OUT" if d.entry == 1 else str(d.entry)
        print(f"{t.strftime('%m/%d %H:%M'):<20} {d.symbol:<10} {typ:<5} {entry:<4} {d.volume:>5.2f} {d.price:>12.5f} {d.profit:>+10.2f} {d.comment}")
        if d.entry == 1:
            total_pnl += d.profit
            if d.profit > 0:
                wins += 1
            elif d.profit < 0:
                losses += 1

    print("-" * 100)
    total_closed = wins + losses
    wr = wins / total_closed * 100 if total_closed > 0 else 0
    print(f"Closed: {total_closed} trades | Wins: {wins} | Losses: {losses} | WR: {wr:.0f}% | Net P/L: ${total_pnl:+.2f}")
else:
    print("No trade history found")

print()

positions = mt5.positions_get()
if positions:
    print(f"=== OPEN POSITIONS ({len(positions)}) ===")
    open_pnl = 0
    for p in positions:
        trade_dir = "BUY" if p.type == 0 else "SELL"
        t = datetime.fromtimestamp(p.time)
        age_hrs = (now - t).total_seconds() / 3600
        sl_dist = abs(p.price_open - p.sl) if p.sl > 0 else 0
        tp_dist = abs(p.tp - p.price_open) if p.tp > 0 else 0
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        print(f"  {p.symbol:<10} {trade_dir} {p.volume:.2f} | entry={p.price_open:.5f} SL={p.sl:.5f} TP={p.tp:.5f} R:R={rr:.2f} | P/L=${p.profit:+.2f} | {age_hrs:.0f}h old")
        open_pnl += p.profit
    print(f"  Total open P/L: ${open_pnl:+.2f}")
else:
    print("No open positions")

mt5.shutdown()
