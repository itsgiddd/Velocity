import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()
info = mt5.account_info()
print(f"Account: {info.login}")
print(f"Balance: ${info.balance:.2f}")
print(f"Equity:  ${info.equity:.2f}")
print()

now = datetime.now()
start = now - timedelta(hours=6)
deals = mt5.history_deals_get(start, now)
if deals:
    print(f"ALL deals in last 6 hours: {len(deals)}")
    print("-" * 100)
    total_profit = 0
    for d in deals:
        symbol = d.symbol or "---"
        volume = d.volume
        price = d.price
        profit = d.profit
        swap = d.swap
        commission = d.commission
        net = profit + swap + commission
        dtype = "BUY" if d.type == 0 else "SELL"
        entry_str = {0: "IN", 1: "OUT", 2: "INOUT", 3: "OUT_BY"}.get(d.entry, f"?{d.entry}")
        t = datetime.fromtimestamp(d.time)
        total_profit += profit
        print(
            f"  {t.strftime('%H:%M:%S')} | {entry_str:5s} | {symbol:8s} {dtype:4s} {volume:.2f} lots "
            f"@ {price:.5f} | P/L: ${profit:+.2f} swap:${swap:.2f} comm:${commission:.2f}"
        )
    print("-" * 100)
    print(f"TOTAL PROFIT COLUMN: ${total_profit:+.2f}")
else:
    print("No deals found in last 6 hours")

print(f"\nBalance went from $200.00 -> ${info.balance:.2f}")
print(f"That's ${info.balance - 200:+.2f} ({(info.balance - 200)/200*100:+.1f}%)")

positions = mt5.positions_get()
if positions and len(positions) > 0:
    print(f"\nOpen positions: {len(positions)}")
else:
    print("\nNo open positions - all closed")

mt5.shutdown()
