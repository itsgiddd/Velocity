import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()
info = mt5.account_info()
print("Account:", info.login)
print("Balance:", info.balance)
print("Equity:", info.equity)
print("Profit:", info.profit)
print()

positions = mt5.positions_get()
if positions:
    print("Open positions:", len(positions))
    print("-" * 90)
    for p in positions:
        d = "BUY" if p.type == 0 else "SELL"
        print(f"  {p.symbol:8s} {d:4s} | Entry: {p.price_open} | Current: {p.price_current} | SL: {p.sl} | TP: {p.tp} | PnL: {p.profit} | Vol: {p.volume}")
    print("-" * 90)
    print("Total P/L:", sum(p.profit for p in positions))
else:
    print("No open positions")

print()
deals = mt5.history_deals_get(datetime.now() - timedelta(hours=6), datetime.now())
if deals:
    closed = [d for d in deals if d.entry == 1]
    if closed:
        print("Closed trades last 6h:", len(closed))
        for d in closed:
            dtype = "BUY" if d.type == 0 else "SELL"
            print(f"  {d.symbol:8s} {dtype:4s} | Profit: {d.profit} | Vol: {d.volume}")
        print("Closed total:", sum(d.profit for d in closed))

mt5.shutdown()
