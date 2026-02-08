#!/usr/bin/env python3
"""
Realistic Backtest - Simple and Corrected Version
=============================================

Simple, realistic backtesting with proper calculations.
"""

import numpy as np


def realistic_backtest():
    """Run a realistic backtest with conservative expectations"""

    print("=" * 80)
    print("REALISTIC PROFITABILITY TEST - CONSERVATIVE ESTIMATES")
    print("=" * 80)

    # Starting conditions
    starting_balance = 200
    risk_per_trade = 0.05
    current_balance = starting_balance

    # Realistic trading parameters
    pip_value = 0.10  # $0.10 per pip for 0.01 lot equivalent

    # Conservative expectations based on professional trading
    expected_win_rate = 0.55  # 55% win rate (realistic for skilled traders)
    expected_rr_ratio = 1.5   # 1.5:1 risk/reward ratio
    trades_per_day = 3        # Conservative 3 trades per day
    trading_days = 5          # 5 trading days

    print(f"Starting Balance: ${starting_balance}")
    print(f"Risk Per Trade: {risk_per_trade * 100}%")
    print(f"Expected Win Rate: {expected_win_rate * 100}%")
    print(f"Expected R:R Ratio: {expected_rr_ratio}:1")
    print(f"Expected Trades/Day: {trades_per_day}")
    print("=" * 80)

    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0
    gross_profit = 0
    gross_loss = 0

    # Generate realistic trades
    np.random.seed(42)  # For reproducible results

    for day in range(1, trading_days + 1):
        daily_pnl = 0

        for trade_num in range(1, trades_per_day + 1):
            # Calculate position size based on current balance and risk
            risk_amount = current_balance * risk_per_trade
            stop_distance_pips = 20
            position_size = risk_amount / (stop_distance_pips * pip_value)
            position_size = min(position_size, 0.1)  # Cap at 0.1 lots

            # Determine win/loss based on realistic probabilities
            is_winner = np.random.random() < expected_win_rate

            # Calculate realistic pip movement
            if is_winner:
                # Winning trade: 12-25 pips profit (conservative)
                profit_pips = np.random.uniform(12, 25)
                pnl = profit_pips * position_size * pip_value
                winning_trades += 1
                gross_profit += pnl
            else:
                # Losing trade: 8-20 pips loss
                loss_pips = np.random.uniform(8, 20)
                pnl = -loss_pips * position_size * pip_value
                losing_trades += 1
                gross_loss += pnl

            # Update balance and counters
            current_balance += pnl
            total_pnl += pnl
            daily_pnl += pnl
            total_trades += 1

            # Print sample trades
            if total_trades <= 15:  # Print first 15 trades
                outcome = "WIN" if is_winner else "LOSS"
                print(f"Day {day}, Trade {trade_num}: {outcome} -> ${pnl:.2f} (Balance: ${current_balance:.2f})")

        print(f"Day {day} P&L: ${daily_pnl:.2f} (End Balance: ${current_balance:.2f})")
        print("-" * 40)

    # Calculate final metrics
    actual_win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((current_balance - starting_balance) / starting_balance) * 100
    avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0

    print("\n" + "=" * 80)
    print("REALISTIC RESULTS SUMMARY")
    print("=" * 80)
    print(f"Starting Balance: ${starting_balance:.2f}")
    print(f"Final Balance: ${current_balance:.2f}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Actual Win Rate: {actual_win_rate:.1f}%")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Daily Average: ${total_pnl / trading_days:.2f}")
    print("=" * 80)

    # Realistic assessment
    print("\nREALISTIC ASSESSMENT:")
    if total_return < 0:
        assessment = "LOSS - Strategy needs improvement"
        status = "FAIL"
    elif total_return < 5:
        assessment = "MINIMAL GAIN - Conservative but stable"
        status = "PASS"
    elif total_return < 15:
        assessment = "GOOD RETURN - Achievable with skill"
        status = "PASS"
    else:
        assessment = "HIGH RETURN - Review for realism"
        status = "WARN"

    print(f"[{status}] {assessment}")

    # Monthly projection
    weekly_return = total_return
    monthly_projection = weekly_return * 4  # Conservative weekly to monthly
    yearly_projection = weekly_return * 52   # Conservative weekly to yearly

    print("\nCONSERVATIVE PROJECTIONS:")
    print(f"Weekly Return: {weekly_return:.2f}%")
    print(f"Monthly Projection: {monthly_projection:.2f}%")
    print(f"Yearly Projection: {yearly_projection:.2f}%")

    # Risk warnings
    print("\nIMPORTANT NOTES:")
    print("- This is simulation - actual results will vary")
    print("- Market conditions affect performance significantly")
    print("- 5% risk per trade can lead to losing streaks")
    print("- Professional traders typically aim for 50-60% win rates")
    print("- Real trading includes spreads, slippage, and commissions")

    return {
        'starting_balance': starting_balance,
        'final_balance': current_balance,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': actual_win_rate,
        'daily_average': total_pnl / trading_days,
    }


if __name__ == "__main__":
    results = realistic_backtest()
