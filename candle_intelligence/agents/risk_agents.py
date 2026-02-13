"""
Risk & Timing Agents (4 agents)
================================
12. RiskRewardAgent     — R:R quality and SL sanity
13. SessionTimingAgent  — trading session quality
14. DrawdownGuardAgent  — account state / risk manager (VETO power)
15. CorrelationAgent    — correlated exposure check
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from candle_intelligence.agent_base import (
    TradingAgent, TradeContext, AgentVerdict,
    STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL,
    VETO_CRITICAL_DRAWDOWN, VETO_RR_BELOW_1_0,
)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 12: Risk/Reward Agent
# ═══════════════════════════════════════════════════════════════════════════

class RiskRewardAgent(TradingAgent):
    """Evaluates R:R ratio, SL distance sanity (vs ATR), and whether
    the path to TP1 is realistic."""

    name = "Risk/Reward"
    specialty = "risk_reward"
    default_weight = 0.7

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        atr = ctx.atr_value if ctx.atr_value > 1e-12 else 1.0
        is_buy = ctx.direction == "BUY"

        sl_dist = abs(ctx.entry_price - ctx.stop_loss)
        tp1_dist = abs(ctx.tp1 - ctx.entry_price)
        tp2_dist = abs(ctx.tp2 - ctx.entry_price)
        tp3_dist = abs(ctx.tp3 - ctx.entry_price)
        rr = tp1_dist / sl_dist if sl_dist > 1e-12 else 0.0

        # Effective R:R for 3-TP partial system:
        # 1/3 at TP1 + 1/3 at TP2 + 1/3 at TP3, risk = full SL
        # Weighted avg TP = (TP1 + TP2 + TP3) / 3
        avg_tp_dist = (tp1_dist + tp2_dist + tp3_dist) / 3.0
        effective_rr = avg_tp_dist / sl_dist if sl_dist > 1e-12 else 0.0

        sl_atr = sl_dist / atr

        score = 0.0
        reasons = []
        flags = []

        # --- Effective R:R evaluation (accounts for 3-TP system) ---
        if effective_rr >= 3.0:
            score += 0.5
            reasons.append(f"excellent eff. R:R {effective_rr:.1f} (TP1 R:R {rr:.1f})")
        elif effective_rr >= 2.0:
            score += 0.3
            reasons.append(f"good eff. R:R {effective_rr:.1f}")
        elif effective_rr >= 1.5:
            score += 0.15
            reasons.append(f"acceptable eff. R:R {effective_rr:.1f}")
        elif effective_rr >= 1.0:
            score += 0.05
            reasons.append(f"marginal eff. R:R {effective_rr:.1f}")
        else:
            score -= 0.3
            reasons.append(f"poor eff. R:R {effective_rr:.1f}")
            flags.append(VETO_RR_BELOW_1_0)

        # --- SL distance sanity check ---
        # Smart Structure SL uses swing highs/lows (typically 2-5x ATR)
        # Wider SLs are normal and expected with structural stops
        if sl_atr > 5.0:
            score -= 0.2
            reasons.append(f"SL very wide ({sl_atr:.1f}x ATR)")
        elif sl_atr < 0.5:
            score -= 0.2
            reasons.append(f"SL too tight ({sl_atr:.1f}x ATR, likely to get hit)")
        elif 1.0 <= sl_atr <= 3.0:
            score += 0.1
            reasons.append(f"SL well-sized ({sl_atr:.1f}x ATR)")

        # --- SL direction sanity ---
        if is_buy and ctx.stop_loss >= ctx.entry_price:
            score -= 0.5
            reasons.append("SL above entry for BUY (invalid)")
            flags.append(VETO_RR_BELOW_1_0)
        elif not is_buy and ctx.stop_loss <= ctx.entry_price:
            score -= 0.5
            reasons.append("SL below entry for SELL (invalid)")
            flags.append(VETO_RR_BELOW_1_0)

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else f"R:R {rr:.1f}"

        return self._verdict(vote, conf, reason, flags)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 13: Session Timing Agent
# ═══════════════════════════════════════════════════════════════════════════

class SessionTimingAgent(TradingAgent):
    """Checks trading session (London/NY/overlap/Asian) and reports
    session quality for this symbol."""

    name = "Session Timing"
    specialty = "session"
    default_weight = 1.0

    # Per-symbol session quality (which sessions work best for each pair)
    # Based on typical forex session activity patterns
    SESSION_QUALITY = {
        # Major pairs: best during London and NY
        "EURUSD": {"london": 0.9, "ny": 0.85, "overlap": 1.0, "asian": 0.3},
        "GBPUSD": {"london": 0.95, "ny": 0.8, "overlap": 0.95, "asian": 0.2},
        "USDJPY": {"london": 0.7, "ny": 0.75, "overlap": 0.85, "asian": 0.8},
        # Cross pairs
        "EURJPY": {"london": 0.8, "ny": 0.7, "overlap": 0.85, "asian": 0.7},
        "GBPJPY": {"london": 0.85, "ny": 0.75, "overlap": 0.9, "asian": 0.6},
        # Commodity currencies
        "AUDUSD": {"london": 0.6, "ny": 0.65, "overlap": 0.75, "asian": 0.85},
        "NZDUSD": {"london": 0.55, "ny": 0.6, "overlap": 0.7, "asian": 0.8},
        "USDCAD": {"london": 0.7, "ny": 0.9, "overlap": 0.85, "asian": 0.3},
        # Crypto
        "BTCUSD": {"london": 0.7, "ny": 0.8, "overlap": 0.85, "asian": 0.6},
    }

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        now = datetime.now(timezone.utc)
        utc_hour = now.hour

        # Determine active session
        if 7 <= utc_hour < 12:
            session = "london"
            session_name = "London"
        elif 12 <= utc_hour < 16:
            session = "overlap"
            session_name = "London/NY Overlap"
        elif 16 <= utc_hour < 21:
            session = "ny"
            session_name = "New York"
        else:
            session = "asian"
            session_name = "Asian"

        # Get symbol quality for this session
        symbol = ctx.symbol.upper().replace(".", "").replace("#", "")
        qualities = self.SESSION_QUALITY.get(symbol, {
            "london": 0.7, "ny": 0.7, "overlap": 0.8, "asian": 0.5
        })

        quality = qualities.get(session, 0.5)

        score = 0.0
        reasons = []

        if quality >= 0.85:
            score += 0.4
            reasons.append(f"{session_name} is prime time for {symbol} ({quality:.0%})")
        elif quality >= 0.65:
            score += 0.15
            reasons.append(f"{session_name} is decent for {symbol} ({quality:.0%})")
        elif quality >= 0.45:
            score -= 0.1
            reasons.append(f"{session_name} is mediocre for {symbol} ({quality:.0%})")
        else:
            score -= 0.35
            reasons.append(f"{session_name} is poor for {symbol} ({quality:.0%})")

        # Weekend proximity check (Friday late session)
        weekday = now.weekday()  # 0=Mon, 4=Fri
        if weekday == 4 and utc_hour >= 19:
            score -= 0.3
            reasons.append("late Friday — weekend gap risk")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else f"{session_name} session"

        return self._verdict(vote, conf, reason)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 14: Drawdown Guard Agent (VETO power)
# ═══════════════════════════════════════════════════════════════════════════

class DrawdownGuardAgent(TradingAgent):
    """Checks account drawdown, consecutive losses, daily/weekly P/L.
    Acts as the risk manager — can VETO trades."""

    name = "Drawdown Guard"
    specialty = "risk_management"
    default_weight = 0.8

    # Thresholds — calibrated for small accounts ($200-$1000)
    # Small accounts have naturally higher % swings per trade
    MAX_DAILY_LOSS_PCT = 0.05      # 5% daily loss cap (was 3%)
    MAX_WEEKLY_LOSS_PCT = 0.10     # 10% weekly loss cap (was 6%)
    MAX_DRAWDOWN_PCT = 0.15        # 15% max drawdown (was 10%)
    MAX_CONSECUTIVE_LOSSES = 4      # After 4 consecutive losses, reduce

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        balance = ctx.balance if ctx.balance > 0 else 1.0
        equity = ctx.equity if ctx.equity > 0 else balance

        score = 0.0
        reasons = []
        flags = []

        # --- Drawdown check ---
        drawdown = (balance - equity) / balance if balance > 0 else 0
        if drawdown > self.MAX_DRAWDOWN_PCT:
            score -= 0.5
            reasons.append(f"high drawdown: {drawdown:.1%} (max {self.MAX_DRAWDOWN_PCT:.0%})")
            flags.append(VETO_CRITICAL_DRAWDOWN)
        elif drawdown > self.MAX_DRAWDOWN_PCT * 0.7:
            score -= 0.2
            reasons.append(f"elevated drawdown: {drawdown:.1%}")
        elif drawdown < 0.02:
            score += 0.15
            reasons.append("healthy account (minimal drawdown)")

        # --- Daily P/L check ---
        daily_pnl_pct = ctx.daily_pnl / balance if balance > 0 else 0
        if daily_pnl_pct < -self.MAX_DAILY_LOSS_PCT:
            score -= 0.4
            reasons.append(f"daily loss {daily_pnl_pct:.1%} exceeds {self.MAX_DAILY_LOSS_PCT:.0%} cap")
            flags.append(VETO_CRITICAL_DRAWDOWN)
        elif daily_pnl_pct < -self.MAX_DAILY_LOSS_PCT * 0.5:
            score -= 0.15
            reasons.append(f"daily loss {daily_pnl_pct:.1%} nearing cap")
        elif daily_pnl_pct > 0.01:
            score += 0.1
            reasons.append(f"positive day ({daily_pnl_pct:.1%})")

        # --- Weekly P/L check ---
        weekly_pnl_pct = ctx.weekly_pnl / balance if balance > 0 else 0
        if weekly_pnl_pct < -self.MAX_WEEKLY_LOSS_PCT:
            score -= 0.3
            reasons.append(f"weekly loss {weekly_pnl_pct:.1%} exceeds cap")
            flags.append(VETO_CRITICAL_DRAWDOWN)
        elif weekly_pnl_pct < -self.MAX_WEEKLY_LOSS_PCT * 0.5:
            score -= 0.15
            reasons.append(f"weekly loss {weekly_pnl_pct:.1%}")

        # --- Consecutive losses ---
        if ctx.recent_trades:
            consecutive_losses = 0
            for trade in reversed(ctx.recent_trades):
                pnl = trade.get("pnl", 0) if isinstance(trade, dict) else getattr(trade, "pnl", 0)
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    break

            if consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
                score -= 0.4
                reasons.append(f"{consecutive_losses} consecutive losses — reduce size")
            elif consecutive_losses >= 2:
                score -= 0.15
                reasons.append(f"{consecutive_losses} consecutive losses")
            elif consecutive_losses == 0 and len(ctx.recent_trades) >= 3:
                score += 0.1
                reasons.append("recent wins — confidence high")

        # --- Open position count ---
        if len(ctx.open_positions) >= 3:
            score -= 0.2
            reasons.append(f"{len(ctx.open_positions)} positions open (concentration risk)")
        elif len(ctx.open_positions) == 0:
            score += 0.1
            reasons.append("no open positions (fresh slate)")

        # If no issues found, give a moderate positive
        if score >= 0 and not flags:
            score = max(score, 0.2)
            if not reasons:
                reasons.append("account state healthy")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "OK"

        return self._verdict(vote, conf, reason, flags)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 15: Correlation Agent
# ═══════════════════════════════════════════════════════════════════════════

class CorrelationAgent(TradingAgent):
    """Checks open positions for correlated exposure. Prevents
    over-concentration in same currency direction."""

    name = "Correlation"
    specialty = "correlation"
    default_weight = 1.0

    # Correlation groups: pairs that move together
    CORRELATION_GROUPS = {
        "EUR_LONG":  ["EURUSD_BUY", "EURJPY_BUY", "GBPUSD_BUY"],
        "EUR_SHORT": ["EURUSD_SELL", "EURJPY_SELL", "GBPUSD_SELL"],
        "JPY_WEAK":  ["USDJPY_BUY", "EURJPY_BUY", "GBPJPY_BUY"],
        "JPY_STRONG":["USDJPY_SELL", "EURJPY_SELL", "GBPJPY_SELL"],
        "USD_WEAK":  ["EURUSD_BUY", "GBPUSD_BUY", "AUDUSD_BUY", "NZDUSD_BUY"],
        "USD_STRONG": ["EURUSD_SELL", "GBPUSD_SELL", "USDCAD_BUY", "USDJPY_BUY"],
        "COMMODITY":  ["AUDUSD_BUY", "NZDUSD_BUY"],
    }

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        symbol = ctx.symbol.upper().replace(".", "").replace("#", "")
        direction = ctx.direction
        proposed = f"{symbol}_{direction}"

        score = 0.0
        reasons = []
        flags = []

        if not ctx.open_positions:
            return self._verdict(
                BUY if direction == "BUY" else SELL,
                0.5,
                "No open positions — no correlation risk"
            )

        # Build list of current open position keys
        open_keys = []
        for pos in ctx.open_positions:
            if isinstance(pos, dict):
                sym = pos.get("symbol", "").upper().replace(".", "").replace("#", "")
                d = "BUY" if pos.get("type", 0) == 0 else "SELL"
            else:
                sym = getattr(pos, "symbol", "").upper().replace(".", "").replace("#", "")
                d = "BUY" if getattr(pos, "type", 0) == 0 else "SELL"
            if sym:
                open_keys.append(f"{sym}_{d}")

        # Check if proposed trade would create correlated exposure
        correlated_count = 0
        correlated_groups = []

        for group_name, members in self.CORRELATION_GROUPS.items():
            if proposed in members:
                # Count how many open positions are in the same group
                same_group = [k for k in open_keys if k in members]
                if same_group:
                    correlated_count += len(same_group)
                    correlated_groups.append(f"{group_name}({len(same_group)+1})")

        # Same symbol already open
        same_symbol = [k for k in open_keys if k.startswith(symbol)]
        if same_symbol:
            score -= 0.5
            reasons.append(f"{symbol} already has open position")
            flags.append("duplicate_symbol")

        # Correlated exposure
        if correlated_count >= 3:
            score -= 0.6
            reasons.append(f"heavy correlated exposure: {', '.join(correlated_groups)}")
            flags.append("over_correlated")
        elif correlated_count >= 2:
            score -= 0.3
            reasons.append(f"moderate correlation: {', '.join(correlated_groups)}")
        elif correlated_count == 1:
            score -= 0.1
            reasons.append(f"light correlation: {', '.join(correlated_groups)}")
        elif correlated_count == 0 and not same_symbol:
            score += 0.2
            reasons.append("diversified — no correlation risk")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "low correlation risk"

        return self._verdict(vote, conf, reason, flags)
