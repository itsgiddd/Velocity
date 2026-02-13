"""
Trend & Structure Agents (4 agents)
====================================
5.  MultiTFTrendAgent      — M15/H1/H4 ZP alignment
6.  MovingAverageAgent     — SMA5/20/50 structure
7.  SupportResistanceAgent — swing-based S/R proximity
8.  PushStructureAgent     — push exhaustion analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from candle_intelligence.agent_base import (
    TradingAgent, TradeContext, AgentVerdict,
    STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL,
)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 5: Multi-TF Trend Agent
# ═══════════════════════════════════════════════════════════════════════════

class MultiTFTrendAgent(TradingAgent):
    """Checks ZeroPoint position direction across M15/H1/H4.
    Reports how many timeframes agree with the proposed trade."""

    name = "Multi-TF Trend"
    specialty = "trend_alignment"
    default_weight = 1.0

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        is_buy = ctx.direction == "BUY"
        agreements = 0
        total = 0
        tf_status = []

        for tf_name, zp_df in [("M15", ctx.zp_m15), ("H1", ctx.zp_h1), ("H4", ctx.zp_h4)]:
            if zp_df is None or len(zp_df) < 2:
                continue
            total += 1
            pos_val = int(zp_df.iloc[-1].get("pos", 0))
            if pos_val == 0:
                tf_status.append(f"{tf_name}=FLAT")
                continue

            tf_is_buy = pos_val == 1
            if tf_is_buy == is_buy:
                agreements += 1
                tf_status.append(f"{tf_name}={'BUY' if tf_is_buy else 'SELL'} OK")
            else:
                tf_status.append(f"{tf_name}={'BUY' if tf_is_buy else 'SELL'} AGAINST")

        if total == 0:
            return self._verdict(NEUTRAL, 0.0, "No ZP data available")

        ratio = agreements / total
        if ratio >= 1.0:
            vote = STRONG_BUY if is_buy else STRONG_SELL
            conf = 0.9
        elif ratio >= 0.67:
            vote = BUY if is_buy else SELL
            conf = 0.7
        elif ratio >= 0.34:
            vote = NEUTRAL
            conf = 0.5
        else:
            vote = SELL if is_buy else BUY
            conf = 0.8

        reason = f"{agreements}/{total} TFs agree | " + ", ".join(tf_status)
        return self._verdict(vote, conf, reason)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 6: Moving Average Agent
# ═══════════════════════════════════════════════════════════════════════════

class MovingAverageAgent(TradingAgent):
    """Analyzes SMA5/20/50 structure: price vs MAs, MA ordering,
    slope direction, golden/death cross proximity."""

    name = "Moving Average"
    specialty = "ma_structure"
    default_weight = 1.0

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        df = ctx.h1_bars
        if df is None or len(df) < 52:
            return self._verdict(NEUTRAL, 0.0, "Need 52+ H1 bars for SMA50")

        close = df["close"].values.astype(float)
        sma5 = np.mean(close[-5:])
        sma20 = np.mean(close[-20:])
        sma50 = np.mean(close[-50:])
        price = close[-1]
        atr = ctx.atr_value if ctx.atr_value > 1e-12 else 1.0

        score = 0.0
        reasons = []
        is_buy = ctx.direction == "BUY"

        # --- Price vs MAs ---
        above_5 = price > sma5
        above_20 = price > sma20
        above_50 = price > sma50

        if is_buy:
            if above_5 and above_20 and above_50:
                score += 0.4
                reasons.append("price above all MAs")
            elif above_5 and above_20:
                score += 0.2
                reasons.append("price above SMA5+20")
            elif not above_20 and not above_50:
                score -= 0.3
                reasons.append("price below SMA20+50")
        else:
            if not above_5 and not above_20 and not above_50:
                score += 0.4
                reasons.append("price below all MAs")
            elif not above_5 and not above_20:
                score += 0.2
                reasons.append("price below SMA5+20")
            elif above_20 and above_50:
                score -= 0.3
                reasons.append("price above SMA20+50")

        # --- MA fan (bullish: SMA5 > SMA20 > SMA50; bearish: reverse) ---
        if sma5 > sma20 > sma50:
            if is_buy:
                score += 0.3
                reasons.append("bullish MA fan")
            else:
                score -= 0.25
                reasons.append("bullish MA fan opposes SELL")
        elif sma5 < sma20 < sma50:
            if not is_buy:
                score += 0.3
                reasons.append("bearish MA fan")
            else:
                score -= 0.25
                reasons.append("bearish MA fan opposes BUY")

        # --- SMA20 slope (using last 5 bars of SMA20) ---
        if len(close) >= 25:
            sma20_recent = np.mean(close[-20:])
            sma20_5ago = np.mean(close[-25:-5])
            slope = (sma20_recent - sma20_5ago) / atr
            if is_buy and slope > 0.3:
                score += 0.15
                reasons.append("SMA20 rising")
            elif not is_buy and slope < -0.3:
                score += 0.15
                reasons.append("SMA20 falling")
            elif is_buy and slope < -0.3:
                score -= 0.15
                reasons.append("SMA20 falling (opposes BUY)")
            elif not is_buy and slope > 0.3:
                score -= 0.15
                reasons.append("SMA20 rising (opposes SELL)")

        # --- Golden/Death cross proximity ---
        sma5_prev = np.mean(close[-6:-1])
        sma20_prev = np.mean(close[-21:-1])
        cross_now = sma5 - sma20
        cross_prev = sma5_prev - sma20_prev
        if cross_now > 0 and cross_prev <= 0:
            if is_buy:
                score += 0.25
                reasons.append("golden cross (SMA5 x SMA20)")
            else:
                score -= 0.2
        elif cross_now < 0 and cross_prev >= 0:
            if not is_buy:
                score += 0.25
                reasons.append("death cross (SMA5 x SMA20)")
            else:
                score -= 0.2

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "neutral MA structure"

        return self._verdict(vote, conf, reason)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 7: Support/Resistance Agent
# ═══════════════════════════════════════════════════════════════════════════

class SupportResistanceAgent(TradingAgent):
    """Identifies nearby S/R from swing highs/lows. Checks if entry is
    near favorable S/R and if path to TP1 is clear of opposing S/R."""

    name = "Support/Resistance"
    specialty = "sr_levels"
    default_weight = 1.3

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        df = ctx.h1_bars
        if df is None or len(df) < 30:
            return self._verdict(NEUTRAL, 0.0, "Need 30+ bars for S/R")

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        price = close[-1]
        atr = ctx.atr_value if ctx.atr_value > 1e-12 else 1.0
        is_buy = ctx.direction == "BUY"

        # Find swing highs and lows (simple: local max/min over 5-bar window)
        swing_highs = []
        swing_lows = []
        for i in range(5, len(high) - 1):
            if high[i] == max(high[i-5:i+1]):
                swing_highs.append(high[i])
            if low[i] == min(low[i-5:i+1]):
                swing_lows.append(low[i])

        if not swing_highs and not swing_lows:
            return self._verdict(NEUTRAL, 0.3, "No clear S/R levels found")

        score = 0.0
        reasons = []
        flags = []

        # --- Nearest support (below price) ---
        supports_below = [s for s in swing_lows if s < price]
        if supports_below:
            nearest_support = max(supports_below)
            dist_to_support = (price - nearest_support) / atr

            if is_buy and dist_to_support < 1.0:
                score += 0.4
                reasons.append(f"near support ({dist_to_support:.1f} ATR below)")
            elif is_buy and dist_to_support > 3.0:
                score -= 0.1
                reasons.append(f"far from support ({dist_to_support:.1f} ATR)")

        # --- Nearest resistance (above price) ---
        resistances_above = [r for r in swing_highs if r > price]
        if resistances_above:
            nearest_resistance = min(resistances_above)
            dist_to_resistance = (nearest_resistance - price) / atr

            if not is_buy and dist_to_resistance < 1.0:
                score += 0.4
                reasons.append(f"near resistance ({dist_to_resistance:.1f} ATR above)")
            elif not is_buy and dist_to_resistance > 3.0:
                score -= 0.1

            # Check if resistance blocks TP1 path for BUY
            if is_buy:
                tp1_dist = (ctx.tp1 - price) / atr
                if dist_to_resistance < tp1_dist * 0.8:
                    score -= 0.4
                    reasons.append(f"resistance at {dist_to_resistance:.1f} ATR blocks TP1")
                    flags.append("sr_blocks_tp1")
                else:
                    score += 0.15
                    reasons.append("clear path to TP1")

        # --- Check if support blocks TP1 path for SELL ---
        if not is_buy and supports_below:
            nearest_support = max(supports_below)
            dist_to_support = (price - nearest_support) / atr
            tp1_dist = (price - ctx.tp1) / atr
            if dist_to_support < tp1_dist * 0.8:
                score -= 0.4
                reasons.append(f"support at {dist_to_support:.1f} ATR blocks TP1")
                flags.append("sr_blocks_tp1")
            else:
                score += 0.15
                reasons.append("clear path to TP1")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "neutral S/R"

        return self._verdict(vote, conf, reason, flags)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 8: Push Structure Agent
# ═══════════════════════════════════════════════════════════════════════════

class PushStructureAgent(TradingAgent):
    """Wraps existing push_structure_analyzer. Reports push count,
    exhaustion proximity, and reversal probability."""

    name = "Push Structure"
    specialty = "push_structure"
    default_weight = 1.0

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        # Try to use existing push profiles from model checkpoint
        profiles = ctx.push_profiles
        symbol = ctx.symbol.upper().replace(".", "").replace("#", "")

        if profiles is None or symbol not in profiles:
            return self._verdict(NEUTRAL, 0.3, "No push profile available")

        profile = profiles[symbol]
        exhaustion_count = profile.get("exhaustion_push_count", 4)
        reversal_probs = profile.get("reversal_prob_by_push", {})

        # We need to estimate current push count from price action
        df = ctx.h1_bars
        if df is None or len(df) < 20:
            return self._verdict(NEUTRAL, 0.3, "Insufficient data for push analysis")

        close = df["close"].values.astype(float)
        is_buy = ctx.direction == "BUY"

        # Simple push count estimate: count consecutive higher closes (BUY)
        # or lower closes (SELL)
        push_count = 0
        for i in range(len(close) - 1, 0, -1):
            if is_buy and close[i] > close[i-1]:
                push_count += 1
            elif not is_buy and close[i] < close[i-1]:
                push_count += 1
            else:
                break

        # Get reversal probability for this push count
        rev_prob = reversal_probs.get(str(push_count), reversal_probs.get(push_count, 0.0))
        if isinstance(rev_prob, str):
            rev_prob = float(rev_prob)

        exhaustion_ratio = push_count / exhaustion_count if exhaustion_count > 0 else 0

        score = 0.0
        reasons = []
        flags = []

        if rev_prob >= 0.60:
            score -= 0.5
            reasons.append(f"push {push_count}: {rev_prob:.0%} reversal probability (EXHAUSTED)")
            flags.append("push_exhausted")
        elif rev_prob >= 0.40:
            score -= 0.2
            reasons.append(f"push {push_count}: {rev_prob:.0%} reversal risk")
        elif push_count >= 2 and exhaustion_ratio < 0.6:
            score += 0.3
            reasons.append(f"push {push_count}/{exhaustion_count}: room to run")
        elif push_count <= 1:
            score += 0.1
            reasons.append(f"early push ({push_count})")

        # At exhaustion
        if push_count >= exhaustion_count:
            score -= 0.6
            reasons.append(f"AT EXHAUSTION ({push_count} >= {exhaustion_count})")
            flags.append("push_exhausted")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else f"push {push_count}"

        return self._verdict(vote, conf, reason, flags)
