"""
Candle Pattern Agents (4 agents)
=================================
1. SingleCandleAgent      — individual candle anatomy
2. MultiBarFormationAgent — 2-3 bar patterns
3. CandleMomentumAgent   — body progression & momentum
4. CandleAnatomyAgent    — wick/close structural analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from candle_intelligence.agent_base import (
    TradingAgent, TradeContext, AgentVerdict,
    STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _body(o, c):
    return c - o

def _body_ratio(o, h, l, c):
    rng = h - l
    if rng < 1e-12:
        return 0.0
    return (c - o) / rng

def _upper_wick(o, h, c):
    return h - max(o, c)

def _lower_wick(o, l, c):
    return min(o, c) - l

def _wick_ratio(wick, rng):
    if rng < 1e-12:
        return 0.0
    return wick / rng


# ═══════════════════════════════════════════════════════════════════════════
# Agent 1: Single Candle Agent
# ═══════════════════════════════════════════════════════════════════════════

class SingleCandleAgent(TradingAgent):
    """Analyzes the signal bar's candle shape: hammer, shooting star,
    doji, marubozu, spinning top. Reports if the candle supports the
    trade direction."""

    name = "Single Candle"
    specialty = "candle_anatomy"
    default_weight = 1.0

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        df = ctx.h1_bars
        if df is None or len(df) < 3:
            return self._verdict(NEUTRAL, 0.0, "No H1 data")

        bar = df.iloc[-1]
        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        atr = ctx.atr_value if ctx.atr_value > 0 else (h - l)

        rng = h - l
        body = abs(c - o)
        br = _body_ratio(o, h, l, c)
        uw = _wick_ratio(_upper_wick(o, h, c), rng)
        lw = _wick_ratio(_lower_wick(o, l, c), rng)
        body_atr = body / atr if atr > 1e-12 else 0.0

        # Detect candle types
        is_doji = abs(br) < 0.1 and rng > 0
        is_hammer = lw > 0.60 and uw < 0.15 and body_atr < 0.4
        is_shooting_star = uw > 0.60 and lw < 0.15 and body_atr < 0.4
        is_marubozu = abs(br) > 0.85 and body_atr > 0.6
        is_spinning_top = abs(br) < 0.25 and uw > 0.3 and lw > 0.3

        score = 0.0
        reasons = []

        if is_doji:
            score -= 0.3
            reasons.append("doji (indecision)")
        elif is_spinning_top:
            score -= 0.2
            reasons.append("spinning top (weak)")

        if is_marubozu:
            if (ctx.direction == "BUY" and c > o) or (ctx.direction == "SELL" and c < o):
                score += 0.7
                reasons.append("marubozu confirms direction")
            else:
                score -= 0.5
                reasons.append("marubozu opposes direction")

        if is_hammer:
            if ctx.direction == "BUY":
                score += 0.5
                reasons.append("hammer (bullish reversal)")
            else:
                score -= 0.3
                reasons.append("hammer opposes SELL")

        if is_shooting_star:
            if ctx.direction == "SELL":
                score += 0.5
                reasons.append("shooting star (bearish reversal)")
            else:
                score -= 0.3
                reasons.append("shooting star opposes BUY")

        # Close position: for BUY, close near high is strong
        close_pos = (c - l) / rng if rng > 1e-12 else 0.5
        if ctx.direction == "BUY" and close_pos > 0.75:
            score += 0.2
            reasons.append("close near high")
        elif ctx.direction == "SELL" and close_pos < 0.25:
            score += 0.2
            reasons.append("close near low")
        elif ctx.direction == "BUY" and close_pos < 0.25:
            score -= 0.2
            reasons.append("close near low (weak for BUY)")
        elif ctx.direction == "SELL" and close_pos > 0.75:
            score -= 0.2
            reasons.append("close near high (weak for SELL)")

        # Strong body in trade direction
        if body_atr > 0.5 and not is_marubozu:
            if (ctx.direction == "BUY" and c > o) or (ctx.direction == "SELL" and c < o):
                score += 0.15
                reasons.append("strong body")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "neutral candle"

        return self._verdict(vote, conf, reason)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 2: Multi-Bar Formation Agent
# ═══════════════════════════════════════════════════════════════════════════

class MultiBarFormationAgent(TradingAgent):
    """Detects 2-3 bar patterns: engulfing, harami, morning/evening star,
    three soldiers/crows, tweezers, three-bar reversals."""

    name = "Multi-Bar Formation"
    specialty = "candle_patterns"
    default_weight = 0.7

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        df = ctx.h1_bars
        if df is None or len(df) < 5:
            return self._verdict(NEUTRAL, 0.0, "Insufficient H1 bars")

        bars = df.tail(5)
        o = bars["open"].values.astype(float)
        h = bars["high"].values.astype(float)
        l = bars["low"].values.astype(float)
        c = bars["close"].values.astype(float)
        atr = ctx.atr_value if ctx.atr_value > 1e-12 else 1.0

        score = 0.0
        reasons = []

        # --- Engulfing (last 2 bars) ---
        eng = self._detect_engulfing(o, h, l, c, atr)
        if eng != 0:
            if (eng > 0 and ctx.direction == "BUY") or (eng < 0 and ctx.direction == "SELL"):
                score += 0.6
                reasons.append("engulfing confirms")
            else:
                score -= 0.4
                reasons.append("engulfing opposes")

        # --- Harami (last 2 bars) ---
        har = self._detect_harami(o, c, atr)
        if har != 0:
            if (har > 0 and ctx.direction == "BUY") or (har < 0 and ctx.direction == "SELL"):
                score += 0.3
                reasons.append("harami confirms")
            else:
                score -= 0.2
                reasons.append("harami opposes")

        # --- Morning/Evening Star (last 3 bars) ---
        star = self._detect_star(o, h, l, c, atr)
        if star != 0:
            if (star > 0 and ctx.direction == "BUY") or (star < 0 and ctx.direction == "SELL"):
                score += 0.7
                reasons.append("star pattern confirms")
            else:
                score -= 0.5
                reasons.append("star pattern opposes")

        # --- Three Soldiers / Crows (last 3 bars) ---
        three = self._detect_three_consecutive(o, c, atr)
        if three != 0:
            if (three > 0 and ctx.direction == "BUY") or (three < 0 and ctx.direction == "SELL"):
                score += 0.5
                reasons.append("three soldiers/crows confirms")
            else:
                score -= 0.4
                reasons.append("three soldiers/crows opposes")

        # --- Tweezer (last 2 bars) ---
        tw = self._detect_tweezer(h, l, atr)
        if tw != 0:
            if (tw > 0 and ctx.direction == "BUY") or (tw < 0 and ctx.direction == "SELL"):
                score += 0.4
                reasons.append("tweezer confirms")
            else:
                score -= 0.3
                reasons.append("tweezer opposes")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "no multi-bar pattern"

        return self._verdict(vote, conf, reason)

    @staticmethod
    def _detect_engulfing(o, h, l, c, atr):
        # Check last 2 bars (idx -2 and -1)
        prev_body = c[-2] - o[-2]
        curr_body = c[-1] - o[-1]
        prev_abs = abs(prev_body)
        curr_abs = abs(curr_body)
        # Engulfing: current body fully covers previous body, opposite direction
        if curr_abs > prev_abs * 1.2 and curr_abs > atr * 0.3:
            if prev_body < 0 and curr_body > 0:  # bullish engulfing
                if c[-1] > o[-2] and o[-1] < c[-2]:
                    return +1
            elif prev_body > 0 and curr_body < 0:  # bearish engulfing
                if c[-1] < o[-2] and o[-1] > c[-2]:
                    return -1
        return 0

    @staticmethod
    def _detect_harami(o, c, atr):
        prev_body = c[-2] - o[-2]
        curr_body = c[-1] - o[-1]
        prev_abs = abs(prev_body)
        curr_abs = abs(curr_body)
        # Harami: small current body inside previous large body
        if prev_abs > atr * 0.4 and curr_abs < prev_abs * 0.5:
            curr_mid = (o[-1] + c[-1]) / 2
            if min(o[-2], c[-2]) < curr_mid < max(o[-2], c[-2]):
                if prev_body < 0 and curr_body > 0:
                    return +1  # bullish harami
                elif prev_body > 0 and curr_body < 0:
                    return -1  # bearish harami
        return 0

    @staticmethod
    def _detect_star(o, h, l, c, atr):
        # 3-bar pattern: large body, small body (gap), large opposite body
        if len(o) < 3:
            return 0
        body1 = c[-3] - o[-3]
        body2_abs = abs(c[-2] - o[-2])
        body3 = c[-1] - o[-1]
        body1_abs = abs(body1)
        body3_abs = abs(body3)

        # Star: first bar large, middle bar small, third bar large opposite
        if body1_abs > atr * 0.4 and body2_abs < atr * 0.15 and body3_abs > atr * 0.3:
            if body1 < 0 and body3 > 0:  # morning star (bullish)
                return +1
            elif body1 > 0 and body3 < 0:  # evening star (bearish)
                return -1
        return 0

    @staticmethod
    def _detect_three_consecutive(o, c, atr):
        if len(o) < 3:
            return 0
        bodies = [c[-3] - o[-3], c[-2] - o[-2], c[-1] - o[-1]]
        # Three white soldiers
        if all(b > atr * 0.15 for b in bodies):
            return +1
        # Three black crows
        if all(b < -atr * 0.15 for b in bodies):
            return -1
        return 0

    @staticmethod
    def _detect_tweezer(h, l, atr):
        # Tweezer top: two bars with nearly equal highs
        if abs(h[-1] - h[-2]) < atr * 0.05:
            return -1  # bearish (double top at same level)
        # Tweezer bottom: two bars with nearly equal lows
        if abs(l[-1] - l[-2]) < atr * 0.05:
            return +1  # bullish (double bottom at same level)
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Agent 3: Candle Momentum Agent
# ═══════════════════════════════════════════════════════════════════════════

class CandleMomentumAgent(TradingAgent):
    """Measures candle body progression: acceleration vs exhaustion,
    consecutive direction count, body ratio trend."""

    name = "Candle Momentum"
    specialty = "candle_momentum"
    default_weight = 1.0

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        df = ctx.h1_bars
        if df is None or len(df) < 10:
            return self._verdict(NEUTRAL, 0.0, "Insufficient data")

        bars = df.tail(10)
        o = bars["open"].values.astype(float)
        c = bars["close"].values.astype(float)
        h = bars["high"].values.astype(float)
        l = bars["low"].values.astype(float)
        atr = ctx.atr_value if ctx.atr_value > 1e-12 else 1.0

        bodies = c - o
        abs_bodies = np.abs(bodies)

        score = 0.0
        reasons = []

        # --- Consecutive direction count ---
        is_buy = ctx.direction == "BUY"
        consecutive = 0
        for i in range(len(bodies) - 1, -1, -1):
            if (is_buy and bodies[i] > 0) or (not is_buy and bodies[i] < 0):
                consecutive += 1
            else:
                break

        if consecutive >= 4:
            score += 0.4
            reasons.append(f"{consecutive} consecutive bars in direction")
        elif consecutive >= 2:
            score += 0.2
            reasons.append(f"{consecutive} bars in direction")
        elif consecutive == 0:
            # Last bar opposed the trade direction
            score -= 0.2
            reasons.append("last bar opposes direction")

        # --- Body acceleration (last 3 bars getting bigger?) ---
        if len(abs_bodies) >= 3:
            recent_3 = abs_bodies[-3:]
            if recent_3[-1] > recent_3[-2] > recent_3[-3]:
                score += 0.3
                reasons.append("body acceleration (expanding)")
            elif recent_3[-1] < recent_3[-2] < recent_3[-3]:
                score -= 0.3
                reasons.append("body deceleration (shrinking)")

        # --- Body ratio trend (linear regression of body ratios) ---
        if len(abs_bodies) >= 5:
            body_norm = abs_bodies[-5:] / atr
            x = np.arange(5, dtype=float)
            if np.std(body_norm) > 0:
                slope = np.polyfit(x, body_norm, 1)[0]
                if slope > 0.05:
                    score += 0.15
                    reasons.append("rising body trend")
                elif slope < -0.05:
                    score -= 0.15
                    reasons.append("falling body trend")

        # --- Average body strength ---
        avg_body_atr = np.mean(abs_bodies[-3:]) / atr
        if avg_body_atr > 0.6:
            score += 0.15
            reasons.append(f"strong avg body ({avg_body_atr:.1f}x ATR)")
        elif avg_body_atr < 0.2:
            score -= 0.15
            reasons.append(f"weak avg body ({avg_body_atr:.1f}x ATR)")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "neutral momentum"

        return self._verdict(vote, conf, reason)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 4: Candle Anatomy Agent
# ═══════════════════════════════════════════════════════════════════════════

class CandleAnatomyAgent(TradingAgent):
    """Studies wick-to-body ratios, close position, gap analysis,
    rejection wicks, and absorption patterns."""

    name = "Candle Anatomy"
    specialty = "candle_anatomy"
    default_weight = 1.0

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        df = ctx.h1_bars
        if df is None or len(df) < 5:
            return self._verdict(NEUTRAL, 0.0, "No data")

        bars = df.tail(5)
        o = bars["open"].values.astype(float)
        h = bars["high"].values.astype(float)
        l = bars["low"].values.astype(float)
        c = bars["close"].values.astype(float)
        atr = ctx.atr_value if ctx.atr_value > 1e-12 else 1.0

        score = 0.0
        reasons = []
        is_buy = ctx.direction == "BUY"

        # --- Current bar analysis ---
        cur_rng = h[-1] - l[-1]
        cur_body = abs(c[-1] - o[-1])
        cur_uw = _upper_wick(o[-1], h[-1], c[-1])
        cur_lw = _lower_wick(o[-1], l[-1], c[-1])

        if cur_rng > 1e-12:
            uw_ratio = cur_uw / cur_rng
            lw_ratio = cur_lw / cur_rng

            # Rejection wick detection (long wick opposing trade direction)
            if is_buy and lw_ratio > 0.5 and uw_ratio < 0.15:
                score += 0.4
                reasons.append("lower rejection wick (bullish)")
            elif not is_buy and uw_ratio > 0.5 and lw_ratio < 0.15:
                score += 0.4
                reasons.append("upper rejection wick (bearish)")
            elif is_buy and uw_ratio > 0.5 and lw_ratio < 0.15:
                score -= 0.3
                reasons.append("upper rejection wick opposes BUY")
            elif not is_buy and lw_ratio > 0.5 and uw_ratio < 0.15:
                score -= 0.3
                reasons.append("lower rejection wick opposes SELL")

            # Wick-to-body ratio (high wick ratio = indecision)
            if cur_body > 0:
                total_wick = cur_uw + cur_lw
                wb_ratio = total_wick / cur_body
                if wb_ratio > 3.0:
                    score -= 0.2
                    reasons.append(f"high wick-to-body ({wb_ratio:.1f}x)")
                elif wb_ratio < 0.5:
                    score += 0.15
                    reasons.append("decisive close (low wicks)")

        # --- Gap analysis ---
        if len(c) >= 2:
            gap = (o[-1] - c[-2]) / atr
            if abs(gap) > 0.1:
                if (is_buy and gap > 0) or (not is_buy and gap < 0):
                    score += 0.25
                    reasons.append(f"gap in direction ({gap:.2f} ATR)")
                else:
                    score -= 0.15
                    reasons.append(f"gap against direction ({gap:.2f} ATR)")

        # --- Absorption detection (current bar absorbs previous move) ---
        if len(c) >= 2:
            prev_body = c[-2] - o[-2]
            curr_body = c[-1] - o[-1]
            if abs(prev_body) > atr * 0.3:
                # Current bar reverses and covers more than 50% of previous
                if prev_body > 0 and curr_body < 0 and abs(curr_body) > abs(prev_body) * 0.5:
                    if not is_buy:
                        score += 0.3
                        reasons.append("bearish absorption")
                    else:
                        score -= 0.3
                        reasons.append("bearish absorption opposes BUY")
                elif prev_body < 0 and curr_body > 0 and abs(curr_body) > abs(prev_body) * 0.5:
                    if is_buy:
                        score += 0.3
                        reasons.append("bullish absorption")
                    else:
                        score -= 0.3
                        reasons.append("bullish absorption opposes SELL")

        # --- Close position in overall recent range ---
        recent_high = np.max(h[-5:])
        recent_low = np.min(l[-5:])
        recent_rng = recent_high - recent_low
        if recent_rng > 1e-12:
            pos_in_range = (c[-1] - recent_low) / recent_rng
            if is_buy and pos_in_range > 0.8:
                score += 0.15
                reasons.append("close near 5-bar high")
            elif not is_buy and pos_in_range < 0.2:
                score += 0.15
                reasons.append("close near 5-bar low")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "neutral anatomy"

        return self._verdict(vote, conf, reason)
