"""
Volume & Flow Agents (3 agents)
================================
9.  VolumeProfileAgent   — volume vs average, up/down candle volume
10. VolumeMomentumAgent  — volume trend analysis
11. SpreadLiquidityAgent — spread/liquidity check
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from candle_intelligence.agent_base import (
    TradingAgent, TradeContext, AgentVerdict,
    STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL,
)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 9: Volume Profile Agent
# ═══════════════════════════════════════════════════════════════════════════

class VolumeProfileAgent(TradingAgent):
    """Analyzes tick volume: current vs SMA20, volume on up/down candles,
    climax detection, dry-up detection."""

    name = "Volume Profile"
    specialty = "volume"
    default_weight = 1.0

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        df = ctx.h1_bars
        if df is None or len(df) < 22:
            return self._verdict(NEUTRAL, 0.0, "Need 22+ bars for volume analysis")

        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        if vol_col not in df.columns:
            return self._verdict(NEUTRAL, 0.0, "No volume data")

        vol = df[vol_col].values.astype(float)
        close = df["close"].values.astype(float)
        opn = df["open"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        atr = ctx.atr_value if ctx.atr_value > 1e-12 else 1.0

        sma_vol_20 = np.mean(vol[-20:])
        if sma_vol_20 < 1:
            return self._verdict(NEUTRAL, 0.2, "Volume too low for analysis")

        cur_vol = vol[-1]
        vol_ratio = cur_vol / sma_vol_20

        score = 0.0
        reasons = []
        flags = []
        is_buy = ctx.direction == "BUY"

        # --- Volume ratio (current vs average) ---
        if vol_ratio > 2.5:
            rng = high[-1] - low[-1]
            if rng > atr * 1.0:
                # Volume climax with range expansion
                body = close[-1] - opn[-1]
                if (is_buy and body > 0) or (not is_buy and body < 0):
                    score += 0.5
                    reasons.append(f"volume climax confirms ({vol_ratio:.1f}x avg)")
                else:
                    score -= 0.3
                    reasons.append(f"volume climax opposes ({vol_ratio:.1f}x avg)")
            else:
                score += 0.1
                reasons.append(f"high volume ({vol_ratio:.1f}x avg)")
        elif vol_ratio > 1.3:
            score += 0.15
            reasons.append(f"above-avg volume ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.4:
            score -= 0.2
            reasons.append(f"volume dry-up ({vol_ratio:.1f}x avg)")
            flags.append("low_volume")
        elif vol_ratio < 0.7:
            score -= 0.1
            reasons.append(f"below-avg volume ({vol_ratio:.1f}x)")

        # --- Volume on up vs down candles (last 10 bars) ---
        last_10_vol = vol[-10:]
        last_10_body = close[-10:] - opn[-10:]
        up_vol = np.sum(last_10_vol[last_10_body > 0])
        down_vol = np.sum(last_10_vol[last_10_body < 0])
        total_vol = up_vol + down_vol

        if total_vol > 0:
            up_pct = up_vol / total_vol
            if is_buy and up_pct > 0.65:
                score += 0.3
                reasons.append(f"vol skews bullish ({up_pct:.0%} on up bars)")
            elif is_buy and up_pct < 0.35:
                score -= 0.25
                reasons.append(f"vol skews bearish ({up_pct:.0%} on up bars)")
            elif not is_buy and up_pct < 0.35:
                score += 0.3
                reasons.append(f"vol skews bearish ({1-up_pct:.0%} on down bars)")
            elif not is_buy and up_pct > 0.65:
                score -= 0.25
                reasons.append(f"vol skews bullish ({up_pct:.0%} on up bars)")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "neutral volume"

        return self._verdict(vote, conf, reason, flags)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 10: Volume Momentum Agent
# ═══════════════════════════════════════════════════════════════════════════

class VolumeMomentumAgent(TradingAgent):
    """Studies volume trends: SMA5 vs SMA20, acceleration, divergence
    between volume and price momentum."""

    name = "Volume Momentum"
    specialty = "volume_momentum"
    default_weight = 0.8

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        df = ctx.h1_bars
        if df is None or len(df) < 22:
            return self._verdict(NEUTRAL, 0.0, "Insufficient data")

        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        if vol_col not in df.columns:
            return self._verdict(NEUTRAL, 0.0, "No volume data")

        vol = df[vol_col].values.astype(float)
        close = df["close"].values.astype(float)

        sma_vol_5 = np.mean(vol[-5:])
        sma_vol_20 = np.mean(vol[-20:])

        if sma_vol_20 < 1:
            return self._verdict(NEUTRAL, 0.2, "Volume too low")

        score = 0.0
        reasons = []
        is_buy = ctx.direction == "BUY"

        # --- Volume trend (SMA5 vs SMA20) ---
        vol_trend = (sma_vol_5 - sma_vol_20) / sma_vol_20
        if vol_trend > 0.3:
            score += 0.25
            reasons.append(f"rising volume trend (+{vol_trend:.0%})")
        elif vol_trend < -0.3:
            score -= 0.2
            reasons.append(f"falling volume trend ({vol_trend:.0%})")

        # --- Volume acceleration (last 3 SMA5 values) ---
        if len(vol) >= 15:
            v5_now = np.mean(vol[-5:])
            v5_5ago = np.mean(vol[-10:-5])
            v5_10ago = np.mean(vol[-15:-10])
            if v5_now > v5_5ago > v5_10ago:
                score += 0.2
                reasons.append("volume accelerating")
            elif v5_now < v5_5ago < v5_10ago:
                score -= 0.15
                reasons.append("volume decelerating")

        # --- Volume-price divergence ---
        if len(close) >= 10:
            price_change = (close[-1] - close[-10]) / close[-10]
            vol_change = (sma_vol_5 / sma_vol_20) - 1.0

            # Price rising but volume falling = bearish divergence
            if price_change > 0.005 and vol_change < -0.2:
                if is_buy:
                    score -= 0.2
                    reasons.append("bearish vol divergence (price up, vol down)")
                else:
                    score += 0.15
            # Price falling but volume falling = bullish divergence
            elif price_change < -0.005 and vol_change < -0.2:
                if not is_buy:
                    score -= 0.2
                    reasons.append("bullish vol divergence (price down, vol down)")
                else:
                    score += 0.15

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "neutral volume momentum"

        return self._verdict(vote, conf, reason)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 11: Spread & Liquidity Agent
# ═══════════════════════════════════════════════════════════════════════════

class SpreadLiquidityAgent(TradingAgent):
    """Checks current spread vs ATR, spread vs typical. Flags thin
    liquidity and high-spread periods."""

    name = "Spread/Liquidity"
    specialty = "spread_liquidity"
    default_weight = 0.8

    def analyze(self, ctx: TradeContext) -> AgentVerdict:
        df = ctx.h1_bars
        if df is None or len(df) < 5:
            return self._verdict(NEUTRAL, 0.0, "No data")

        if "spread" not in df.columns:
            return self._verdict(NEUTRAL, 0.2, "No spread data")

        spread = df["spread"].values.astype(float)
        atr = ctx.atr_value if ctx.atr_value > 1e-12 else 1.0

        cur_spread = spread[-1]
        avg_spread = np.mean(spread[-20:]) if len(spread) >= 20 else np.mean(spread)

        score = 0.0
        reasons = []
        flags = []

        if avg_spread < 1e-12:
            return self._verdict(NEUTRAL, 0.2, "Spread data unavailable")

        spread_ratio = cur_spread / avg_spread if avg_spread > 0 else 1.0

        # Spread is in points, convert to price for ATR comparison
        # We'll use the ratio as a relative measure
        if spread_ratio > 2.5:
            score -= 0.5
            reasons.append(f"spread {spread_ratio:.1f}x normal (very thin liquidity)")
            flags.append("wide_spread")
        elif spread_ratio > 1.5:
            score -= 0.25
            reasons.append(f"spread {spread_ratio:.1f}x normal (elevated)")
            flags.append("elevated_spread")
        elif spread_ratio < 0.8:
            score += 0.15
            reasons.append(f"tight spread ({spread_ratio:.1f}x normal)")
        else:
            score += 0.05
            reasons.append("normal spread")

        # Check if spread is a significant portion of the SL distance
        sl_dist = abs(ctx.entry_price - ctx.stop_loss)
        if sl_dist > 0:
            # Estimate spread in price terms (rough: spread_points * smallest tick)
            # Since we don't have point value, use spread as percentage of SL
            # A high spread relative to SL means more cost
            spread_pct_of_sl = (cur_spread * 0.0001) / sl_dist  # rough for 5-digit
            if "JPY" in ctx.symbol.upper():
                spread_pct_of_sl = (cur_spread * 0.01) / sl_dist

            if spread_pct_of_sl > 0.15:
                score -= 0.2
                reasons.append(f"spread costs {spread_pct_of_sl:.0%} of SL distance")
                flags.append("high_spread_cost")

        score = max(-1.0, min(1.0, score))
        vote = self._direction_vote(score, ctx.direction)
        conf = min(1.0, abs(score))
        reason = "; ".join(reasons) if reasons else "normal liquidity"

        return self._verdict(vote, conf, reason, flags)
