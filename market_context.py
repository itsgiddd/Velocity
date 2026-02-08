from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class _TrendSnapshot:
    slope: float
    momentum: float


class MarketContextAnalyzer:
    """
    Lightweight market-state estimator used by both classic and enhanced brains.
    """

    def _safe_snapshot(self, df: pd.DataFrame, lookback: int) -> _TrendSnapshot:
        if df is None or len(df) < max(lookback, 5):
            return _TrendSnapshot(0.0, 0.0)

        close = df["close"].tail(lookback).astype(float).values
        x = np.arange(len(close), dtype=float)
        slope = float(np.polyfit(x, close, 1)[0]) if len(close) > 1 else 0.0
        momentum = float((close[-1] - close[0]) / (abs(close[0]) + 1e-12))
        return _TrendSnapshot(slope=slope, momentum=momentum)

    def _session_score(self) -> float:
        hour = datetime.now(timezone.utc).hour
        if 7 <= hour <= 16:
            return 0.9  # London-heavy liquidity
        if 12 <= hour <= 21:
            return 1.0  # NY overlap
        if 0 <= hour <= 6:
            return 0.6  # Asia
        return 0.5

    def get_market_state(
        self,
        symbol: str,
        h1: pd.DataFrame,
        h4: pd.DataFrame,
        d1: pd.DataFrame,
    ) -> Dict:
        h1_snap = self._safe_snapshot(h1, 30)
        h4_snap = self._safe_snapshot(h4, 30)
        d1_snap = self._safe_snapshot(d1, 30)

        trend_score = (0.5 * np.sign(h1_snap.momentum)) + (0.3 * np.sign(h4_snap.momentum)) + (0.2 * np.sign(d1_snap.momentum))
        if trend_score > 0.2:
            global_trend = 1
            sentiment = "bullish"
            trend_label = "UP"
        elif trend_score < -0.2:
            global_trend = -1
            sentiment = "bearish"
            trend_label = "DOWN"
        else:
            global_trend = 0
            sentiment = "neutral"
            trend_label = "NEUTRAL"

        if h1 is not None and len(h1) >= 30:
            returns = h1["close"].pct_change().dropna()
            vol = float(returns.tail(30).std()) if len(returns) > 0 else 0.0
        else:
            vol = 0.0

        if vol > 0.004:
            vol_label = "HIGH"
        elif vol < 0.0015:
            vol_label = "LOW"
        else:
            vol_label = "NORMAL"

        # 0..1 quality score.
        trend_strength = float(min(abs(h1_snap.momentum) * 120.0, 1.0))
        session = self._session_score()
        strength = float(max(0.0, min((trend_strength * 0.7) + (session * 0.3), 1.0)))

        support_resistance = "DEFINED" if h1 is not None and len(h1) >= 20 else "NONE"

        return {
            "trend": trend_label,
            "global_trend": global_trend,
            "sentiment": sentiment,
            "volatility": vol_label,
            "volatility_value": vol,
            "support_resistance": support_resistance,
            "session": session,
            "strength": strength,
            "symbol": symbol,
        }
