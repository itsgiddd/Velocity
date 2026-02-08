from typing import Any

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - allows offline unit tests
    mt5 = None


class AdaptiveRiskManager:
    """
    Position sizing based on account size, confidence, and stop distance.
    """

    def _get_attr(self, obj: Any, name: str, default: float = 0.0) -> float:
        try:
            return float(getattr(obj, name))
        except Exception:
            return default

    def _normalize_confidence(self, confidence_score: float) -> float:
        # Supports both 0..1 confidence and integer confluence scores.
        if confidence_score > 1.0:
            return min(max(confidence_score / 8.0, 0.0), 1.0)
        return min(max(confidence_score, 0.0), 1.0)

    def calculate_lot_size(
        self,
        symbol: str,
        price: float,
        sl_price: float,
        confidence_score: float,
        account_info,
        symbol_info,
    ) -> float:
        balance = self._get_attr(account_info, "balance", 0.0)
        if balance <= 0:
            return 0.0

        point = self._get_attr(symbol_info, "point", 0.0)
        if point <= 0:
            return 0.0

        # Base risk curve by balance.
        risk_pct = 0.02
        if balance < 1000:
            risk_pct = 0.04
        elif balance < 5000:
            risk_pct = 0.03

        conf = self._normalize_confidence(confidence_score)
        if conf < 0.35:
            return 0.0

        # Scale risk by confidence (35% -> 25% risk, 100% -> 100% risk).
        risk_mult = 0.25 + 0.75 * conf
        risk_money = balance * risk_pct * risk_mult

        sl_points = abs(price - sl_price) / point
        if sl_points <= 0:
            return 0.0

        tick_value = self._get_attr(symbol_info, "trade_tick_value", 0.0)
        if tick_value <= 0:
            tick_value = 1.0

        loss_per_lot = sl_points * tick_value
        if loss_per_lot <= 0:
            return 0.0

        raw_lot = risk_money / loss_per_lot

        step = self._get_attr(symbol_info, "volume_step", 0.01)
        min_vol = self._get_attr(symbol_info, "volume_min", 0.01)
        max_vol = self._get_attr(symbol_info, "volume_max", 100.0)

        if step <= 0:
            step = 0.01

        lot = round(raw_lot / step) * step
        lot = max(min_vol, min(lot, max_vol))

        free_margin = self._get_attr(account_info, "free_margin", 0.0)
        if free_margin <= 0:
            free_margin = self._get_attr(account_info, "margin_free", 0.0)

        if mt5 is not None and free_margin > 0 and mt5.initialize():
            try:
                margin_needed = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lot, price)
                if margin_needed and margin_needed > free_margin * 0.95:
                    lot *= (free_margin * 0.95 / margin_needed)
                    lot = round(lot / step) * step
            except Exception:
                pass

        lot = max(min_vol, min(lot, max_vol))
        return float(round(lot, 2))
