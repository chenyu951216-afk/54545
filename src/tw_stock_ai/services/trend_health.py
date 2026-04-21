from __future__ import annotations

from tw_stock_ai.config import Settings
from tw_stock_ai.models import PriceBar
from tw_stock_ai.services.indicators import calculate_indicator_bundle


def analyze_trend_health(bars: list[PriceBar], settings: Settings) -> dict:
    indicators = calculate_indicator_bundle(bars, settings)
    ordered = sorted(bars, key=lambda item: item.trade_date)
    latest_close = float(ordered[-1].close) if ordered else None

    if indicators is None or latest_close is None:
        return {
            "trend_status": "unavailable",
            "exit_signal": "unavailable",
            "metrics": {"reason": "insufficient_price_bars_for_holding_analysis"},
        }

    trend_status = "neutral"
    if (
        indicators.ema20 is not None
        and indicators.ema60 is not None
        and indicators.rsi is not None
        and indicators.macd_histogram is not None
    ):
        if latest_close > indicators.ema20 > indicators.ema60 and indicators.rsi >= 55 and indicators.macd_histogram > 0:
            trend_status = "strong_uptrend"
        elif latest_close > indicators.ema20 and indicators.ema20 >= indicators.ema60:
            trend_status = "uptrend"
        elif latest_close < indicators.ema20 and indicators.ema20 < indicators.ema60:
            trend_status = "downtrend"
        elif latest_close < indicators.ema20 or indicators.rsi < settings.holding_trend_rsi_weak_threshold:
            trend_status = "weakening"

    if trend_status in {"strong_uptrend", "uptrend"}:
        exit_signal = "hold"
    elif trend_status == "weakening":
        exit_signal = "exit_watch"
    elif trend_status == "downtrend":
        exit_signal = "review_exit"
    else:
        exit_signal = "hold"

    return {
        "trend_status": trend_status,
        "exit_signal": exit_signal,
        "metrics": {
            "latest_close": latest_close,
            "ema20": indicators.ema20,
            "ema60": indicators.ema60,
            "rsi": indicators.rsi,
            "macd_histogram": indicators.macd_histogram,
            "adx": indicators.adx,
            "atr": indicators.atr,
            "average_volume_20d": indicators.average_volume_20d,
            "volume_ratio": indicators.volume_ratio,
        },
    }
