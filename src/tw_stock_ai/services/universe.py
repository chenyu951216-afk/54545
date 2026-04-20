from __future__ import annotations

from statistics import mean

from tw_stock_ai.config import Settings
from tw_stock_ai.models import PriceBar
from tw_stock_ai.services.short_term_types import IndicatorBundle, UniverseDecision


def apply_universe_filter(
    bars: list[PriceBar],
    indicators: IndicatorBundle | None,
    settings: Settings,
) -> UniverseDecision:
    ordered = sorted(bars, key=lambda item: item.trade_date)
    if len(ordered) < settings.min_price_bars_for_screening:
        return UniverseDecision(
            eligible=False,
            reasons=["insufficient_price_history"],
            metrics={"bar_count": len(ordered), "required": settings.min_price_bars_for_screening},
        )
    if indicators is None:
        return UniverseDecision(eligible=False, reasons=["indicator_bundle_unavailable"], metrics={})

    close = indicators.latest_close
    reasons: list[str] = []
    recent_closes = [float(item.close) for item in ordered[-20:]]
    recent_range_ratio = ((max(recent_closes) - min(recent_closes)) / close) if close > 0 else 0.0

    if close < settings.universe_min_close_price:
        reasons.append("price_below_minimum")
    if close > settings.universe_max_close_price:
        reasons.append("price_above_maximum")
    if (indicators.average_volume_20d or 0) < settings.universe_min_average_volume_20d:
        reasons.append("insufficient_liquidity_volume")
    if (indicators.average_turnover_20d or 0) < settings.universe_min_average_turnover_20d:
        reasons.append("insufficient_liquidity_turnover")
    if recent_range_ratio <= settings.universe_exclude_stagnant_range_ratio_20d:
        reasons.append("stagnant_price_action")

    return UniverseDecision(
        eligible=not reasons,
        reasons=reasons,
        metrics={
            "latest_close": close,
            "average_volume_20d": indicators.average_volume_20d,
            "average_turnover_20d": indicators.average_turnover_20d,
            "recent_range_ratio_20d": round(recent_range_ratio, 4),
            "average_close_20d": round(mean(recent_closes), 2),
        },
    )
