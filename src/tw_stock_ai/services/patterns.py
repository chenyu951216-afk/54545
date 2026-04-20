from __future__ import annotations

from tw_stock_ai.config import Settings
from tw_stock_ai.models import PriceBar
from tw_stock_ai.services.short_term_types import IndicatorBundle, PatternDecision


def detect_patterns(
    bars: list[PriceBar],
    indicators: IndicatorBundle | None,
    settings: Settings,
) -> PatternDecision:
    if indicators is None or len(bars) < settings.breakout_lookback_days + 1:
        return PatternDecision(
            breakout_detected=False,
            consolidation_detected=False,
            pattern_label="unavailable",
            confidence=0.0,
            metrics={"reason": "insufficient_data"},
        )

    ordered = sorted(bars, key=lambda item: item.trade_date)
    prior_high = max(float(item.high) for item in ordered[-settings.breakout_lookback_days - 1 : -1])
    latest_close = indicators.latest_close
    breakout_level = prior_high * (1 + settings.breakout_buffer_pct)
    breakout_detected = latest_close > breakout_level
    consolidation_detected = (
        indicators.consolidation_range_ratio is not None
        and indicators.consolidation_range_ratio <= settings.consolidation_max_range_ratio
    )
    confidence = 0.0
    if breakout_detected:
        confidence += 55.0
    if consolidation_detected:
        confidence += 30.0
    if (indicators.volume_ratio or 0.0) >= settings.volume_surge_ratio_threshold:
        confidence += 15.0

    if breakout_detected and consolidation_detected:
        label = "consolidation_breakout"
    elif breakout_detected:
        label = "breakout"
    elif consolidation_detected:
        label = "consolidation"
    else:
        label = "none"

    return PatternDecision(
        breakout_detected=breakout_detected,
        consolidation_detected=consolidation_detected,
        pattern_label=label,
        confidence=min(confidence, 100.0),
        metrics={
            "prior_high": round(prior_high, 2),
            "breakout_level": round(breakout_level, 2),
            "consolidation_range_ratio": indicators.consolidation_range_ratio,
            "volume_ratio": indicators.volume_ratio,
        },
    )
