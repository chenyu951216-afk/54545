from __future__ import annotations

from tw_stock_ai.config import Settings
from tw_stock_ai.services.short_term_types import IndicatorBundle, PatternDecision, RiskPlan


def build_risk_plan(
    indicators: IndicatorBundle | None,
    pattern: PatternDecision,
    settings: Settings,
) -> RiskPlan:
    if indicators is None or indicators.atr is None or indicators.atr <= 0:
        return RiskPlan(
            entry_zone_low=None,
            entry_zone_high=None,
            stop_loss=None,
            take_profit_1=None,
            take_profit_2=None,
            risk_reward_ratio=None,
            holding_days_min=settings.short_term_holding_days_min,
            holding_days_max=settings.short_term_holding_days_max,
            risk_flags={"reasons": ["atr_unavailable"]},
            metrics={},
        )

    close = indicators.latest_close
    atr = indicators.atr
    entry_buffer = atr * settings.risk_entry_buffer_atr_multiplier
    entry_zone_low = round(close - entry_buffer, 2)
    entry_zone_high = round(close + entry_buffer, 2)

    if pattern.breakout_detected and indicators.recent_high_20d is not None:
        stop_anchor = min(entry_zone_low, indicators.recent_high_20d)
    else:
        stop_anchor = entry_zone_low
    stop_loss = round(stop_anchor - (atr * settings.risk_stop_atr_multiplier), 2)

    reference_entry = (entry_zone_low + entry_zone_high) / 2
    risk_per_share = max(reference_entry - stop_loss, 0.0)
    take_profit_1 = round(reference_entry + (risk_per_share * settings.risk_take_profit1_rr), 2)
    take_profit_2 = round(reference_entry + (risk_per_share * settings.risk_take_profit2_rr), 2)
    rr_ratio = round((take_profit_1 - reference_entry) / risk_per_share, 2) if risk_per_share > 0 else None

    reasons: list[str] = []
    if rr_ratio is None:
        reasons.append("invalid_risk_structure")
    elif rr_ratio < settings.risk_min_reward_risk_ratio:
        reasons.append("reward_risk_below_threshold")
    if indicators.atr / close > 0.12:
        reasons.append("high_volatility")
    if (indicators.adx or 0.0) < settings.adx_trend_threshold:
        reasons.append("weak_trend_strength")

    return RiskPlan(
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        risk_reward_ratio=rr_ratio,
        holding_days_min=settings.short_term_holding_days_min,
        holding_days_max=settings.short_term_holding_days_max,
        risk_flags={"reasons": reasons},
        metrics={
            "atr": round(atr, 4),
            "reference_entry": round(reference_entry, 2),
            "risk_per_share": round(risk_per_share, 2),
        },
    )
