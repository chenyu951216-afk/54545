from __future__ import annotations

from tw_stock_ai.config import Settings


def build_exit_analysis(*, trend_status: str, alerts: list[dict], settings: Settings) -> dict:
    reasons = [alert["alert_type"] for alert in alerts]
    severities = {alert["severity"] for alert in alerts}

    action = "hold"
    confidence = 0.25

    if "critical" in severities or "stop_loss_break" in reasons:
        action = "exit_now"
        confidence = settings.holding_exit_now_confidence_base
    elif "support_break" in reasons and "negative_news" in reasons:
        action = "exit_now"
        confidence = max(confidence, settings.holding_exit_now_confidence_base - 0.05)
    elif "take_profit_zone" in reasons or "volume_price_anomaly" in reasons:
        action = "reduce"
        confidence = settings.holding_reduce_confidence_base
    elif "trend_weakening" in reasons or trend_status in {"weakening", "downtrend"}:
        action = "exit_watch"
        confidence = settings.holding_exit_confidence_base

    if action == "hold" and trend_status in {"strong_uptrend", "uptrend"}:
        confidence = 0.7
    elif action == "hold":
        confidence = 0.45

    return {
        "action": action,
        "confidence": round(min(confidence + (0.05 * max(len(alerts) - 1, 0)), 0.99), 2),
        "reasons": sorted(set(reasons)) if reasons else ["trend_healthy"],
    }
