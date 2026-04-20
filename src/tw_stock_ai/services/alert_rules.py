from __future__ import annotations

from datetime import datetime, timezone

from tw_stock_ai.config import Settings
from tw_stock_ai.models import Holding, NewsItem, PriceBar


def _negative_keywords(settings: Settings) -> list[str]:
    return [item.strip() for item in settings.holding_negative_news_keywords.split(",") if item.strip()]


def build_position_alerts(
    *,
    holding: Holding,
    bars: list[PriceBar],
    trend: dict,
    news_items: list[NewsItem],
    settings: Settings,
) -> list[dict]:
    ordered = sorted(bars, key=lambda item: item.trade_date)
    if not ordered:
        return []

    latest = ordered[-1]
    latest_close = float(latest.close)
    latest_open = float(latest.open)
    alerts: list[dict] = []
    support_lookback = ordered[-settings.holding_support_lookback_days :]
    support_level = min(float(item.low) for item in support_lookback) if support_lookback else None
    avg_volume = trend.get("metrics", {}).get("average_volume_20d") or 0.0
    volume_ratio = trend.get("metrics", {}).get("volume_ratio") or 0.0
    daily_return = ((latest_close / latest_open) - 1.0) if latest_open else 0.0

    if support_level is not None and latest_close < support_level:
        alerts.append(
            {
                "alert_type": "support_break",
                "severity": "high",
                "message": "跌破近期支撐",
                "evidence": {"latest_close": latest_close, "support_level": round(support_level, 2)},
            }
        )

    if holding.custom_stop_loss is not None and latest_close <= holding.custom_stop_loss:
        alerts.append(
            {
                "alert_type": "stop_loss_break",
                "severity": "critical",
                "message": "跌破自訂止損",
                "evidence": {"latest_close": latest_close, "custom_stop_loss": holding.custom_stop_loss},
            }
        )

    if trend.get("trend_status") in {"weakening", "downtrend"}:
        alerts.append(
            {
                "alert_type": "trend_weakening",
                "severity": "medium" if trend.get("trend_status") == "weakening" else "high",
                "message": "趨勢轉弱",
                "evidence": trend.get("metrics", {}),
            }
        )

    if volume_ratio >= settings.holding_volume_anomaly_ratio and daily_return <= settings.holding_distribution_drop_pct:
        alerts.append(
            {
                "alert_type": "volume_price_anomaly",
                "severity": "high",
                "message": "量價異常",
                "evidence": {
                    "daily_return_pct": round(daily_return * 100, 2),
                    "volume_ratio": round(volume_ratio, 2),
                    "average_volume_20d": avg_volume,
                },
            }
        )

    negative_hits = []
    keywords = _negative_keywords(settings)
    for item in news_items:
        matched = [keyword for keyword in keywords if keyword.lower() in item.title.lower()]
        if matched:
            negative_hits.append(
                {
                    "title": item.title,
                    "source_name": item.source_name,
                    "source_url": item.source_url,
                    "published_at": item.published_at.isoformat(),
                    "matched_keywords": matched,
                }
            )
    if negative_hits:
        alerts.append(
            {
                "alert_type": "negative_news",
                "severity": "high",
                "message": "新聞轉壞",
                "evidence": {"matched_news": negative_hits},
            }
        )

    if holding.custom_target_price is not None and latest_close >= holding.custom_target_price:
        alerts.append(
            {
                "alert_type": "take_profit_zone",
                "severity": "medium",
                "message": "達到止盈區",
                "evidence": {"latest_close": latest_close, "custom_target_price": holding.custom_target_price},
            }
        )

    for alert in alerts:
        alert["triggered_at"] = datetime.now(timezone.utc)
    return alerts
