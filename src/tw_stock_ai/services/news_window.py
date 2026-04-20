from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tw_stock_ai.config import Settings


def premium_news_window_active(settings: Settings, *, now: datetime | None = None) -> bool:
    if not settings.premium_news_provider_enabled:
        return False

    current = now or datetime.now(ZoneInfo(settings.scheduler_timezone))
    local_time = current.astimezone(ZoneInfo(settings.scheduler_timezone))
    return settings.premium_news_window_start_hour <= local_time.hour < settings.premium_news_window_end_hour
