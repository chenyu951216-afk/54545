from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tw_stock_ai.adapters.base import AdapterFetchRequest, AdapterFetchResult, MarketCalendarAdapter
from tw_stock_ai.adapters.http_utils import http_get_json
from tw_stock_ai.adapters.unavailable import UnavailableMarketCalendarAdapter
from tw_stock_ai.config import Settings, get_settings


class TwseHolidayCalendarAdapter(MarketCalendarAdapter):
    adapter_name = "twse_holiday_calendar"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        market_code = request.market_code or "TWSE"
        start_date = request.start_date or datetime.now(timezone.utc).date()
        end_date = request.end_date or start_date
        fetched_at = datetime.now(timezone.utc)

        raw_items: list[dict] = []
        holiday_map: dict[str, dict] = {}
        for year in range(start_date.year, end_date.year + 1):
            url = self.settings.twse_holiday_schedule_url.format(year=year - 1911)
            payload = http_get_json(url, timeout=self.settings.mops_timeout_seconds)
            for row in payload.get("data", []):
                if len(row) < 3:
                    continue
                trade_date = datetime.fromisoformat(str(row[0])).date()
                holiday_map[trade_date.isoformat()] = {
                    "name": row[1],
                    "description": row[2],
                    "source_url": url,
                }
                raw_items.append(
                    {
                        "record_key": f"{market_code}:{trade_date.isoformat()}",
                        "source_url": url,
                        "payload": {"date": row[0], "name": row[1], "description": row[2]},
                    }
                )

        cleaned_items: list[dict] = []
        current = start_date
        while current <= end_date:
            holiday = holiday_map.get(current.isoformat())
            holiday_name = holiday["name"] if holiday else None
            explicit_trading_day = bool(
                holiday_name and any(keyword in holiday_name for keyword in ("開始交易", "最後交易日"))
            )
            is_trading_day = explicit_trading_day or (current.weekday() < 5 and holiday is None)
            cleaned_items.append(
                {
                    "market_code": market_code,
                    "trade_date": current,
                    "is_trading_day": is_trading_day,
                    "session_type": "regular" if is_trading_day else "closed",
                    "holiday_name": holiday_name,
                    "source_name": self.adapter_name,
                    "source_url": (
                        holiday["source_url"]
                        if holiday
                        else self.settings.twse_holiday_schedule_url.format(year=current.year - 1911)
                    ),
                    "raw_payload": holiday or {"weekday": current.weekday()},
                }
            )
            current += timedelta(days=1)

        return AdapterFetchResult(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            status="ready",
            fetched_at=fetched_at,
            raw_items=raw_items,
            cleaned_items=cleaned_items,
            detail=None if cleaned_items else "no_market_calendar_rows_loaded",
            metadata={"provider": "twse"},
        )

__all__ = ["MarketCalendarAdapter", "TwseHolidayCalendarAdapter", "UnavailableMarketCalendarAdapter"]
