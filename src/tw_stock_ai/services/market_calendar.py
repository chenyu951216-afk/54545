from __future__ import annotations

from datetime import date

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tw_stock_ai.models import MarketCalendarDay


class MarketCalendarService:
    def get_day(self, session: Session, *, trade_date: date, market_code: str = "TWSE") -> dict:
        record = session.scalar(
            select(MarketCalendarDay)
            .where(
                MarketCalendarDay.trade_date == trade_date,
                MarketCalendarDay.market_code == market_code,
            )
            .order_by(desc(MarketCalendarDay.fetched_at))
        )
        if record is None:
            return {
                "status": "unavailable",
                "trade_date": trade_date.isoformat(),
                "market_code": market_code,
                "is_trading_day": None,
                "reason": "market_calendar_unavailable",
            }
        return {
            "status": "ready",
            "trade_date": trade_date.isoformat(),
            "market_code": record.market_code,
            "is_trading_day": record.is_trading_day,
            "session_type": record.session_type,
            "holiday_name": record.holiday_name,
            "source_name": record.source_name,
            "source_url": record.source_url,
            "fetched_at": record.fetched_at.isoformat(),
        }
