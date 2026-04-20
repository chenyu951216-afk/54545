from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tw_stock_ai.models import PriceBar, SecurityProfile


def resolve_symbol_name(session: Session, symbol: str, bars: list[PriceBar]) -> str | None:
    profile = session.scalar(select(SecurityProfile).where(SecurityProfile.symbol == symbol))
    if profile and profile.name:
        return profile.name

    ordered = sorted(bars, key=lambda item: item.trade_date, reverse=True)
    for bar in ordered:
        payload = bar.raw_payload or {}
        for key in ("symbol_name", "name", "stock_name", "company_name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None
