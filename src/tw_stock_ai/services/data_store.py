from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from tw_stock_ai.adapters.base import (
    DATASET_FUNDAMENTALS,
    DATASET_MARKET_CALENDAR,
    DATASET_NEWS,
    DATASET_PRICE,
    DATASET_REVENUE,
    DATASET_VOLUME,
    AdapterFetchResult,
)
from tw_stock_ai.models import (
    DailyVolume,
    FinancialStatementSnapshot,
    FundamentalSnapshot,
    MarketCalendarDay,
    NewsItem,
    PriceBar,
    RawDataRecord,
    RevenueSnapshot,
    SecurityProfile,
)
from tw_stock_ai.services.serialization import to_jsonable


class DataStoreService:
    def persist_raw(self, session: Session, result: AdapterFetchResult) -> int:
        stored = 0
        for index, item in enumerate(result.raw_items):
            session.add(
                RawDataRecord(
                    adapter_name=result.adapter_name,
                    dataset=result.dataset,
                    record_key=item.get("record_key", f"{result.dataset}:{index}"),
                    source_url=item.get("source_url"),
                    fetched_at=result.fetched_at,
                    payload=to_jsonable(item),
                )
            )
            stored += 1
        session.flush()
        return stored

    def persist_cleaned(self, session: Session, result: AdapterFetchResult) -> int:
        handlers = {
            DATASET_PRICE: self._persist_price,
            DATASET_VOLUME: self._persist_volume,
            DATASET_NEWS: self._persist_news,
            DATASET_REVENUE: self._persist_revenue,
            DATASET_FUNDAMENTALS: self._persist_fundamentals,
            DATASET_MARKET_CALENDAR: self._persist_market_calendar,
        }
        handler = handlers.get(result.dataset)
        if handler is None:
            return 0
        return handler(session, result)

    def _persist_price(self, session: Session, result: AdapterFetchResult) -> int:
        stored = 0
        for item in result.cleaned_items:
            existing = session.scalar(
                select(PriceBar).where(
                    PriceBar.symbol == item["symbol"],
                    PriceBar.trade_date == item["trade_date"],
                    PriceBar.source_name == item["source_name"],
                )
            )
            if existing is not None:
                existing.open = item["open"]
                existing.high = item["high"]
                existing.low = item["low"]
                existing.close = item["close"]
                existing.volume = item["volume"]
                existing.source_url = item["source_url"]
                existing.fetched_at = result.fetched_at
                existing.raw_payload = to_jsonable(item.get("raw_payload", {}))
            else:
                session.add(
                    PriceBar(
                        symbol=item["symbol"],
                        trade_date=item["trade_date"],
                        open=item["open"],
                        high=item["high"],
                        low=item["low"],
                        close=item["close"],
                        volume=item["volume"],
                        source_name=item["source_name"],
                        source_url=item["source_url"],
                        fetched_at=result.fetched_at,
                        raw_payload=to_jsonable(item.get("raw_payload", {})),
                    )
                )
            stored += 1
        session.flush()
        return stored

    def _persist_volume(self, session: Session, result: AdapterFetchResult) -> int:
        stored = 0
        for item in result.cleaned_items:
            existing = session.scalar(
                select(DailyVolume).where(
                    DailyVolume.symbol == item["symbol"],
                    DailyVolume.trade_date == item["trade_date"],
                    DailyVolume.source_name == item["source_name"],
                )
            )
            if existing is not None:
                existing.volume = item["volume"]
                existing.turnover_value = item.get("turnover_value")
                existing.source_url = item["source_url"]
                existing.fetched_at = result.fetched_at
                existing.raw_payload = to_jsonable(item.get("raw_payload", {}))
            else:
                session.add(
                    DailyVolume(
                        symbol=item["symbol"],
                        trade_date=item["trade_date"],
                        volume=item["volume"],
                        turnover_value=item.get("turnover_value"),
                        source_name=item["source_name"],
                        source_url=item["source_url"],
                        fetched_at=result.fetched_at,
                        raw_payload=to_jsonable(item.get("raw_payload", {})),
                    )
                )
            stored += 1
        session.flush()
        return stored

    def _persist_news(self, session: Session, result: AdapterFetchResult) -> int:
        stored = 0
        for item in result.cleaned_items:
            session.add(
                NewsItem(
                    symbol=item.get("symbol"),
                    title=item["title"],
                    source_name=item["source_name"],
                    source_url=item["source_url"],
                    published_at=item["published_at"],
                    raw_payload=to_jsonable(item.get("raw_payload", {})),
                )
            )
            stored += 1
        session.flush()
        return stored

    def _persist_revenue(self, session: Session, result: AdapterFetchResult) -> int:
        stored = 0
        for item in result.cleaned_items:
            existing = session.scalar(
                select(RevenueSnapshot).where(
                    RevenueSnapshot.symbol == item["symbol"],
                    RevenueSnapshot.revenue_month == item["revenue_month"],
                    RevenueSnapshot.source_name == item["source_name"],
                )
            )
            if existing is not None:
                existing.monthly_revenue = item.get("monthly_revenue")
                existing.revenue_yoy = item.get("revenue_yoy")
                existing.revenue_mom = item.get("revenue_mom")
                existing.source_url = item["source_url"]
                existing.fetched_at = result.fetched_at
                existing.raw_payload = to_jsonable(item.get("raw_payload", {}))
            else:
                session.add(
                    RevenueSnapshot(
                        symbol=item["symbol"],
                        revenue_month=item["revenue_month"],
                        monthly_revenue=item.get("monthly_revenue"),
                        revenue_yoy=item.get("revenue_yoy"),
                        revenue_mom=item.get("revenue_mom"),
                        source_name=item["source_name"],
                        source_url=item["source_url"],
                        fetched_at=result.fetched_at,
                        raw_payload=to_jsonable(item.get("raw_payload", {})),
                    )
                )
            stored += 1
        session.flush()
        return stored

    def _persist_fundamentals(self, session: Session, result: AdapterFetchResult) -> int:
        stored = 0
        for item in result.cleaned_items:
            kind = item.get("statement_kind", "fundamentals")
            if kind == "security_profile":
                existing_profile = session.scalar(
                    select(SecurityProfile).where(SecurityProfile.symbol == item["symbol"])
                )
                if existing_profile is not None:
                    existing_profile.name = item.get("symbol_name") or item.get("company_name")
                    existing_profile.market = item.get("market", existing_profile.market or "TWSE")
                    existing_profile.industry = item.get("industry")
                    existing_profile.source_name = item["source_name"]
                    existing_profile.source_url = item["source_url"]
                    existing_profile.fetched_at = result.fetched_at
                    existing_profile.raw_payload = to_jsonable(item.get("raw_payload", {}))
                else:
                    session.add(
                        SecurityProfile(
                            symbol=item["symbol"],
                            name=item.get("symbol_name") or item.get("company_name"),
                            market=item.get("market", "TWSE"),
                            industry=item.get("industry"),
                            source_name=item["source_name"],
                            source_url=item["source_url"],
                            fetched_at=result.fetched_at,
                            raw_payload=to_jsonable(item.get("raw_payload", {})),
                        )
                    )
            elif kind == "financial_statement":
                existing_statement = session.scalar(
                    select(FinancialStatementSnapshot).where(
                        FinancialStatementSnapshot.symbol == item["symbol"],
                        FinancialStatementSnapshot.statement_date == item["statement_date"],
                        FinancialStatementSnapshot.source_name == item["source_name"],
                    )
                )
                if existing_statement is not None:
                    existing_statement.period_type = item.get("period_type", "quarterly")
                    existing_statement.revenue = item.get("revenue")
                    existing_statement.gross_profit = item.get("gross_profit")
                    existing_statement.operating_income = item.get("operating_income")
                    existing_statement.net_income = item.get("net_income")
                    existing_statement.eps = item.get("eps")
                    existing_statement.source_url = item["source_url"]
                    existing_statement.fetched_at = result.fetched_at
                    existing_statement.raw_payload = to_jsonable(item.get("raw_payload", {}))
                else:
                    session.add(
                        FinancialStatementSnapshot(
                            symbol=item["symbol"],
                            statement_date=item["statement_date"],
                            period_type=item.get("period_type", "quarterly"),
                            revenue=item.get("revenue"),
                            gross_profit=item.get("gross_profit"),
                            operating_income=item.get("operating_income"),
                            net_income=item.get("net_income"),
                            eps=item.get("eps"),
                            source_name=item["source_name"],
                            source_url=item["source_url"],
                            fetched_at=result.fetched_at,
                            raw_payload=to_jsonable(item.get("raw_payload", {})),
                        )
                    )
            else:
                existing = session.scalar(
                    select(FundamentalSnapshot).where(
                        FundamentalSnapshot.symbol == item["symbol"],
                        FundamentalSnapshot.snapshot_date == item["snapshot_date"],
                        FundamentalSnapshot.source_name == item["source_name"],
                    )
                )
                if existing is not None:
                    existing.revenue_yoy = item.get("revenue_yoy")
                    existing.revenue_mom = item.get("revenue_mom")
                    existing.eps = item.get("eps")
                    existing.roe = item.get("roe")
                    existing.gross_margin = item.get("gross_margin")
                    existing.operating_margin = item.get("operating_margin")
                    existing.free_cash_flow = item.get("free_cash_flow")
                    existing.debt_ratio = item.get("debt_ratio")
                    existing.pe_ratio = item.get("pe_ratio")
                    existing.pb_ratio = item.get("pb_ratio")
                    existing.dividend_yield = item.get("dividend_yield")
                    existing.source_url = item["source_url"]
                    existing.fetched_at = result.fetched_at
                    existing.raw_payload = to_jsonable(item.get("raw_payload", {}))
                else:
                    session.add(
                        FundamentalSnapshot(
                            symbol=item["symbol"],
                            snapshot_date=item["snapshot_date"],
                            source_name=item["source_name"],
                            source_url=item["source_url"],
                            fetched_at=result.fetched_at,
                            revenue_yoy=item.get("revenue_yoy"),
                            revenue_mom=item.get("revenue_mom"),
                            eps=item.get("eps"),
                            roe=item.get("roe"),
                            gross_margin=item.get("gross_margin"),
                            operating_margin=item.get("operating_margin"),
                            free_cash_flow=item.get("free_cash_flow"),
                            debt_ratio=item.get("debt_ratio"),
                            pe_ratio=item.get("pe_ratio"),
                            pb_ratio=item.get("pb_ratio"),
                            dividend_yield=item.get("dividend_yield"),
                            raw_payload=to_jsonable(item.get("raw_payload", {})),
                        )
                    )
            stored += 1
        session.flush()
        return stored

    def _persist_market_calendar(self, session: Session, result: AdapterFetchResult) -> int:
        stored = 0
        for item in result.cleaned_items:
            existing = session.scalar(
                select(MarketCalendarDay).where(
                    MarketCalendarDay.market_code == item["market_code"],
                    MarketCalendarDay.trade_date == item["trade_date"],
                    MarketCalendarDay.source_name == item["source_name"],
                )
            )
            if existing is not None:
                existing.is_trading_day = item["is_trading_day"]
                existing.session_type = item.get("session_type")
                existing.holiday_name = item.get("holiday_name")
                existing.source_url = item["source_url"]
                existing.fetched_at = result.fetched_at
                existing.raw_payload = to_jsonable(item.get("raw_payload", {}))
            else:
                session.add(
                    MarketCalendarDay(
                        market_code=item["market_code"],
                        trade_date=item["trade_date"],
                        is_trading_day=item["is_trading_day"],
                        session_type=item.get("session_type"),
                        holiday_name=item.get("holiday_name"),
                        source_name=item["source_name"],
                        source_url=item["source_url"],
                        fetched_at=result.fetched_at,
                        raw_payload=to_jsonable(item.get("raw_payload", {})),
                    )
                )
            stored += 1
        session.flush()
        return stored
