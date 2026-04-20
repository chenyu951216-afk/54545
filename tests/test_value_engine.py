from __future__ import annotations

from datetime import date, datetime, timezone

from tw_stock_ai.config import get_settings
from tw_stock_ai.models import FundamentalSnapshot, NewsItem, RevenueSnapshot
from tw_stock_ai.services.value_engine import build_value_payload


def test_value_engine_builds_scores_when_factors_are_available() -> None:
    settings = get_settings()
    fundamental = FundamentalSnapshot(
        symbol="2330",
        snapshot_date=date(2026, 3, 31),
        source_name="test",
        source_url="https://example.com/fundamentals",
        fetched_at=datetime.now(timezone.utc),
        revenue_yoy=22.0,
        revenue_mom=6.0,
        eps=8.5,
        roe=19.0,
        gross_margin=53.0,
        operating_margin=41.0,
        free_cash_flow=1000000000.0,
        debt_ratio=24.0,
        pe_ratio=17.0,
        pb_ratio=1.9,
        dividend_yield=3.5,
        raw_payload={},
    )
    revenue = RevenueSnapshot(
        symbol="2330",
        revenue_month=date(2026, 3, 1),
        monthly_revenue=100000000.0,
        revenue_yoy=22.0,
        revenue_mom=6.0,
        source_name="test",
        source_url="https://example.com/revenue",
        fetched_at=datetime.now(timezone.utc),
        raw_payload={},
    )
    news = [
        NewsItem(
            symbol="2330",
            title="AI server expansion and new order update",
            source_name="test",
            source_url="https://example.com/news",
            published_at=datetime.now(timezone.utc),
            raw_payload={},
        )
    ]

    payload = build_value_payload(
        fundamental=fundamental,
        revenue_snapshot=revenue,
        news_items=news,
        settings=settings,
    )

    assert payload["treasure_status"] == "ready"
    assert payload["value_score"] is not None
    assert payload["growth_score"] is not None
    assert payload["quality_score"] is not None
    assert payload["valuation_score"] is not None
    assert payload["catalyst_score"] is not None
    assert payload["value_summary"]
    assert isinstance(payload["value_risks"]["reasons"], list)


def test_value_engine_marks_unavailable_when_factors_are_insufficient() -> None:
    settings = get_settings()
    payload = build_value_payload(
        fundamental=None,
        revenue_snapshot=None,
        news_items=[],
        settings=settings,
    )

    assert payload["treasure_status"] == "unavailable"
    assert payload["value_score"] is None
    assert "insufficient_treasure_factors" in payload["value_risks"]["reasons"]
