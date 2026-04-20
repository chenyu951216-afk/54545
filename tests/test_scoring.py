from datetime import date, datetime, timedelta, timezone

from tw_stock_ai.models import FundamentalSnapshot, NewsItem, PriceBar, RevenueSnapshot
from tw_stock_ai.services.scoring import build_candidate_payload


def test_scoring_returns_missing_when_no_bars() -> None:
    payload = build_candidate_payload(symbol="2330", bars=[], fundamental=None, min_bars=20)
    assert payload["status"] == "missing_data"
    assert payload["overall_score"] is None
    assert payload["treasure_status"] == "unavailable"


def _make_bar(index: int, *, volume: int = 800000, volume_step: int = 2000) -> PriceBar:
    trade_date = date(2026, 1, 1) + timedelta(days=index)
    close = 100 + (index * 0.9)
    return PriceBar(
        symbol="2330",
        trade_date=trade_date,
        open=close - 0.8,
        high=close + 1.4,
        low=close - 1.0,
        close=close,
        volume=volume + (index * volume_step),
        source_name="test",
        source_url="https://example.com",
        fetched_at=datetime.now(timezone.utc),
        raw_payload={"symbol_name": "台積電"},
    )


def test_scoring_builds_ready_candidate_for_trending_breakout_stock() -> None:
    bars = [_make_bar(index) for index in range(140)]
    bars[-1].close = 232.0
    bars[-1].high = 233.0
    bars[-1].open = 228.0
    bars[-1].volume = 2000000

    payload = build_candidate_payload(
        symbol="2330",
        bars=bars,
        fundamental=None,
        min_bars=120,
        symbol_name="台積電",
    )

    assert payload["status"] == "ready"
    assert payload["overall_score"] is not None
    assert payload["take_profit_2"] > payload["take_profit_1"] > payload["entry_zone_high"]
    assert payload["risk_reward_ratio"] is not None
    assert payload["risk_reward_ratio"] >= 1.5
    assert payload["holding_days_min"] == 3
    assert payload["holding_days_max"] == 10


def test_scoring_risk_reward_ratio_moves_with_technical_range() -> None:
    bars = [_make_bar(index) for index in range(140)]
    for offset, value in enumerate((205.0, 206.0, 208.0, 209.0, 211.0, 214.0, 216.0, 220.0, 224.0, 232.0), start=130):
        bars[offset].close = value
        bars[offset].open = value - 2.5
        bars[offset].high = value + 3.0
        bars[offset].low = value - 3.5
    bars[-1].volume = 2400000

    payload = build_candidate_payload(
        symbol="2330",
        bars=bars,
        fundamental=None,
        min_bars=120,
        symbol_name="TSMC",
    )

    assert payload["status"] == "ready"
    assert payload["risk_reward_ratio"] is not None
    assert payload["risk_reward_ratio"] != 1.8


def test_scoring_filters_out_illiquid_stock() -> None:
    bars = [_make_bar(index, volume=10000, volume_step=0) for index in range(140)]
    payload = build_candidate_payload(
        symbol="9999",
        bars=bars,
        fundamental=None,
        min_bars=120,
        symbol_name="測試股",
    )

    assert payload["status"] == "filtered_out"
    assert "insufficient_liquidity_volume" in payload["risk_flags"]["reasons"]


def test_scoring_includes_value_columns_alongside_short_term_columns() -> None:
    bars = [_make_bar(index) for index in range(140)]
    bars[-1].close = 232.0
    bars[-1].high = 233.0
    bars[-1].open = 228.0
    bars[-1].volume = 2000000
    fundamental = FundamentalSnapshot(
        symbol="2330",
        snapshot_date=date(2026, 3, 31),
        source_name="test",
        source_url="https://example.com/fundamentals",
        fetched_at=datetime.now(timezone.utc),
        revenue_yoy=20.0,
        revenue_mom=5.0,
        eps=7.5,
        roe=18.0,
        gross_margin=52.0,
        operating_margin=40.0,
        free_cash_flow=1000000000.0,
        debt_ratio=22.0,
        pe_ratio=16.0,
        pb_ratio=1.8,
        dividend_yield=3.2,
        raw_payload={},
    )
    revenue = RevenueSnapshot(
        symbol="2330",
        revenue_month=date(2026, 3, 1),
        monthly_revenue=100000000.0,
        revenue_yoy=20.0,
        revenue_mom=5.0,
        source_name="test",
        source_url="https://example.com/revenue",
        fetched_at=datetime.now(timezone.utc),
        raw_payload={},
    )
    news = [
        NewsItem(
            symbol="2330",
            title="AI server order update",
            source_name="test",
            source_url="https://example.com/news",
            published_at=datetime.now(timezone.utc),
            raw_payload={},
        )
    ]

    payload = build_candidate_payload(
        symbol="2330",
        bars=bars,
        fundamental=fundamental,
        revenue_snapshot=revenue,
        news_items=news,
        min_bars=120,
        symbol_name="TSMC",
    )

    assert payload["status"] == "ready"
    assert payload["value_score"] is not None
    assert payload["growth_score"] is not None
    assert payload["quality_score"] is not None
    assert payload["valuation_score"] is not None
    assert payload["catalyst_score"] is not None
    assert payload["value_summary"] is not None
