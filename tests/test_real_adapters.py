from __future__ import annotations

from datetime import date

from tw_stock_ai.adapters.base import AdapterFetchRequest
from tw_stock_ai.adapters.fundamentals import TwseMopsListedFundamentalsAdapter
from tw_stock_ai.adapters.market_calendar import TwseHolidayCalendarAdapter
from tw_stock_ai.adapters.news import MopsListedCompanyNewsAdapter
from tw_stock_ai.adapters.price import FugleHistoricalPriceAdapter
from tw_stock_ai.adapters.revenue import MopsListedRevenueAdapter
from tw_stock_ai.adapters.volume import FugleHistoricalVolumeAdapter
from tw_stock_ai.config import Settings


def test_fugle_price_and_volume_adapters_parse_verified_payloads(monkeypatch) -> None:
    def fake_http_get_json(url: str, *, headers=None, timeout=30):  # noqa: ANN001
        assert headers["X-API-KEY"] == "test-key"
        return {
            "symbol": "2330",
            "exchange": "TWSE",
            "market": "TSE",
            "timeframe": "D",
            "data": [
                {
                    "date": "2026-04-17",
                    "open": 850,
                    "high": 860,
                    "low": 845,
                    "close": 858,
                    "volume": 9239321,
                    "turnover": 1234567890,
                }
            ],
        }

    monkeypatch.setattr("tw_stock_ai.adapters.price.http_get_json", fake_http_get_json)
    monkeypatch.setattr("tw_stock_ai.adapters.volume.http_get_json", fake_http_get_json)
    settings = Settings(
        fugle_api_key="test-key",
        price_data_provider="fugle",
        volume_data_provider="fugle",
    )
    request = AdapterFetchRequest(symbols=["2330"], start_date=date(2026, 4, 1), end_date=date(2026, 4, 17))

    price_result = FugleHistoricalPriceAdapter(settings).fetch(request)
    volume_result = FugleHistoricalVolumeAdapter(settings).fetch(request)

    assert price_result.status == "ready"
    assert price_result.cleaned_items[0]["close"] == 858
    assert volume_result.status == "ready"
    assert volume_result.cleaned_items[0]["turnover_value"] == 1234567890


def test_mops_revenue_and_news_adapters_parse_verified_csvs(monkeypatch) -> None:
    def fake_http_get_csv_rows(url: str, *, headers=None, timeout=30):  # noqa: ANN001
        if "t187ap05" in url:
            return (
                [
                    {
                        "出表日期": "1150417",
                        "資料年月": "11503",
                        "公司代號": "1101",
                        "公司名稱": "台泥",
                        "產業別": "水泥工業",
                        "營業收入-當月營收": "12412837",
                        "營業收入-上月比較增減(%)": "44.44",
                        "營業收入-去年同月增減(%)": "-8.96",
                    }
                ],
                ["出表日期", "資料年月", "公司代號"],
            )
        return (
            [
                {
                    "出表日期": "1150419",
                    "發言日期": "1150418",
                    "發言時間": "070003",
                    "公司代號": "1463",
                    "公司名稱": "強盛新",
                    "主旨": "公告公司更名",
                    "符合條款": "第51款",
                    "事實發生日": "1150211",
                    "說明": "測試說明",
                }
            ],
            ["出表日期", "發言日期", "發言時間"],
        )

    monkeypatch.setattr("tw_stock_ai.adapters.revenue.http_get_csv_rows", fake_http_get_csv_rows)
    monkeypatch.setattr("tw_stock_ai.adapters.news.http_get_csv_rows", fake_http_get_csv_rows)
    settings = Settings(
        revenue_data_provider="mops_listed_monthly_revenue",
        news_data_provider="mops_listed_daily_info",
    )

    revenue_result = MopsListedRevenueAdapter(settings).fetch(AdapterFetchRequest(symbols=["1101"]))
    news_result = MopsListedCompanyNewsAdapter(settings).fetch(AdapterFetchRequest(symbols=["1463"]))

    assert revenue_result.status == "ready"
    assert revenue_result.cleaned_items[0]["revenue_month"] == date(2026, 3, 1)
    assert news_result.status == "ready"
    assert news_result.cleaned_items[0]["title"] == "公告公司更名"


def test_twse_mops_fundamentals_adapter_merges_profile_statement_and_valuation(monkeypatch) -> None:
    def fake_http_get_csv_rows(url: str, *, headers=None, timeout=30):  # noqa: ANN001
        if "t187ap03" in url:
            return ([{"出表日期": "1150419", "公司代號": "1101", "公司名稱": "台泥", "公司簡稱": "台泥", "產業別": "水泥工業"}], [])
        if "BWIBBU_ALL" in url:
            return ([{"日期": "1150417", "股票代號": "1101", "股票名稱": "台泥", "本益比": "10", "殖利率(%)": "3.19", "股價淨值比": "1.20"}], [])
        if "t187ap14" in url:
            return ([{"出表日期": "1150419", "公司代號": "1101", "基本每股盈餘(元)": "2.5"}], [])
        if "t187ap06" in url:
            return ([{"出表日期": "1150419", "公司代號": "1101", "營業收入": "100", "營業毛利（毛損）淨額": "25", "營業利益（損失）": "15", "本期淨利（淨損）": "10", "基本每股盈餘（元）": "2.5"}], [])
        return ([{"出表日期": "1150419", "公司代號": "1101", "資產總計": "200", "負債總計": "80", "權益總計": "120"}], [])

    monkeypatch.setattr("tw_stock_ai.adapters.fundamentals.http_get_csv_rows", fake_http_get_csv_rows)
    settings = Settings(fundamentals_data_provider="twse_mops_listed")
    result = TwseMopsListedFundamentalsAdapter(settings).fetch(AdapterFetchRequest(symbols=["1101"]))

    profile_row = next(item for item in result.cleaned_items if item.get("statement_kind") == "security_profile")
    fundamental_row = next(item for item in result.cleaned_items if item.get("statement_kind") == "fundamentals")
    statement_row = next(item for item in result.cleaned_items if item.get("statement_kind") == "financial_statement")

    assert profile_row["symbol_name"] == "台泥"
    assert round(fundamental_row["debt_ratio"], 2) == 40.0
    assert statement_row["gross_profit"] == 25.0


def test_twse_holiday_calendar_adapter_parses_official_json(monkeypatch) -> None:
    def fake_http_get_json(url: str, *, headers=None, timeout=30):  # noqa: ANN001
        assert "date=115" in url
        return {
            "stat": "ok",
            "date": "20260101",
            "title": "115 年市場開休市日期",
            "fields": ["日期", "名稱", "說明"],
            "data": [
                ["2026-01-01", "中華民國開國紀念日", "依規定放假1日。"],
                ["2026-01-02", "國曆新年開始交易日", "國曆新年開始交易。"],
            ],
        }

    monkeypatch.setattr("tw_stock_ai.adapters.market_calendar.http_get_json", fake_http_get_json)
    settings = Settings(market_calendar_provider="twse_holiday_schedule")
    request = AdapterFetchRequest(market_code="TWSE", start_date=date(2026, 1, 1), end_date=date(2026, 1, 3))
    result = TwseHolidayCalendarAdapter(settings).fetch(request)

    assert result.status == "ready"
    january_1 = next(item for item in result.cleaned_items if item["trade_date"] == date(2026, 1, 1))
    january_2 = next(item for item in result.cleaned_items if item["trade_date"] == date(2026, 1, 2))
    assert january_1["is_trading_day"] is False
    assert january_2["is_trading_day"] is True
