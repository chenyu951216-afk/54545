from io import BytesIO

from tw_stock_ai.adapters.manual_csv import ManualCsvPriceAdapter


def test_manual_csv_requires_headers() -> None:
    adapter = ManualCsvPriceAdapter()
    result = adapter.ingest(BytesIO(b"symbol,trade_date\n2330,2026-01-01\n"))
    assert result.status == "failed"
    assert result.detail is not None
