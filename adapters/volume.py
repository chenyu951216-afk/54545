from __future__ import annotations

from datetime import datetime, timezone

from tw_stock_ai.adapters.base import AdapterFetchRequest, AdapterFetchResult, VolumeDataAdapter
from tw_stock_ai.adapters.http_utils import (
    HttpFetchError,
    build_url,
    http_get_json,
    parse_float,
    parse_int,
    parse_iso_or_date,
    quote_path,
)
from tw_stock_ai.adapters.unavailable import UnavailableVolumeAdapter
from tw_stock_ai.config import Settings, get_settings


class FugleHistoricalVolumeAdapter(VolumeDataAdapter):
    adapter_name = "fugle_historical_volume"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        if not self.settings.fugle_api_key:
            return self.unavailable_result("fugle_api_key_missing", request=request)

        fetched_at = datetime.now(timezone.utc)
        raw_items: list[dict] = []
        cleaned_items: list[dict] = []
        errors: list[str] = []
        symbols = self._resolve_symbols(request, errors)
        if not symbols:
            return self.unavailable_result("no_symbols_resolved_for_fugle_volume", request=request, detail=";".join(errors))

        for symbol in symbols:
            url = build_url(
                self.settings.fugle_base_url,
                f"historical/candles/{quote_path(symbol)}",
                {
                    "timeframe": "D",
                    "sort": "asc",
                    "fields": "volume,turnover,close",
                    "from": request.start_date.isoformat() if request.start_date else None,
                    "to": request.end_date.isoformat() if request.end_date else None,
                },
            )
            try:
                payload = http_get_json(
                    url,
                    headers={"X-API-KEY": self.settings.fugle_api_key},
                    timeout=self.settings.fugle_timeout_seconds,
                )
            except HttpFetchError as exc:
                errors.append(f"{symbol}:{exc}")
                continue

            candles = payload.get("data")
            if not isinstance(candles, list):
                errors.append(f"{symbol}:missing_data_array")
                continue

            for row in candles:
                trade_date = parse_iso_or_date(str(row.get("date", "")))
                volume = parse_int(str(row.get("volume", "")))
                turnover = parse_float(str(row.get("turnover", "")))
                if trade_date is None or volume is None:
                    errors.append(f"{symbol}:incomplete_volume")
                    continue
                raw_items.append(
                    {
                        "record_key": f"{symbol}:{trade_date.isoformat()}",
                        "source_url": url,
                        "symbol": symbol,
                        "payload": row,
                    }
                )
                cleaned_items.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "volume": volume,
                        "turnover_value": turnover,
                        "source_name": self.adapter_name,
                        "source_url": url,
                        "raw_payload": {"provider": "fugle", **row},
                    }
                )

        return AdapterFetchResult(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            status="ready" if cleaned_items or not errors else "failed",
            fetched_at=fetched_at,
            raw_items=raw_items,
            cleaned_items=cleaned_items,
            detail=None if cleaned_items else "no_volume_rows_loaded",
            errors=errors,
            metadata={"provider": "fugle"},
        )

    def _resolve_symbols(self, request: AdapterFetchRequest, errors: list[str]) -> list[str]:
        if request.symbols:
            return list(request.symbols)
        exchanges = ["TWSE", "TPEx"] if request.market_code.upper() == "ALL" else [request.market_code]
        symbols: list[str] = []
        for exchange in exchanges:
            url = build_url(
                self.settings.fugle_base_url,
                "intraday/tickers",
                {"type": "EQUITY", "exchange": exchange, "isNormal": "true"},
            )
            try:
                payload = http_get_json(
                    url,
                    headers={"X-API-KEY": self.settings.fugle_api_key},
                    timeout=self.settings.fugle_timeout_seconds,
                )
            except HttpFetchError as exc:
                errors.append(f"{exchange}:{exc}")
                continue
            for item in payload.get("data", []):
                symbol = str(item.get("symbol", "")).strip()
                if symbol:
                    symbols.append(symbol)
                    if request.limit and len(symbols) >= request.limit:
                        return symbols
        return symbols

__all__ = ["VolumeDataAdapter", "FugleHistoricalVolumeAdapter", "UnavailableVolumeAdapter"]
