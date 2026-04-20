from __future__ import annotations

from datetime import datetime, timezone

from tw_stock_ai.adapters.base import AdapterFetchRequest, AdapterFetchResult, RevenueDataAdapter
from tw_stock_ai.adapters.http_utils import http_get_csv_rows, parse_float, roc_year_month_to_date
from tw_stock_ai.adapters.unavailable import UnavailableRevenueAdapter
from tw_stock_ai.config import Settings, get_settings


def _merge_results(
    *,
    adapter_name: str,
    dataset: str,
    fetched_at: datetime,
    results: list[AdapterFetchResult],
    metadata: dict | None = None,
) -> AdapterFetchResult:
    statuses = [result.status for result in results]
    if any(status == "ready" for status in statuses):
        status = "ready"
    elif any(status == "failed" for status in statuses):
        status = "failed"
    elif any(status == "unavailable" for status in statuses):
        status = "unavailable"
    else:
        status = "ready"

    raw_items: list[dict] = []
    cleaned_items: list[dict] = []
    details: list[str] = []
    errors: list[str] = []
    for result in results:
        raw_items.extend(result.raw_items)
        cleaned_items.extend(result.cleaned_items)
        errors.extend(result.errors)
        if result.detail:
            details.append(f"{result.adapter_name}:{result.detail}")

    return AdapterFetchResult(
        adapter_name=adapter_name,
        dataset=dataset,
        status=status,
        fetched_at=fetched_at,
        raw_items=raw_items,
        cleaned_items=cleaned_items,
        detail=" | ".join(details) if details else None,
        errors=errors,
        metadata=metadata or {},
    )


class MopsRevenueAdapterBase(RevenueDataAdapter):
    adapter_name = "mops_revenue"
    source_scope = "listed_companies"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def source_url(self) -> str:
        raise NotImplementedError

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        fetched_at = datetime.now(timezone.utc)
        rows, fieldnames = http_get_csv_rows(
            self.source_url,
            timeout=self.settings.mops_timeout_seconds,
        )
        raw_items: list[dict] = []
        cleaned_items: list[dict] = []
        symbol_filter = set(request.symbols)

        for row in rows:
            symbol = str(row.get("公司代號", "")).strip()
            if not symbol or (symbol_filter and symbol not in symbol_filter):
                continue
            revenue_month = roc_year_month_to_date(row.get("資料年月"))
            if revenue_month is None:
                continue
            if request.start_date and revenue_month < request.start_date:
                continue
            if request.end_date and revenue_month > request.end_date:
                continue
            raw_items.append(
                {
                    "record_key": f"{symbol}:{revenue_month.isoformat()}",
                    "source_url": self.source_url,
                    "symbol": symbol,
                    "payload": row,
                }
            )
            cleaned_items.append(
                {
                    "symbol": symbol,
                    "revenue_month": revenue_month,
                    "monthly_revenue": parse_float(row.get("營業收入-當月營收")),
                    "revenue_mom": parse_float(row.get("營業收入-上月比較增減(%)")),
                    "revenue_yoy": parse_float(row.get("營業收入-去年同月增減(%)")),
                    "source_name": self.adapter_name,
                    "source_url": self.source_url,
                    "raw_payload": {
                        "provider": "mops",
                        "scope": self.source_scope,
                        "fieldnames": fieldnames,
                        **row,
                    },
                }
            )
            if request.limit and len(cleaned_items) >= request.limit:
                break

        return AdapterFetchResult(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            status="ready",
            fetched_at=fetched_at,
            raw_items=raw_items,
            cleaned_items=cleaned_items,
            detail=None if cleaned_items else "no_revenue_rows_loaded",
            metadata={"provider": "mops", "scope": self.source_scope},
        )


class MopsListedRevenueAdapter(MopsRevenueAdapterBase):
    adapter_name = "mops_listed_revenue"
    source_scope = "listed_companies"

    @property
    def source_url(self) -> str:
        return self.settings.mops_listed_monthly_revenue_url


class MopsOtcRevenueAdapter(MopsRevenueAdapterBase):
    adapter_name = "mops_otc_revenue"
    source_scope = "otc_companies"

    @property
    def source_url(self) -> str:
        return self.settings.mops_otc_monthly_revenue_url


class MopsAllRevenueAdapter(RevenueDataAdapter):
    adapter_name = "mops_all_revenue"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.listed_adapter = MopsListedRevenueAdapter(self.settings)
        self.otc_adapter = MopsOtcRevenueAdapter(self.settings)

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        fetched_at = datetime.now(timezone.utc)
        merged = _merge_results(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            fetched_at=fetched_at,
            results=[
                self.listed_adapter.fetch(request),
                self.otc_adapter.fetch(request),
            ],
            metadata={"provider": "mops", "scope": "listed_and_otc"},
        )
        merged.cleaned_items.sort(key=lambda item: (item["revenue_month"], item["symbol"]), reverse=True)
        if request.limit:
            merged.cleaned_items = merged.cleaned_items[: request.limit]
            merged.raw_items = merged.raw_items[: request.limit]
        return merged


__all__ = [
    "RevenueDataAdapter",
    "MopsListedRevenueAdapter",
    "MopsOtcRevenueAdapter",
    "MopsAllRevenueAdapter",
    "UnavailableRevenueAdapter",
]
