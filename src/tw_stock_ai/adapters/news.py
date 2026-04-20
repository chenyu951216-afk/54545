from __future__ import annotations

from datetime import datetime, timezone

from tw_stock_ai.adapters.base import AdapterFetchRequest, AdapterFetchResult, NewsDataAdapter
from tw_stock_ai.adapters.http_utils import HttpFetchError, build_url, http_get_csv_rows, http_get_json, roc_datetime_to_utc
from tw_stock_ai.adapters.unavailable import UnavailableNewsAdapter
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
    errors: list[str] = []
    details: list[str] = []
    unavailable_reasons: list[str] = []
    for result in results:
        raw_items.extend(result.raw_items)
        cleaned_items.extend(result.cleaned_items)
        errors.extend(result.errors)
        if result.detail:
            details.append(f"{result.adapter_name}:{result.detail}")
        if result.unavailable_reason:
            unavailable_reasons.append(result.unavailable_reason)

    return AdapterFetchResult(
        adapter_name=adapter_name,
        dataset=dataset,
        status=status,
        fetched_at=fetched_at,
        raw_items=raw_items,
        cleaned_items=cleaned_items,
        detail=" | ".join(details) if details else None,
        unavailable_reason=",".join(sorted(set(unavailable_reasons))) or None,
        errors=errors,
        metadata=metadata or {},
    )


class MopsCompanyNewsAdapterBase(NewsDataAdapter):
    adapter_name = "mops_company_news"
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
            if symbol_filter and symbol not in symbol_filter:
                continue
            published_at = roc_datetime_to_utc(row.get("發言日期"), row.get("發言時間"))
            if published_at is None:
                continue
            if request.start_date and published_at.date() < request.start_date:
                continue
            if request.end_date and published_at.date() > request.end_date:
                continue
            title = str(row.get("主旨", "")).strip()
            if not title:
                continue

            raw_items.append(
                {
                    "record_key": f"{symbol}:{row.get('發言日期')}:{row.get('發言時間')}",
                    "source_url": self.source_url,
                    "symbol": symbol,
                    "payload": row,
                }
            )
            cleaned_items.append(
                {
                    "symbol": symbol or None,
                    "title": title,
                    "source_name": self.adapter_name,
                    "source_url": self.source_url,
                    "published_at": published_at,
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
            detail=None if cleaned_items else "no_news_rows_loaded",
            metadata={"provider": "mops", "scope": self.source_scope},
        )


class MopsListedCompanyNewsAdapter(MopsCompanyNewsAdapterBase):
    adapter_name = "mops_listed_company_news"
    source_scope = "listed_companies"

    @property
    def source_url(self) -> str:
        return self.settings.mops_listed_daily_info_url


class MopsOtcCompanyNewsAdapter(MopsCompanyNewsAdapterBase):
    adapter_name = "mops_otc_company_news"
    source_scope = "otc_companies"

    @property
    def source_url(self) -> str:
        return self.settings.mops_otc_daily_info_url


class MopsAllCompanyNewsAdapter(NewsDataAdapter):
    adapter_name = "mops_all_company_news"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.listed_adapter = MopsListedCompanyNewsAdapter(self.settings)
        self.otc_adapter = MopsOtcCompanyNewsAdapter(self.settings)

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
        if request.limit:
            merged.cleaned_items = merged.cleaned_items[: request.limit]
            merged.raw_items = merged.raw_items[: request.limit]
        return merged


class FinMindTaiwanStockNewsAdapter(NewsDataAdapter):
    adapter_name = "finmind_taiwan_stock_news"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        if not self.settings.finmind_api_token:
            return self.unavailable_result("finmind_api_token_missing", request=request)

        fetched_at = datetime.now(timezone.utc)
        raw_items: list[dict] = []
        cleaned_items: list[dict] = []
        errors: list[str] = []
        symbols = list(dict.fromkeys(request.symbols))
        if not symbols:
            return self.unavailable_result(
                "finmind_news_symbol_scope_required",
                request=request,
                detail="finmind_news_requires_symbols",
            )

        for symbol in symbols:
            params = {
                "dataset": self.settings.finmind_news_dataset,
                "data_id": symbol,
                "start_date": request.start_date.isoformat() if request.start_date else None,
                "end_date": request.end_date.isoformat() if request.end_date else None,
            }
            try:
                payload = http_get_json(
                    build_url(self.settings.finmind_api_base_url, params=params),
                    headers={"Authorization": f"Bearer {self.settings.finmind_api_token}"},
                    timeout=self.settings.finmind_timeout_seconds,
                )
            except HttpFetchError as exc:
                errors.append(f"{symbol}:{exc}")
                continue

            for row in payload.get("data", []):
                published_text = str(row.get("date", "")).strip()
                if not published_text:
                    continue
                try:
                    published_at = datetime.fromisoformat(published_text.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                title = str(row.get("title", "")).strip()
                if not title:
                    continue
                source_name = str(row.get("source", "")).strip() or self.adapter_name
                link = str(row.get("link", "")).strip() or self.settings.finmind_api_base_url
                raw_items.append(
                    {
                        "record_key": f"{symbol}:{published_at.isoformat()}:{link}",
                        "source_url": self.settings.finmind_api_base_url,
                        "symbol": symbol,
                        "payload": row,
                    }
                )
                cleaned_items.append(
                    {
                        "symbol": symbol,
                        "title": title,
                        "source_name": source_name,
                        "source_url": link,
                        "published_at": published_at,
                        "raw_payload": {
                            "provider": "finmind",
                            "dataset": self.settings.finmind_news_dataset,
                            **row,
                        },
                    }
                )
                if request.limit and len(cleaned_items) >= request.limit:
                    break
            if request.limit and len(cleaned_items) >= request.limit:
                break

        return AdapterFetchResult(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            status="ready" if cleaned_items else ("failed" if errors else "unavailable"),
            fetched_at=fetched_at,
            raw_items=raw_items,
            cleaned_items=cleaned_items,
            detail=None if cleaned_items else "no_finmind_news_rows_loaded",
            unavailable_reason=None if cleaned_items else "finmind_news_unavailable",
            errors=errors,
            metadata={"provider": "finmind", "dataset": self.settings.finmind_news_dataset},
        )


class HybridTaiwanMarketNewsAdapter(NewsDataAdapter):
    adapter_name = "hybrid_taiwan_market_news"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.mops_adapter = MopsAllCompanyNewsAdapter(self.settings)
        self.finmind_adapter = FinMindTaiwanStockNewsAdapter(self.settings)

    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        fetched_at = datetime.now(timezone.utc)
        results = [self.mops_adapter.fetch(request)]
        if request.symbols:
            results.append(self.finmind_adapter.fetch(request))
        merged = _merge_results(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            fetched_at=fetched_at,
            results=results,
            metadata={"provider": "hybrid", "scope": "mops_plus_finmind"},
        )
        merged.cleaned_items.sort(
            key=lambda item: (item.get("published_at") or datetime.min.replace(tzinfo=timezone.utc), item.get("symbol") or ""),
            reverse=True,
        )
        if request.limit:
            merged.cleaned_items = merged.cleaned_items[: request.limit]
            merged.raw_items = merged.raw_items[: request.limit]
        return merged


__all__ = [
    "NewsDataAdapter",
    "MopsListedCompanyNewsAdapter",
    "MopsOtcCompanyNewsAdapter",
    "MopsAllCompanyNewsAdapter",
    "FinMindTaiwanStockNewsAdapter",
    "HybridTaiwanMarketNewsAdapter",
    "UnavailableNewsAdapter",
]
