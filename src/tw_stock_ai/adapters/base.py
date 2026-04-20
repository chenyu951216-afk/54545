from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, BinaryIO

from tw_stock_ai.schemas import ImportResult


DATASET_PRICE = "price"
DATASET_VOLUME = "volume"
DATASET_NEWS = "news"
DATASET_REVENUE = "revenue"
DATASET_FUNDAMENTALS = "fundamentals"
DATASET_MARKET_CALENDAR = "market_calendar"


@dataclass(slots=True)
class AdapterFetchRequest:
    symbols: list[str] = field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    market_code: str = "TWSE"
    force_refresh: bool = False
    cache_ttl_seconds: int = 900
    limit: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def cache_key(self) -> str:
        symbols_key = ",".join(sorted(self.symbols))
        return "|".join(
            [
                self.market_code,
                symbols_key,
                self.start_date.isoformat() if self.start_date else "",
                self.end_date.isoformat() if self.end_date else "",
                str(self.limit or ""),
            ]
        )


@dataclass(slots=True)
class AdapterFetchResult:
    adapter_name: str
    dataset: str
    status: str
    fetched_at: datetime
    raw_items: list[dict[str, Any]] = field(default_factory=list)
    cleaned_items: list[dict[str, Any]] = field(default_factory=list)
    detail: str | None = None
    from_cache: bool = False
    unavailable_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def records_received(self) -> int:
        return len(self.raw_items)

    @property
    def records_cleaned(self) -> int:
        return len(self.cleaned_items)


class PriceIngestAdapter(ABC):
    adapter_name: str

    @abstractmethod
    def ingest(self, file_obj: BinaryIO) -> ImportResult:
        raise NotImplementedError


class BaseDataAdapter(ABC):
    adapter_name: str
    dataset: str

    @abstractmethod
    def fetch(self, request: AdapterFetchRequest) -> AdapterFetchResult:
        raise NotImplementedError

    def unavailable_result(
        self,
        reason: str,
        *,
        request: AdapterFetchRequest | None = None,
        detail: str | None = None,
    ) -> AdapterFetchResult:
        metadata = request.metadata if request else {}
        return AdapterFetchResult(
            adapter_name=self.adapter_name,
            dataset=self.dataset,
            status="unavailable",
            fetched_at=datetime.now(timezone.utc),
            detail=detail or reason,
            unavailable_reason=reason,
            metadata=metadata,
        )

    def cache_expiry(self, request: AdapterFetchRequest) -> datetime:
        return datetime.now(timezone.utc) + timedelta(seconds=request.cache_ttl_seconds)


class PriceDataAdapter(BaseDataAdapter):
    dataset = DATASET_PRICE


class VolumeDataAdapter(BaseDataAdapter):
    dataset = DATASET_VOLUME


class NewsDataAdapter(BaseDataAdapter):
    dataset = DATASET_NEWS


class RevenueDataAdapter(BaseDataAdapter):
    dataset = DATASET_REVENUE


class FundamentalsDataAdapter(BaseDataAdapter):
    dataset = DATASET_FUNDAMENTALS


class MarketCalendarAdapter(BaseDataAdapter):
    dataset = DATASET_MARKET_CALENDAR
