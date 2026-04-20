from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from tw_stock_ai.adapters.base import AdapterFetchResult
from tw_stock_ai.models import AdapterCacheEntry
from tw_stock_ai.services.serialization import to_jsonable


class AdapterCacheService:
    def _normalize_datetime(self, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def get(self, session: Session, *, adapter_name: str, cache_key: str) -> AdapterCacheEntry | None:
        entry = session.scalar(
            select(AdapterCacheEntry).where(
                AdapterCacheEntry.adapter_name == adapter_name,
                AdapterCacheEntry.cache_key == cache_key,
            )
        )
        if entry is None:
            return None
        expires_at = self._normalize_datetime(entry.expires_at)
        if expires_at is not None and expires_at < datetime.now(timezone.utc):
            return None
        return entry

    def set(
        self,
        session: Session,
        *,
        adapter_name: str,
        dataset: str,
        cache_key: str,
        result: AdapterFetchResult,
        expires_at: datetime,
    ) -> AdapterCacheEntry:
        entry = session.scalar(
            select(AdapterCacheEntry).where(
                AdapterCacheEntry.adapter_name == adapter_name,
                AdapterCacheEntry.cache_key == cache_key,
            )
        )
        payload = {
            "raw_items": to_jsonable(result.raw_items),
            "cleaned_items": to_jsonable(result.cleaned_items),
            "detail": result.detail,
            "unavailable_reason": result.unavailable_reason,
            "errors": to_jsonable(result.errors),
            "metadata": to_jsonable(result.metadata),
        }
        if entry is None:
            entry = AdapterCacheEntry(
                adapter_name=adapter_name,
                dataset=dataset,
                cache_key=cache_key,
                status=result.status,
                expires_at=expires_at,
                fetched_at=result.fetched_at,
                payload=payload,
            )
            session.add(entry)
        else:
            entry.dataset = dataset
            entry.status = result.status
            entry.expires_at = expires_at
            entry.fetched_at = result.fetched_at
            entry.payload = payload
        session.flush()
        return entry

    def to_result(self, entry: AdapterCacheEntry) -> AdapterFetchResult:
        payload = entry.payload or {}
        return AdapterFetchResult(
            adapter_name=entry.adapter_name,
            dataset=entry.dataset,
            status=entry.status,
            fetched_at=entry.fetched_at,
            raw_items=payload.get("raw_items", []),
            cleaned_items=payload.get("cleaned_items", []),
            detail=payload.get("detail"),
            from_cache=True,
            unavailable_reason=payload.get("unavailable_reason"),
            errors=payload.get("errors", []),
            metadata=payload.get("metadata", {}),
        )
